"""Qwen-Image 2.1 model core — unified text-to-image and image-editing backend.

Mirrors the slice of flux_core's surface that web_server.py uses (load_model,
generate_image, encode_prompt_once, decode_latents_to_preview, the state
flags), so the server swaps cores with `import qwen_core as flux_core` when
launched with --qwen — the same arrangement sd_core.py uses for SDXL.

One pipeline (`QwenImage21Pipeline`) serves every job this backend does: with
no condition image it is text-to-image, with one it is an instruction editor,
with up to MAX_REFERENCE_IMAGES it composes a new scene from all of them. The
text encoder is a Qwen3-VL 8B vision-language model, so a condition image is
read by the *encoder* as well as by the VAE — which is why a reference here is
genuinely read rather than merely denoised from.

Differences from the FLUX cores that callers should expect:

- **Guidance is off by default.** The model is meant to be sampled without
  CFG, so `true_cfg_scale` stays at 1.0 unless the caller supplies a negative
  prompt (SUPPORTS_NEGATIVE_PROMPT). Then, and only then, the guidance value
  becomes the true-CFG scale — which costs a second transformer pass per step.
  Unlike the SDXL core there is no default negative prompt: silently enabling
  CFG would halve throughput on a model tuned not to need it.
- **Up to 10 reference images** (FLUX.2 takes 3, Kontext stitches them, FLUX.1
  img2img takes one). They are passed natively as a list.
- **Native RGBA.** The VAE has 4 output channels, so a generation can carry
  real transparency. `transparent=True` wraps the prompt in the phrasing the
  model card asks for; alpha is kept only when the result actually uses it
  (see _finalize_alpha), so ordinary jobs still hand back plain RGB.
- **No strength.** There is no img2img denoise fraction — conditioning is by
  token, not by partial noising. The parameter is accepted and ignored so the
  server's generic call works unchanged.
- **Masked editing without a mask argument.** The pipeline takes none; the
  model is trained to act on *annotated* images. See _annotate_edit_region.
- **Dimensions snap to 32px** (vae_scale_factor 16, unpatched latents), not
  FLUX's 16.

Requires diffusers with QwenImage21Pipeline (merged 2026-09-18; newer than the
0.40.0 release, so a git-main build).
"""

import os
import time

import torch
from PIL import Image, ImageChops, ImageFilter

from diffusers import QwenImage21Pipeline

# Reuse the sidecar writer so image_manager.py keeps parsing outputs, and the
# preview/degenerate helpers so all three cores agree on what a black frame is
# and on how a preview latent is shrunk.
from flux_core import (save_prompt_file, _is_degenerate_image,  # noqa: F401
                       _infer_latent_grid, _shrink_latents_for_preview)

device = "cuda:0"

DEFAULT_QWEN_MODEL = os.environ.get("QWEN_MODEL", "Qwen/Qwen-Image-2.1")

# Optional LoRA (QwenImage21Pipeline carries diffusers' Qwen LoRA loader): an
# HF repo id or a local .safetensors path, fused after load like the FLUX
# LoRAs so it costs nothing per step.
QWEN_LORA = os.environ.get("QWEN_LORA", "").strip() or None

pipe = None
_model_id = None
_cpu_offload = False

# Latents are unpatched and the VAE compresses 16x, so one token covers a
# 16x16 tile and every dimension must be a multiple of 32.
MULTIPLE_OF = 32

# Condition images are resized to this side length before they reach the text
# encoder and the VAE (the pipeline's `output_resolution`). It is not the
# output size — that comes from width/height — but it does set what each
# reference costs: (res/16)^2 tokens, so 1024 is 4096 tokens per image and ten
# references already make a long sequence. 1024 is the model card's default.
try:
    CONDITION_RESOLUTION = int(os.environ.get("QWEN_CONDITION_RESOLUTION", "1024"))
except (TypeError, ValueError):
    CONDITION_RESOLUTION = 1024

# true_cfg_scale used when a negative prompt is supplied but the caller's
# guidance value is <= 1 — i.e. the UI slider sits where a FLUX config left it
# and would otherwise disable the negative prompt the user just typed.
try:
    DEFAULT_TRUE_CFG = float(os.environ.get("QWEN_TRUE_CFG", "4.0"))
except (TypeError, ValueError):
    DEFAULT_TRUE_CFG = 4.0

# Prefix KV cache: the text and condition-image prefix is encoded once and
# reused across steps (valid because the checkpoint sets causal_condition).
# Large win for multi-reference edits. QWEN_KV_CACHE=0 turns it off — the two
# settings give equally valid but not bit-identical samples, so a seed only
# reproduces at a fixed setting.
USE_KV_CACHE = os.environ.get("QWEN_KV_CACHE", "1").strip().lower() not in ("0", "false", "no")

# The phrasing the model card asks for around a transparency request. The
# model generates RGBA natively, but it has to be told to.
RGBA_PROMPT_PREFIX = "This is an RGBA image with transparency. "
RGBA_PROMPT_SUFFIX = " The image has alpha channel and the background is transparent."

# Prepended when the caller paints a mask; _annotate_edit_region draws the
# outline this sentence refers to.
MASK_PROMPT_HINT = ("Edit only the area outlined in magenta, and leave everything "
                    "outside that outline exactly as it is. ")

# flux_core-compatible state flags (web_server reads these directly).
# _flux_version = 2 gives Qwen the same web-API constraints as FLUX.2 —
# multiple reference images, masked editing, and no strength slider (a
# reference is a conditioning token here, not a partially-noised init image).
# It is a constraints class, not a claim to be FLUX: _model_type_string() and
# /model-info name this backend from the server's _QWEN_ACTIVE flag.
_flux_version = 2
_kontext_enabled = False
_turbo_enabled = False
_schnell_enabled = False
_uncensored_enabled = False
_local_encoder_active = True  # Qwen3-VL runs in-process; there is no remote API

OUTPUT_PREFIX = "qwen"  # qwen_{timestamp}_{hex}.png
SUPPORTS_NEGATIVE_PROMPT = True   # via true CFG, off unless one is supplied
SUPPORTS_INPAINT = True           # annotate + composite; see generate_image
SUPPORTS_TRANSPARENCY = True      # RGBA output, and RGBA references
SUPPORTS_MULTI_REFERENCE = True
MAX_REFERENCE_IMAGES = 10

# Mirrors flux_core so web_server can report the setting uniformly.
_vae_tiling_mode = 'auto'
VAE_TILING_DEFAULT_THRESHOLD_MP = 1.9
try:
    _vae_tiling_threshold_mp = float(
        os.environ.get('VAE_TILING_THRESHOLD_MP', VAE_TILING_DEFAULT_THRESHOLD_MP))
except (TypeError, ValueError):
    _vae_tiling_threshold_mp = VAE_TILING_DEFAULT_THRESHOLD_MP


def model_name():
    return os.path.basename(_model_id or DEFAULT_QWEN_MODEL)


def _set_vae_tiling(width, height):
    """Match the VAE's tiling state to the resolution about to be generated —
    same policy as flux_core._set_vae_tiling. Qwen is 2K-native, so the final
    full-resolution decode is routinely the peak-memory moment of a job and
    'auto' will usually be tiling."""
    if pipe is None:
        return False
    if _vae_tiling_mode == 'always':
        want = True
    elif _vae_tiling_mode == 'off':
        want = False
    else:
        want = (width * height) > (_vae_tiling_threshold_mp * 1_000_000)
    vae = getattr(pipe, 'vae', None)
    if vae is None or not hasattr(vae, 'enable_tiling'):
        return want
    if want:
        vae.enable_tiling()
    elif hasattr(vae, 'disable_tiling'):
        vae.disable_tiling()
    return want


def load_model(model_id=None, quantize_encoder=False, cpu_offload=False,
               vae_tiling='auto', lora=None, **_flux_kwargs):
    """Load Qwen-Image 2.1. Extra flux_core-style kwargs are accepted and
    ignored so the server's generic load call works unchanged.

    Args:
        model_id: HF repo id or local diffusers directory (default
            Qwen/Qwen-Image-2.1, or the QWEN_MODEL env var).
        quantize_encoder: load the Qwen3-VL text encoder 4-bit NF4 while the
            transformer stays bf16. The two together are ~30GB in bf16, which
            leaves nothing on a 32GB card for the activations of a 2K decode —
            and on Windows that overflow is not an OOM but a silent WDDM
            page-out to system RAM that makes every step 10-100x slower (the
            sysmem-fallback note in CLAUDE.md). NF4 puts the encoder at ~5GB,
            for ~20GB total.
        cpu_offload: use diffusers' model CPU offload, which keeps only the
            component in use on the card. Much slower per step, but it is the
            fallback that fits anywhere.
        vae_tiling: 'auto' (default), 'always' or 'off'.
        lora: optional LoRA repo/path, fused after load (or the QWEN_LORA env).
    """
    global pipe, _model_id, _vae_tiling_mode, _cpu_offload

    _model_id = model_id or DEFAULT_QWEN_MODEL
    _cpu_offload = bool(cpu_offload)
    dtype = torch.bfloat16
    t0 = time.perf_counter()
    print(f"Loading Qwen-Image 2.1 ({_model_id})...")
    load_timings = {}

    if quantize_encoder:
        from transformers import (BitsAndBytesConfig as TransformersBnbConfig,
                                  Qwen3VLForConditionalGeneration)
        t_enc = time.perf_counter()
        print("  Loading text encoder (Qwen3-VL 8B, 4-bit NF4)...")
        # bnb modules can't be .to()-moved after load, so pin to GPU 0 here.
        text_encoder = Qwen3VLForConditionalGeneration.from_pretrained(
            _model_id, subfolder="text_encoder", torch_dtype=dtype,
            quantization_config=TransformersBnbConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=dtype),
            device_map={"": 0},
        )
        load_timings['text_encoder'] = time.perf_counter() - t_enc
        print(f"    Text encoder loaded in {load_timings['text_encoder']:.2f}s")

        t_pipe = time.perf_counter()
        pipe = QwenImage21Pipeline.from_pretrained(
            _model_id, text_encoder=text_encoder, torch_dtype=dtype)
        load_timings['pipeline'] = time.perf_counter() - t_pipe
        if not cpu_offload:
            # The quantized encoder is already on the card and refuses to move,
            # so move the movable components one at a time instead of calling
            # pipe.to(device) across the whole pipeline.
            pipe.transformer.to(device)
            pipe.vae.to(device)
    else:
        t_pipe = time.perf_counter()
        pipe = QwenImage21Pipeline.from_pretrained(_model_id, torch_dtype=dtype)
        load_timings['pipeline'] = time.perf_counter() - t_pipe
        if not cpu_offload:
            pipe.to(device)

    if cpu_offload:
        pipe.enable_model_cpu_offload(device=device)

    lora = lora or QWEN_LORA
    if lora:
        t_lora = time.perf_counter()
        print(f"  Loading LoRA {lora}...")
        pipe.load_lora_weights(lora)
        # Fusing folds the LoRA into the base weights so it costs nothing per
        # step — same reasoning as flux_core._fuse_loaded_lora.
        pipe.fuse_lora()
        load_timings['lora'] = time.perf_counter() - t_lora
        print(f"    LoRA fused in {load_timings['lora']:.2f}s")

    if isinstance(vae_tiling, bool):        # accept the old boolean
        vae_tiling = 'always' if vae_tiling else 'off'
    _vae_tiling_mode = vae_tiling

    encoder_note = "NF4 encoder" if quantize_encoder else "bf16 encoder"
    offload_note = ", CPU offload" if cpu_offload else ""
    load_timings['total'] = time.perf_counter() - t0
    print(f"Qwen-Image 2.1 ready in {load_timings['total']:.1f}s "
          f"({model_name()}, bf16 transformer, {encoder_note}{offload_note}, "
          f"VAE tiling: {_vae_tiling_mode})")
    return load_timings


def encode_prompt_once(prompt):
    """Pre-encode the prompt for a batch: the Qwen3-VL encoder is LLM-sized,
    so re-running it once per image is seconds of pure waste.

    Text only. A condition image is encoded *with* the prompt (it occupies
    vision slots in the same sequence), so text-only embeddings cannot be
    reused by a job that has references — generate_image drops them there.
    """
    if pipe is None:
        return None
    try:
        with torch.inference_mode():
            prompt_embeds, prompt_embeds_mask, _ = pipe.encode_prompt(
                prompt=prompt, image=None, device=torch.device(device))
        return {"prompt_embeds": prompt_embeds,
                "prompt_embeds_mask": prompt_embeds_mask}
    except Exception as e:
        print(f"[encode] Qwen pre-encode failed, encoding per call instead: {e}",
              flush=True)
        return None


def load_turbo_lora():
    print("Warning: turbo LoRA is FLUX.2-only, skipping on the Qwen backend")


def load_uncensored_lora():
    print("Warning: the uncensored LoRA is FLUX.1-only, skipping on the Qwen backend")


def compile_pipeline():
    """torch.compile the transformer (lazy: the first generation per resolution
    is slow). Needs triton, so this is Linux-only in practice."""
    pipe.transformer = torch.compile(pipe.transformer, mode="max-autotune", fullgraph=False)
    print("Qwen transformer wrapped with torch.compile")


def decode_latents_to_preview(pipe_obj, latents, height, width, max_size=512):
    """Decode intermediate Qwen 2.1 latents into a PIL preview image.

    Packing here is a plain spatial flatten — (B, h*w, 64) — so unpacking is a
    transpose and a reshape. The grid is recovered from the sequence length
    rather than trusted from height/width, which the pipeline may have snapped.
    Latents are shrunk before the decode so a preview never pays for a full 2K
    VAE pass. Best-effort: returns None on any failure.
    """
    if pipe_obj is None or latents is None:
        return None
    try:
        vae = pipe_obj.vae
        vae_scale_factor = getattr(pipe_obj, "vae_scale_factor", 16)
        with torch.inference_mode():
            lat = latents[:1]
            _, seq, channels = lat.shape
            gh, gw = _infer_latent_grid(seq, height, width, vae_scale_factor)
            lat = lat.transpose(1, 2).reshape(1, channels, gh, gw)
            lat = _shrink_latents_for_preview(lat, vae_scale_factor, max_size)
            lat = lat.to(vae.dtype).unsqueeze(2)  # decoder wants (B, C, frame, H, W)
            latents_mean = torch.tensor(vae.config.latents_mean).view(
                1, vae.config.z_dim, 1, 1, 1).to(lat.device, lat.dtype)
            latents_std = torch.tensor(vae.config.latents_std).view(
                1, vae.config.z_dim, 1, 1, 1).to(lat.device, lat.dtype)
            lat = lat * latents_std + latents_mean
            decoded = vae.decode(lat, return_dict=False)[0][:, :, 0]
        image = pipe_obj.image_processor.postprocess(decoded, output_type="pil")[0]
        if max_size and max(image.size) > max_size:
            ratio = max_size / max(image.size)
            image = image.resize(
                (int(image.width * ratio), int(image.height * ratio)),
                Image.Resampling.LANCZOS)
        return image
    except Exception as e:
        print(f"[preview] Qwen decode failed: {e}", flush=True)
        return None


def _snap(value, minimum=256):
    """Round a dimension to the multiple of 32 the latent grid requires."""
    return max(minimum, int(round(int(value) / MULTIPLE_OF) * MULTIPLE_OF))


def _binary_mask(mask_image, size):
    """The painted mask as a hard black/white L image at `size` (white = edit)."""
    return mask_image.convert("L").resize(size, Image.Resampling.LANCZOS) \
                     .point(lambda v: 255 if v > 127 else 0)


def _annotate_edit_region(image, binary, width_hint):
    """Return a copy of `image` with the painted region ringed in magenta.

    Qwen 2.1 has no mask input: local edits are expressed by *annotating* the
    image — the model is trained on circled and painted regions — so the mask
    the UI paints is turned into an annotation the model can actually read.

    The ring is drawn in a band just OUTSIDE the painted region rather than on
    its border. Everything outside the mask is restored from the original in
    the composite step below, so drawing there guarantees no magenta can
    survive into the result; a ring drawn on the boundary itself would sit in
    the feathered blend and could leave a colored fringe.
    """
    radius = max(3, width_hint // 150)
    # A blur-and-threshold grows the region by roughly `radius` px without
    # PIL's MaxFilter, which only takes small odd kernel sizes.
    grown = binary.filter(ImageFilter.GaussianBlur(radius)).point(
        lambda v: 255 if v > 8 else 0)
    ring = ImageChops.subtract(grown, binary)
    annotated = image.convert("RGBA")
    annotated.paste(Image.new("RGBA", annotated.size, (255, 0, 255, 255)), (0, 0), ring)
    return annotated


def _finalize_alpha(image, transparent):
    """Decide whether a result keeps its alpha channel.

    The VAE always decodes 4 channels, so every generation arrives as RGBA
    even when it is a perfectly ordinary opaque picture. Handing that to the
    rest of the server would make every Qwen output an RGBA PNG and change how
    composites and downstream references behave. Keep the alpha only when it
    carries information — the caller asked for transparency, or the model
    produced some unasked.

    The test is what fraction of the frame is see-through, not the minimum
    alpha: measured over opaque outputs, a decode routinely leaves a handful
    of stray sub-255 pixels (one sample bottomed out at 141) while keeping
    0.00% of the frame below 200, whereas an actually transparent sticker had
    69% of it below 128. A minimum-alpha test flips on the stray pixel.
    """
    if image.mode != "RGBA":
        return image
    if transparent:
        return image
    see_through = sum(image.getchannel("A").histogram()[:200])
    if see_through > 0.005 * image.width * image.height:
        return image
    return image.convert("RGB")


def generate_image(prompt, seed=None, steps=40, width=1024, height=1024,
                   local_encoder=False, input_image=None, strength=0.75,
                   sigmas=None, guidance_scale=None, callback_on_step_end=None,
                   mask_image=None, prompt_embeds_kwargs=None,
                   negative_prompt=None, transparent=False, _retry_depth=0):
    """flux_core.generate_image-compatible Qwen-Image 2.1 generation.

    Returns (PIL image, seed, timings). `input_image` is one PIL image or a
    list of up to MAX_REFERENCE_IMAGES; `strength` is accepted and ignored
    (there is no denoise fraction here). `transparent` asks for RGBA output;
    `negative_prompt` turns on true CFG at `guidance_scale`.
    """
    if pipe is None:
        raise RuntimeError("Model must be loaded before generating")

    refs = []
    if input_image is not None:
        refs = [im for im in (input_image if isinstance(input_image, (list, tuple))
                              else [input_image]) if im is not None]
    if len(refs) > MAX_REFERENCE_IMAGES:
        raise ValueError(f"at most {MAX_REFERENCE_IMAGES} reference images are supported")
    if mask_image is not None and not refs:
        raise ValueError("A masked edit (mask_image) needs an input image.")

    if seed is None:
        seed = torch.randint(0, 2**32, (1,)).item()

    width, height = _snap(width), _snap(height)
    _set_vae_tiling(width, height)

    full_prompt = prompt
    base_image = None
    binary = None
    if mask_image is not None:
        # Masked edit: the model is steered by an annotation, and the promise
        # that unpainted pixels are untouched is kept by compositing, not by
        # the model. Output size follows the source so the composite lines up.
        base_image = refs[0]
        width, height = _snap(base_image.width), _snap(base_image.height)
        binary = _binary_mask(mask_image, base_image.size)
        refs = [_annotate_edit_region(base_image, binary, width)] + refs[1:]
        full_prompt = MASK_PROMPT_HINT + full_prompt

    if transparent and "rgba" not in full_prompt.lower():
        full_prompt = RGBA_PROMPT_PREFIX + full_prompt + RGBA_PROMPT_SUFFIX

    negative = (negative_prompt or "").strip() or None
    if negative:
        # Guidance only means anything here alongside a negative prompt, and
        # only above 1.0 — below that the pipeline would silently ignore the
        # negative prompt the caller went to the trouble of sending.
        true_cfg = guidance_scale if (guidance_scale or 0) > 1 else DEFAULT_TRUE_CFG
    else:
        true_cfg = 1.0

    pipe_kwargs = {
        "negative_prompt": negative,
        "true_cfg_scale": true_cfg,
        "width": width,
        "height": height,
        "num_inference_steps": steps,
        "generator": torch.Generator(device=device).manual_seed(seed),
        "output_resolution": CONDITION_RESOLUTION,
        "use_kv_cache": USE_KV_CACHE,
    }
    if refs:
        pipe_kwargs["image"] = refs
    if sigmas is not None:
        pipe_kwargs["sigmas"] = sigmas
    if callback_on_step_end is not None:
        pipe_kwargs["callback_on_step_end"] = callback_on_step_end

    # Cached embeddings cover the text alone, so they are only valid for a job
    # with no condition images (the encoder interleaves image tokens into the
    # same sequence) and only for the prompt as it was cached — a transparency
    # wrapper or a mask hint has since rewritten it.
    if prompt_embeds_kwargs and not refs and full_prompt == prompt:
        pipe_kwargs.update(prompt_embeds_kwargs)
    else:
        pipe_kwargs["prompt"] = full_prompt

    t0 = time.perf_counter()
    with torch.inference_mode():
        image = pipe(**pipe_kwargs).images[0]
    # The pipeline encodes inside the call, so there is no separate encode span
    # to report — same as the SDXL core, the server shows encoding as 0.
    timings = {"encoding": 0.0, "diffusion": time.perf_counter() - t0}

    if base_image is not None:
        # Restore every pixel outside the painted region from the source. The
        # model re-renders the whole frame, so without this a masked edit
        # drifts faces and backgrounds it was never asked to touch; a feathered
        # edge hides the seam.
        image = image.resize(base_image.size, Image.Resampling.LANCZOS)
        feather = max(4, min(base_image.size) // 128)
        soft = binary.filter(ImageFilter.GaussianBlur(feather))
        source = base_image.convert(image.mode)
        image = Image.composite(image, source, soft)

    image = _finalize_alpha(image, transparent)

    if _is_degenerate_image(image) and _retry_depth == 0:
        # Same guard the FLUX cores use: a NaN or saturated decode is an
        # all-black frame that would otherwise be saved as a success.
        print("Warning: degenerate (all-black) result, retrying once with a new seed",
              flush=True)
        return generate_image(
            prompt, seed=None, steps=steps, width=width, height=height,
            local_encoder=local_encoder, input_image=input_image,
            strength=strength, sigmas=sigmas, guidance_scale=guidance_scale,
            callback_on_step_end=callback_on_step_end, mask_image=mask_image,
            prompt_embeds_kwargs=prompt_embeds_kwargs,
            negative_prompt=negative_prompt, transparent=transparent,
            _retry_depth=1)

    return image, seed, timings
