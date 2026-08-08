"""SDXL model core — uncensored Stable Diffusion XL backend for the web server.

Mirrors the slice of flux_core's surface that web_server.py uses (load_model,
generate_image, encode_prompt_once, decode_latents_to_preview, the state
flags), so the server swaps cores with `import sd_core as flux_core` when
launched with --sdxl. Unlike FLUX, SDXL-family checkpoints have no
instruction-refusal behavior, and community checkpoints are trained
uncensored — this is the permissive generation/img2img option next to the
Kontext editor.

The default checkpoint is LUSTIFY! (photorealistic NSFW/SFW SDXL merge) in
diffusers format; override with the SD_MODEL env var, `--sdxl <repo-or-path>`,
or a local single-file .safetensors path (Civitai downloads load via
from_single_file).

Differences from FLUX callers should expect:
- negative prompts are supported (SUPPORTS_NEGATIVE_PROMPT); when the caller
  passes none, DEFAULT_NEGATIVE_PROMPT is applied because raw SDXL output
  quality depends heavily on one (pass "" explicitly to disable).
- true CFG: guidance defaults to 6.0, not FLUX's distilled 2.5-4.
- one reference image max (img2img); no instruction editing. Masked
  inpainting IS supported via a dedicated inpainting checkpoint
  (SD_INPAINT_MODEL, lazy-loaded on the first masked generation).
"""

import os
import time

import torch
from PIL import Image
from diffusers import (StableDiffusionXLPipeline, StableDiffusionXLImg2ImgPipeline,
                       StableDiffusionXLInpaintPipeline)

# Reuse the sidecar writer so image_manager.py keeps parsing outputs, and the
# same reference-count contract the web API validates against.
from flux_core import save_prompt_file, MAX_REFERENCE_IMAGES  # noqa: F401

device = "cuda:0"

DEFAULT_SDXL_MODEL = os.environ.get("SD_MODEL", "John6666/lustify-sdxl-nsfwsfw-v2-sdxl")

# Dedicated inpainting checkpoint (9-channel UNet, trained for masked edits) —
# loaded lazily on the first masked generation. LUSTIFY v2.0 INPAINTING is the
# official inpainting variant of the default checkpoint.
DEFAULT_SD_INPAINT_MODEL = os.environ.get(
    "SD_INPAINT_MODEL", "andro-flock/LUSTIFY-SDXL-NSFW-checkpoint-v2-0-INPAINTING")

# Quality boilerplate applied when the caller doesn't send a negative prompt.
# SDXL merges assume one; an explicit empty string disables it.
DEFAULT_NEGATIVE_PROMPT = (
    "worst quality, low quality, jpeg artifacts, blurry, deformed, "
    "disfigured, bad anatomy, extra limbs, watermark, signature, text"
)

pipe = None
pipe_img2img = None
pipe_inpaint = None
_model_id = None
# Mirrors flux_core so web_server can report the setting uniformly.
_vae_tiling_mode = 'auto'
VAE_TILING_DEFAULT_THRESHOLD_MP = 1.9
try:
    _vae_tiling_threshold_mp = float(
        os.environ.get('VAE_TILING_THRESHOLD_MP', VAE_TILING_DEFAULT_THRESHOLD_MP))
except (TypeError, ValueError):
    _vae_tiling_threshold_mp = VAE_TILING_DEFAULT_THRESHOLD_MP


def _set_vae_tiling(width, height):
    """Match the VAE's tiling state to the resolution about to be generated —
    same policy as flux_core._set_vae_tiling."""
    if _vae_tiling_mode == 'always':
        want = True
    elif _vae_tiling_mode == 'off':
        want = False
    else:
        want = (width * height) > (_vae_tiling_threshold_mp * 1_000_000)
    for pipeline in (pipe, pipe_img2img, pipe_inpaint):
        vae = getattr(pipeline, 'vae', None) if pipeline is not None else None
        if vae is None or not hasattr(vae, 'enable_tiling'):
            continue
        if want:
            vae.enable_tiling()
        elif hasattr(vae, 'disable_tiling'):
            vae.disable_tiling()
    return want

# flux_core-compatible state flags (web_server reads these directly).
# _flux_version = 1 gives SDXL the same web-API constraints as FLUX.1:
# single reference image, no mask/inpainting.
_flux_version = 1
_kontext_enabled = False
_turbo_enabled = False
_schnell_enabled = False
_uncensored_enabled = True  # the checkpoint itself is uncensored
_local_encoder_active = True

OUTPUT_PREFIX = "sdxl"  # sdxl_{timestamp}_{hex}.png instead of flux1_...
SUPPORTS_NEGATIVE_PROMPT = True
SUPPORTS_INPAINT = True  # masked editing via the dedicated inpainting checkpoint


def model_name():
    return os.path.basename(_model_id or DEFAULT_SDXL_MODEL)


def load_model(model_id=None, vae_tiling='auto', **_flux_kwargs):
    """Load an SDXL checkpoint (HF repo id, local diffusers dir, or single
    .safetensors file). Extra flux_core-style kwargs are accepted and ignored
    so the server's generic load call works unchanged.

    vae_tiling mirrors flux_core: 'auto' (tile only above the threshold),
    'always', or 'off'. SDXL's VAE stays in bf16 here, so the memory pressure is
    lower than on the FLUX cores, but the policy is honored for consistency.
    """
    global pipe, pipe_img2img, _model_id

    _model_id = model_id or DEFAULT_SDXL_MODEL
    t0 = time.perf_counter()
    print(f"Loading SDXL checkpoint {_model_id}...")

    # bf16 throughout: same convention as the FLUX cores. SDXL's known VAE
    # NaN problem is an fp16 range overflow; bf16 has fp32's range, so the
    # VAE is stable without the fp32 upcast dance.
    if os.path.isfile(_model_id):
        pipe = StableDiffusionXLPipeline.from_single_file(
            _model_id, torch_dtype=torch.bfloat16)
    else:
        pipe = StableDiffusionXLPipeline.from_pretrained(
            _model_id, torch_dtype=torch.bfloat16)

    _fix_edm_scheduler(pipe)
    pipe.to(device)

    # img2img shares every component with the txt2img pipeline — no extra VRAM.
    pipe_img2img = StableDiffusionXLImg2ImgPipeline.from_pipe(pipe)

    global _vae_tiling_mode
    if isinstance(vae_tiling, bool):        # accept the old boolean
        vae_tiling = 'always' if vae_tiling else 'off'
    _vae_tiling_mode = vae_tiling

    print(f"SDXL ready in {time.perf_counter() - t0:.1f}s "
          f"({model_name()}, bf16, VAE tiling: {_vae_tiling_mode})")
    return {"pipeline": time.perf_counter() - t0}


def _fix_edm_scheduler(p):
    """Auto-converted Civitai repos (John6666/..., andro-flock/...) sometimes
    ship an EDM scheduler config, but these checkpoints are standard epsilon
    SDXL — EDM's sigma preconditioning mis-scales every denoise step and the
    output is pure noise ("colorful blobs") for ALL generations. Swap in the
    standard SDXL noise schedule. Verified: the default LUSTIFY checkpoint
    produces noise with its shipped EDMDPMSolverMultistep config and clean
    images with Euler Ancestral."""
    if type(p.scheduler).__name__.startswith("EDM"):
        from diffusers import EulerAncestralDiscreteScheduler
        bad = type(p.scheduler).__name__
        p.scheduler = EulerAncestralDiscreteScheduler(
            beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear",
            num_train_timesteps=1000, steps_offset=1, timestep_spacing="leading")
        print(f"  scheduler: replaced {bad} (EDM, wrong for this checkpoint) "
              "with EulerAncestralDiscreteScheduler")


def _get_inpaint_pipe():
    """Load the dedicated inpainting checkpoint on first use.

    It's a separate 9-channel-UNet model, so it can't share the base UNet;
    both stay resident (fine on unified-memory systems). First call downloads
    ~7GB if the checkpoint isn't cached."""
    global pipe_inpaint
    if pipe_inpaint is not None:
        return pipe_inpaint
    t0 = time.perf_counter()
    print(f"Loading SDXL inpainting checkpoint {DEFAULT_SD_INPAINT_MODEL} "
          "(first masked edit; ~7GB download if uncached)...")
    pipe_inpaint = StableDiffusionXLInpaintPipeline.from_pretrained(
        DEFAULT_SD_INPAINT_MODEL, torch_dtype=torch.bfloat16)
    _fix_edm_scheduler(pipe_inpaint)
    pipe_inpaint.to(device)
    print(f"SDXL inpainting ready in {time.perf_counter() - t0:.1f}s")
    return pipe_inpaint


def encode_prompt_once(prompt):
    """SDXL's CLIP encoders are milliseconds per call — batch pre-encoding
    isn't worth the plumbing. None tells the server to encode per-call."""
    return None


def load_turbo_lora():
    print("Warning: turbo LoRA is FLUX.2-only, skipping on the SDXL backend")


def load_uncensored_lora():
    print("SDXL backend: checkpoint is already uncensored, no LoRA needed")


def compile_pipeline():
    """torch.compile the UNet (lazy: first generation per resolution is slow)."""
    global pipe
    pipe.unet = torch.compile(pipe.unet, mode="max-autotune", fullgraph=False)
    print("SDXL UNet wrapped with torch.compile")


def decode_latents_to_preview(pipe_obj, latents, height, width, max_size=512):
    """Decode intermediate SDXL latents (B, 4, h/8, w/8) into a PIL preview.

    Latents are spatially downsampled first so the preview never pays for a
    full-res VAE decode. Best-effort: returns None on any failure.
    """
    if pipe_obj is None or latents is None:
        return None
    try:
        with torch.inference_mode():
            lat = latents[:1].to(pipe_obj.vae.dtype)
            vae_scale = getattr(pipe_obj, "vae_scale_factor", 8)
            max_lat = max(1, max_size // vae_scale)
            if max(lat.shape[-2:]) > max_lat:
                scale = max_lat / max(lat.shape[-2:])
                lat = torch.nn.functional.interpolate(
                    lat, scale_factor=scale, mode="bilinear", align_corners=False)
            decoded = pipe_obj.vae.decode(
                lat / pipe_obj.vae.config.scaling_factor, return_dict=False)[0]
        image = pipe_obj.image_processor.postprocess(decoded, output_type="pil")[0]
        if max_size and max(image.size) > max_size:
            ratio = max_size / max(image.size)
            image = image.resize(
                (int(image.width * ratio), int(image.height * ratio)),
                Image.Resampling.LANCZOS)
        return image
    except Exception as e:
        print(f"[preview] SDXL decode failed: {e}", flush=True)
        return None


def generate_image(prompt, seed=None, steps=25, width=1024, height=1024,
                   local_encoder=False, input_image=None, strength=0.75,
                   sigmas=None, guidance_scale=None, callback_on_step_end=None,
                   mask_image=None, prompt_embeds_kwargs=None,
                   negative_prompt=None):
    """flux_core.generate_image-compatible SDXL generation.

    Returns (PIL image, seed, timings) like the FLUX cores. negative_prompt
    is the SDXL-only extra: None applies DEFAULT_NEGATIVE_PROMPT, "" disables.
    """
    if pipe is None:
        raise RuntimeError("Model must be loaded before generating")
    # Tiling decided per job from the output size (see _set_vae_tiling).
    _set_vae_tiling(width, height)
    if mask_image is not None and input_image is None:
        raise ValueError("Inpainting (mask_image) needs an input image.")

    if isinstance(input_image, (list, tuple)):
        refs = [im for im in input_image if im is not None]
        if len(refs) > 1:
            raise ValueError("SDXL img2img takes a single reference image.")
        input_image = refs[0] if refs else None

    if seed is None:
        seed = torch.randint(0, 2**32, (1,)).item()
    if guidance_scale is None:
        guidance_scale = 6.0  # real CFG — SDXL's sweet spot, not FLUX's 2.5-4
    if negative_prompt is None:
        negative_prompt = DEFAULT_NEGATIVE_PROMPT

    # Snap to multiples of 64: SDXL was trained on 64-aligned resolution
    # buckets (~1MP), and off-bucket sizes cost quality (smearing, doubled
    # limbs). 1360x768 → 1344x768, 800x1248 → 768x1280, etc.
    width = max(512, round(int(width) / 64) * 64)
    height = max(512, round(int(height) / 64) * 64)

    # img2img runs int(steps * strength) denoise steps; keep at least one.
    if input_image is not None and strength is not None:
        min_strength = 1.0 / max(1, steps)
        if strength < min_strength:
            print(f"Warning: strength {strength} too low for {steps} steps, "
                  f"using {min_strength:.2f}.")
            strength = min_strength

    timings = {"encoding": 0}
    common = {
        "prompt": prompt,
        "negative_prompt": negative_prompt or None,
        "generator": torch.Generator(device=device).manual_seed(seed),
        "num_inference_steps": steps,
        "guidance_scale": guidance_scale,
    }
    if callback_on_step_end is not None:
        common["callback_on_step_end"] = callback_on_step_end

    t0 = time.perf_counter()
    with torch.inference_mode():
        if mask_image is not None:
            # Masked edit via the dedicated inpainting checkpoint: white mask
            # areas are repainted from the prompt. Denoising ~0.7 rewrites the
            # masked region while still reading its context; callers can pass
            # strength for subtler edits.
            base = input_image.convert("RGB").resize((width, height))
            mask_l = mask_image.convert("L").resize((width, height))
            image = _get_inpaint_pipe()(
                image=base,
                mask_image=mask_l,
                width=width, height=height,
                strength=strength if strength is not None else 0.7,
                **common,
            ).images[0]
            # The inpaint UNet only *approximately* preserves the unmasked
            # region (everything rides through the VAE and denoiser, so faces
            # and backgrounds drift). Composite the result back onto the
            # source so unmasked pixels are literally the original; a
            # feathered mask edge hides the seam.
            from PIL import ImageFilter
            feather = max(4, min(width, height) // 128)
            image = Image.composite(
                image, base, mask_l.filter(ImageFilter.GaussianBlur(feather)))
        elif input_image is not None:
            image = pipe_img2img(
                image=input_image.convert("RGB").resize((width, height)),
                strength=strength if strength is not None else 0.75,
                **common,
            ).images[0]
        else:
            image = pipe(width=width, height=height, **common).images[0]
    timings["diffusion"] = time.perf_counter() - t0

    return image, seed, timings
