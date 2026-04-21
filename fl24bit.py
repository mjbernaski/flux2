import argparse
import torch
import os
import socket
import logging
import warnings
from PIL import Image

# Suppress CLIP tokenizer truncation warning - expected for long prompts since T5 handles full text
warnings.filterwarnings("ignore", message="Token indices sequence length is longer than the specified maximum sequence length")

# Suppress verbose logging from HTTP and ML libraries
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("requests").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("transformers").setLevel(logging.WARNING)
logging.getLogger("diffusers").setLevel(logging.WARNING)
logging.getLogger("accelerate").setLevel(logging.WARNING)
logging.getLogger("huggingface_hub").setLevel(logging.WARNING)

# Performance optimizations for Blackwell/DGX Spark GPUs
torch.backends.cuda.matmul.allow_tf32 = True  # ~3x faster matmul with minimal precision loss
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True  # Auto-tune convolution algorithms

# DGX Spark / Unified Memory fix: Patch safetensors to avoid double memory allocation
# On unified memory systems, the default copy=True causes memory to double during loading
import safetensors.torch
_original_load_file = safetensors.torch.load_file

def _patched_load_file(filename, device="cpu"):
    result = _original_load_file(filename, device="cpu")
    if device != "cpu":
        result = {k: v.to(device, copy=False) for k, v in result.items()}
    return result

safetensors.torch.load_file = _patched_load_file

# FLUX.1 classes
from diffusers import FluxPipeline, FluxImg2ImgPipeline, FluxTransformer2DModel
# FLUX.2 classes (different architecture - img2img is built into Flux2Pipeline)
from diffusers import Flux2Pipeline, Flux2Transformer2DModel
from huggingface_hub import get_token
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import io
from datetime import datetime
import uuid
import time
import subprocess
import shutil

# FLUX.1 repos
FLUX1_REPO_4BIT = "diffusers/FLUX.1-dev-bnb-4bit"
FLUX1_REPO_FULL = "black-forest-labs/FLUX.1-dev"
FLUX1_REPO_SCHNELL = "black-forest-labs/FLUX.1-schnell"
FLUX1_GGUF_MODELS = {
    "bf16": "https://huggingface.co/city96/FLUX.1-dev-gguf/blob/main/flux1-dev-BF16.gguf",
    "q8": "https://huggingface.co/city96/FLUX.1-dev-gguf/blob/main/flux1-dev-Q8_0.gguf",
    "q4": "https://huggingface.co/city96/FLUX.1-dev-gguf/blob/main/flux1-dev-Q4_K_S.gguf",
}

# FLUX.2 repos
FLUX2_REPO_4BIT = "diffusers/FLUX.2-dev-bnb-4bit"
FLUX2_REPO_FULL = "black-forest-labs/FLUX.2-dev"

# FLUX.2 klein (smaller 9B model, uses Flux2KleinPipeline). Full bf16 only — NVFP4
# variants are blocked by a diffusers upstream bug in the Flux2 single-file converter
# (qkv-chunking assumes unquantized fused weights; NVFP4 scale tensors break it).
FLUX2_KLEIN_REPO_FULL = "black-forest-labs/FLUX.2-klein-9B"
device = "cuda:0"
torch_dtype = torch.bfloat16

# Lazy-loaded model components
transformer = None
pipe = None
pipe_img2img = None  # Img2img pipeline (created on-demand from pipe)
_model_type = None  # Track which model is loaded
_flux_version = 1  # Track FLUX version (1 or 2)
_turbo_enabled = False  # Track if turbo LoRA is loaded
_schnell_enabled = False  # Track if using schnell (4-step) model

# Pre-shifted custom sigmas for 8-step turbo inference (FLUX.2 only)
TURBO_SIGMAS = [1.0, 0.6509, 0.4374, 0.2932, 0.1893, 0.1108, 0.0495, 0.00031]

# Connection pooling with retry strategy for transient failures
_session = requests.Session()
_retry_strategy = Retry(
    total=3,
    backoff_factor=0.5,
    status_forcelist=[500, 502, 503, 504],
    allowed_methods=["POST"],
    raise_on_status=False,
)
_adapter = HTTPAdapter(max_retries=_retry_strategy)
_session.mount("https://", _adapter)

# Embedding cache - avoids redundant API calls for same prompts
_embedding_cache = {}


def _wrap_vae_for_dtype_safety(pipeline):
    """Wrap VAE encode/decode methods to handle dtype mismatches automatically.

    For quantized models (4-bit BNB, GGUF), there can be dtype mismatches:
    - Pipeline uses torch_dtype (bfloat16) for image preprocessing
    - VAE may be in float32
    - Transformer outputs may be in a different dtype

    This wrapper ensures inputs are cast to match VAE dtype before processing,
    and outputs are cast back to the expected dtype.
    """
    if not hasattr(pipeline, 'vae') or pipeline.vae is None:
        return

    vae = pipeline.vae

    # Get VAE's parameter dtype
    try:
        vae_dtype = next(vae.parameters()).dtype
    except StopIteration:
        return

    # Only wrap if VAE dtype differs from torch_dtype
    if vae_dtype == torch_dtype:
        return  # No wrapping needed

    # Store original methods
    original_encode = vae.encode
    original_decode = vae.decode

    def wrapped_encode(x, *args, **kwargs):
        # Cast input to VAE dtype
        if x.dtype != vae_dtype:
            x = x.to(vae_dtype)
        result = original_encode(x, *args, **kwargs)
        return result

    def wrapped_decode(z, *args, **kwargs):
        # Cast latents to VAE dtype
        if z.dtype != vae_dtype:
            z = z.to(vae_dtype)
        result = original_decode(z, *args, **kwargs)
        return result

    # Apply wrappers
    vae.encode = wrapped_encode
    vae.decode = wrapped_decode


def load_model(local_encoder=False, full_model=False, gguf_quant=None, flux2=False, schnell=False, for_lora=False, klein=False):
    """Load the FLUX model components. Call this before generating images.

    Args:
        local_encoder: Use local text encoder instead of remote API
        full_model: Use full FLUX model instead of 4-bit quantized
        gguf_quant: GGUF quantization level ('bf16', 'q8', 'q4') - FLUX.1 only
        flux2: Use FLUX.2 model instead of FLUX.1
        schnell: Use FLUX.1-schnell (fast 4-step model) - FLUX.1 only
        for_lora: Use simple loading path compatible with LoRA (avoids meta tensor issues)
        klein: Use FLUX.2-klein (9B) variant instead of FLUX.2-dev (32B). Implies flux2.
            Klein always loads as full bf16 (no quantized variant wired up — NVFP4 blocked by
            diffusers upstream qkv-chunking bug in the Flux2 single-file converter).
    """
    global transformer, pipe, _model_type, _flux_version, _schnell_enabled
    if pipe is not None:
        return {}  # Already loaded

    # klein implies flux2 + full (bf16). Klein has no working 4-bit/NVFP4 path in current diffusers.
    if klein:
        flux2 = True
        full_model = True

    _flux_version = 2 if flux2 else 1
    _schnell_enabled = schnell and not flux2  # Schnell only for FLUX.1
    flux_name = f"FLUX.{_flux_version}"

    # Schnell only exists as full model (no 4-bit quantized version available)
    if schnell and not flux2:
        full_model = True

    # FLUX.2 uses Mistral3 text encoder - no remote API available, so local encoder is required
    # for 4-bit mode (full model always loads encoder anyway)
    if flux2 and not full_model and not local_encoder:
        print("Note: FLUX.2 4-bit requires local Mistral3 encoder (no remote API available)")
        local_encoder = True

    # Select repos based on version
    if flux2:
        repo_4bit = FLUX2_REPO_4BIT
        repo_full = FLUX2_KLEIN_REPO_FULL if klein else FLUX2_REPO_FULL
        if gguf_quant:
            print("Warning: GGUF not available for FLUX.2, using 4-bit instead")
            gguf_quant = None
        if schnell:
            print("Warning: Schnell not available for FLUX.2, using standard model")
    else:
        repo_4bit = FLUX1_REPO_4BIT
        repo_full = FLUX1_REPO_SCHNELL if schnell else FLUX1_REPO_FULL

    load_timings = {}
    total_start = time.perf_counter()

    # Determine model type
    klein_tag = "-klein" if klein else "-dev"
    if schnell and not flux2:
        _model_type = "schnell"
        model_desc = f"FLUX.1-schnell (4-step)"
    elif gguf_quant:
        _model_type = f"gguf-{gguf_quant}"
        model_desc = f"GGUF {gguf_quant.upper()} {flux_name}"
    elif full_model:
        _model_type = "full-klein" if klein else "full"
        model_desc = f"full {flux_name}{klein_tag}"
    else:
        _model_type = "4bit"
        model_desc = f"4-bit quantized {flux_name}"

    hostname = socket.gethostname()
    print(f"[{hostname}] Loading {model_desc}...")

    if gguf_quant:
        # GGUF models - recommended for DGX Spark unified memory systems (FLUX.1 only)
        from diffusers import GGUFQuantizationConfig

        gguf_url = FLUX1_GGUF_MODELS[gguf_quant]
        print(f"Loading GGUF transformer from {gguf_quant.upper()}...")
        t0 = time.perf_counter()
        transformer = FluxTransformer2DModel.from_single_file(
            gguf_url,
            quantization_config=GGUFQuantizationConfig(compute_dtype=torch_dtype),
            torch_dtype=torch_dtype,
        )
        # Move GGUF transformer to GPU (from_single_file doesn't support device_map)
        transformer = transformer.to(device)
        load_timings['transformer'] = time.perf_counter() - t0
        print(f"  Transformer loaded in {load_timings['transformer']:.2f}s")

        print(f"Loading {flux_name} pipeline...")
        t0 = time.perf_counter()
        if local_encoder:
            pipe = FluxPipeline.from_pretrained(
                repo_full, transformer=transformer, torch_dtype=torch_dtype,
                device_map="balanced"
            )
        else:
            pipe = FluxPipeline.from_pretrained(
                repo_full, transformer=transformer, text_encoder=None,
                text_encoder_2=None, torch_dtype=torch_dtype,
                device_map="balanced"
            )
        load_timings['pipeline'] = time.perf_counter() - t0
        load_timings['to_device'] = 0  # Already on GPU via device_map
        print(f"  Pipeline loaded in {load_timings['pipeline']:.2f}s")

    elif for_lora and not flux2:
        # Simple loading path for LoRA compatibility (FLUX.1 only)
        # Uses DiffusionPipeline directly to avoid meta tensor issues with parallel loading
        from diffusers import DiffusionPipeline

        repo_id = repo_full
        print(f"Loading {flux_name} pipeline (LoRA-compatible mode)...")
        t0 = time.perf_counter()
        pipe = DiffusionPipeline.from_pretrained(
            repo_id,
            torch_dtype=torch_dtype,
        )
        load_timings['pipeline'] = time.perf_counter() - t0
        print(f"  Pipeline loaded in {load_timings['pipeline']:.2f}s")

        print("Moving to GPU...")
        t0 = time.perf_counter()
        pipe = pipe.to(device)
        load_timings['to_device'] = time.perf_counter() - t0
        load_timings['transformer'] = 0  # Included in pipeline load
        print(f"  Moved to GPU in {load_timings['to_device']:.2f}s")

    elif full_model:
        # Full model - load transformer and text encoder in parallel for faster startup
        import concurrent.futures

        repo_id = repo_full
        if flux2:
            print(f"Loading {flux_name} components sequentially (memory-optimized)...")
        else:
            print(f"Loading {flux_name} components in parallel...")
        t0 = time.perf_counter()

        if flux2:
            # FLUX.2-dev uses Mistral3 text encoder; FLUX.2-klein uses Qwen3.
            # Load sequentially (not parallel) to reduce peak memory usage on unified memory systems
            if klein:
                from transformers import Qwen3ForCausalLM
                text_encoder_cls = Qwen3ForCausalLM
                text_encoder_label = "Qwen3"
            else:
                from transformers import Mistral3ForConditionalGeneration
                text_encoder_cls = Mistral3ForConditionalGeneration
                text_encoder_label = "Mistral3"

            param_desc = "9B" if klein else "32B"
            print(f"  Loading transformer ({param_desc} params)...")
            t_trans = time.perf_counter()
            transformer = Flux2Transformer2DModel.from_pretrained(
                repo_id, subfolder="transformer", torch_dtype=torch_dtype,
                device_map="auto", low_cpu_mem_usage=True, use_safetensors=True
            )
            load_timings['transformer'] = time.perf_counter() - t_trans
            print(f"    Transformer loaded in {load_timings['transformer']:.2f}s")

            print(f"  Loading text encoder ({text_encoder_label})...")
            t_enc = time.perf_counter()
            # Don't use device_map="auto" - it can place embedding layer on CPU causing
            # index_select device mismatch errors. Load to CPU then move to GPU.
            text_encoder = text_encoder_cls.from_pretrained(
                repo_id, subfolder="text_encoder", torch_dtype=torch_dtype,
                use_safetensors=True
            ).to(device)
            load_timings['text_encoder'] = time.perf_counter() - t_enc
            print(f"    Text encoder loaded in {load_timings['text_encoder']:.2f}s")

            load_timings['sequential_load'] = time.perf_counter() - t0
            print(f"  Components loaded sequentially in {load_timings['sequential_load']:.2f}s")

            # Assemble FLUX.2 pipeline
            # Note: Use device_map="balanced" (not "cuda" + low_cpu_mem_usage) to avoid
            # meta tensor errors when loading remaining components (VAE, scheduler, etc.)
            if klein:
                from diffusers import Flux2KleinPipeline
                pipeline_cls = Flux2KleinPipeline
            else:
                pipeline_cls = Flux2Pipeline
            print(f"Assembling {pipeline_cls.__name__}...")
            t0 = time.perf_counter()
            pipe = pipeline_cls.from_pretrained(
                repo_id,
                transformer=transformer,
                text_encoder=text_encoder,
                torch_dtype=torch_dtype,
                device_map="balanced",
                use_safetensors=True,
            )
        else:
            # FLUX.1 uses T5 + CLIP text encoders
            from transformers import CLIPTextModel, T5EncoderModel

            def load_transformer():
                # Use device_map="auto" instead of "cuda" + low_cpu_mem_usage to avoid
                # meta tensor errors (especially with schnell model)
                return FluxTransformer2DModel.from_pretrained(
                    repo_id, subfolder="transformer", torch_dtype=torch_dtype,
                    device_map="auto", use_safetensors=True
                )

            def load_text_encoder():
                # T5 doesn't support low_cpu_mem_usage=True with device_map="cuda"
                # (causes meta tensor dispatch error). Use device_map="auto" instead.
                return T5EncoderModel.from_pretrained(
                    repo_id, subfolder="text_encoder_2", torch_dtype=torch_dtype,
                    device_map="auto", use_safetensors=True
                )

            def load_text_encoder_clip():
                # CLIP is small enough to load directly to CUDA
                return CLIPTextModel.from_pretrained(
                    repo_id, subfolder="text_encoder", torch_dtype=torch_dtype,
                    device_map="auto", use_safetensors=True
                )

            # Load heavy components in parallel
            with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
                transformer_future = executor.submit(load_transformer)
                t5_future = executor.submit(load_text_encoder)
                clip_future = executor.submit(load_text_encoder_clip)

                transformer = transformer_future.result()
                text_encoder_2 = t5_future.result()
                text_encoder = clip_future.result()

            load_timings['parallel_load'] = time.perf_counter() - t0
            print(f"  Components loaded in parallel in {load_timings['parallel_load']:.2f}s")

            # Assemble FLUX.1 pipeline
            # Note: Use device_map="balanced" (not "cuda" + low_cpu_mem_usage) to avoid
            # meta tensor errors when loading remaining components (VAE, scheduler, etc.)
            print("Assembling pipeline...")
            t0 = time.perf_counter()
            pipe = FluxPipeline.from_pretrained(
                repo_id,
                transformer=transformer,
                text_encoder=text_encoder,
                text_encoder_2=text_encoder_2,
                torch_dtype=torch_dtype,
                device_map="balanced",
                use_safetensors=True,
            )

        load_timings['transformer'] = 0  # Already counted above
        load_timings['pipeline'] = time.perf_counter() - t0
        load_timings['to_device'] = 0  # Already on GPU via device_map
        print(f"  Pipeline assembled in {load_timings['pipeline']:.2f}s")
    else:
        # 4-bit BNB model
        repo_id = repo_4bit
        print(f"Loading {flux_name} transformer (4-bit)...")
        t0 = time.perf_counter()

        if flux2:
            # FLUX.2 uses different transformer class
            transformer = Flux2Transformer2DModel.from_pretrained(
                repo_id, subfolder="transformer", torch_dtype=torch_dtype
            )
        else:
            transformer = FluxTransformer2DModel.from_pretrained(
                repo_id, subfolder="transformer", torch_dtype=torch_dtype
            )
        load_timings['transformer'] = time.perf_counter() - t0
        print(f"  Transformer loaded in {load_timings['transformer']:.2f}s")

        print(f"Loading {flux_name} pipeline...")
        t0 = time.perf_counter()
        if flux2:
            # FLUX.2 uses different pipeline class
            if local_encoder:
                # Load Mistral3 encoder separately to avoid meta tensor errors
                # (device_map="balanced" with from_pretrained doesn't work well for Mistral3)
                # Note: Don't use device_map="auto" - it can place embedding layer on CPU
                # causing index_select device mismatch errors. Load to CPU then move to GPU.
                from transformers import Mistral3ForConditionalGeneration
                print("Loading local text encoder (Mistral3, requires more VRAM)...")
                text_encoder = Mistral3ForConditionalGeneration.from_pretrained(
                    repo_id, subfolder="text_encoder", torch_dtype=torch_dtype,
                    use_safetensors=True
                ).to(device)
                pipe = Flux2Pipeline.from_pretrained(
                    repo_id, transformer=transformer, text_encoder=text_encoder,
                    torch_dtype=torch_dtype, device_map="balanced"
                )
            else:
                pipe = Flux2Pipeline.from_pretrained(
                    repo_id, transformer=transformer, text_encoder=None, torch_dtype=torch_dtype,
                    device_map="balanced"
                )
        else:
            # FLUX.1 pipeline
            if local_encoder:
                print("Loading local text encoders (this requires more VRAM)...")
                pipe = FluxPipeline.from_pretrained(
                    repo_id, transformer=transformer, torch_dtype=torch_dtype,
                    device_map="balanced"
                )
            else:
                pipe = FluxPipeline.from_pretrained(
                    repo_id, transformer=transformer, text_encoder=None, torch_dtype=torch_dtype,
                    device_map="balanced"
                )
        load_timings['pipeline'] = time.perf_counter() - t0
        load_timings['to_device'] = 0  # Already on GPU via device_map
        print(f"  Pipeline loaded in {load_timings['pipeline']:.2f}s")

    load_timings['total'] = time.perf_counter() - total_start
    print(f"[{hostname}] Model ready in {load_timings['total']:.2f}s (transformer={load_timings['transformer']:.2f}s, pipeline={load_timings['pipeline']:.2f}s)")

    return load_timings


def load_turbo_lora():
    """Load the FLUX.2 turbo LoRA for faster 8-step inference.

    Only works with FLUX.2. Must be called after load_model().
    """
    global _turbo_enabled

    if pipe is None:
        raise RuntimeError("Model must be loaded before loading LoRA")

    if _flux_version != 2:
        print("Warning: Turbo LoRA only available for FLUX.2, skipping")
        return

    if _turbo_enabled:
        print("Turbo LoRA already loaded")
        return

    print("Loading FLUX.2 turbo LoRA (fal/FLUX.2-dev-Turbo)...")
    t0 = time.perf_counter()
    pipe.load_lora_weights(
        "fal/FLUX.2-dev-Turbo",
        weight_name="flux.2-turbo-lora.safetensors"
    )
    load_time = time.perf_counter() - t0
    print(f"  Turbo LoRA loaded in {load_time:.2f}s")
    _turbo_enabled = True


_uncensored_enabled = False  # Track if uncensored LoRA is loaded

def load_uncensored_lora():
    """Load the Flux-Uncensored-V2 LoRA for FLUX.1.

    Only works with FLUX.1. Must be called after load_model().
    """
    global _uncensored_enabled

    if pipe is None:
        raise RuntimeError("Model must be loaded before loading LoRA")

    if _flux_version != 1:
        print("Warning: Uncensored LoRA only available for FLUX.1, skipping")
        return

    if _uncensored_enabled:
        print("Uncensored LoRA already loaded")
        return

    print("Loading Flux-Uncensored-V2 LoRA (enhanceaiteam/Flux-Uncensored-V2)...")
    t0 = time.perf_counter()
    pipe.load_lora_weights("enhanceaiteam/Flux-Uncensored-V2")
    load_time = time.perf_counter() - t0
    print(f"  Uncensored LoRA loaded in {load_time:.2f}s")
    _uncensored_enabled = True


def remote_text_encoder(prompt, use_cache=True):
    if use_cache and prompt in _embedding_cache:
        return _embedding_cache[prompt]

    response = _session.post(
        "https://remote-text-encoder-flux-2.huggingface.co/predict",
        json={"prompt": prompt},
        headers={
            "Authorization": f"Bearer {get_token()}",
            "Content-Type": "application/json"
        },
        timeout=(10, 60),  # (connect timeout, read timeout) in seconds
    )
    response.raise_for_status()
    prompt_embeds = torch.load(io.BytesIO(response.content))
    result = prompt_embeds.to(device)

    if use_cache:
        _embedding_cache[prompt] = result

    return result

def decode_latents_to_preview(pipe_obj, latents, height, width, max_size=512):
    """Decode intermediate Flux latents into a PIL preview image.

    Handles both FLUX.1 (scaling_factor/shift_factor + _unpack_latents) and
    FLUX.2 (batch-norm stats + _unpatchify_latents). Returns None on any
    failure so callers can treat previews as best-effort.
    """
    if pipe_obj is None or latents is None:
        return None
    try:
        vae_scale_factor = getattr(pipe_obj, "vae_scale_factor", 8)
        unpatchify = getattr(pipe_obj, "_unpatchify_latents", None)

        with torch.inference_mode():
            if unpatchify is not None:
                # FLUX.2: latents are (B, H*W, C*4) with contiguous position ids,
                # so unpacking is a reshape+permute. Then apply vae.bn stats and
                # unpatchify before decoding.
                patch_h = int(height) // (vae_scale_factor * 2)
                patch_w = int(width) // (vae_scale_factor * 2)
                B, seq, ch = latents.shape
                if seq != patch_h * patch_w:
                    raise ValueError(
                        f"latent seq {seq} != patch_h*patch_w {patch_h * patch_w}"
                    )
                lat = latents.view(B, patch_h, patch_w, ch).permute(0, 3, 1, 2).contiguous()
                vae = pipe_obj.vae
                bn_mean = vae.bn.running_mean.view(1, -1, 1, 1).to(lat.device, lat.dtype)
                bn_std = torch.sqrt(
                    vae.bn.running_var.view(1, -1, 1, 1) + vae.config.batch_norm_eps
                ).to(lat.device, lat.dtype)
                lat = lat * bn_std + bn_mean
                lat = unpatchify(lat)
                decoded = vae.decode(lat, return_dict=False)[0]
            else:
                # FLUX.1
                unpack = getattr(pipe_obj, "_unpack_latents", None)
                lat = unpack(latents, height, width, vae_scale_factor) if unpack is not None else latents
                vae_cfg = pipe_obj.vae.config
                lat = (lat / vae_cfg.scaling_factor) + vae_cfg.shift_factor
                decoded = pipe_obj.vae.decode(lat, return_dict=False)[0]

        image = pipe_obj.image_processor.postprocess(decoded, output_type="pil")[0]
        if max_size and max(image.size) > max_size:
            ratio = max_size / max(image.size)
            image = image.resize(
                (int(image.width * ratio), int(image.height * ratio)),
                Image.Resampling.LANCZOS,
            )
        return image
    except Exception as e:
        print(f"[preview] decode failed: {e}", flush=True)
        return None


def generate_image(prompt, seed=None, steps=6, width=1024, height=1024, local_encoder=False, input_image=None, strength=0.75, sigmas=None, guidance_scale=None, callback_on_step_end=None):
    """Generate an image from a text prompt.

    Args:
        prompt: Text description of the image to generate
        seed: Random seed for reproducibility
        steps: Number of inference steps
        width: Output image width
        height: Output image height
        local_encoder: Use local text encoder instead of remote API
        input_image: Optional PIL Image for img2img generation
        strength: Denoising strength for img2img (0.0-1.0, higher = more change)
        sigmas: Custom noise schedule (for turbo LoRA, use TURBO_SIGMAS)
        guidance_scale: Classifier-free guidance scale (default: 4, turbo uses 2.5)
        callback_on_step_end: Optional callback called after each inference step.
            Signature: callback(pipe, step_index, timestep, callback_kwargs) -> callback_kwargs
    """
    global pipe_img2img

    if seed is None:
        seed = torch.randint(0, 2**32, (1,)).item()

    # Auto-configure for turbo mode if enabled
    if _turbo_enabled and sigmas is None:
        sigmas = TURBO_SIGMAS
        steps = 8  # Turbo uses 8 steps

    # Auto-configure for schnell mode if enabled
    # Schnell is a distilled model that doesn't use CFG - must force guidance_scale=0
    if _schnell_enabled:
        # For img2img, effective steps = int(steps * strength), so we need more base steps
        # to ensure enough denoising. With 8 steps and 0.75 strength = 6 effective steps.
        if input_image is not None:
            steps = 8  # More steps for img2img to compensate for strength reduction
        else:
            steps = 4  # Schnell is optimized for 4 steps txt2img
        guidance_scale = 0  # Always force 0 for schnell

    # Ensure strength won't result in zero pipeline steps for img2img
    if input_image is not None and strength is not None:
        min_strength = 1.0 / steps
        if strength < min_strength:
            print(f"Warning: strength {strength} too low for {steps} steps (minimum {min_strength:.2f}). Using {min_strength:.2f}.")
            strength = min_strength

    if guidance_scale is None:
        # Turbo uses 2.5, others use 4
        if _turbo_enabled:
            guidance_scale = 2.5
        else:
            guidance_scale = 4

    timings = {}

    # Generate with inference_mode for better performance
    with torch.inference_mode():
        if input_image is not None:
            # Img2img mode
            if _flux_version == 2:
                # FLUX.2 has image conditioning built into the main pipeline
                # (no strength parameter - image is used as reference/conditioning)
                input_image = input_image.resize((width, height))

                t0 = time.perf_counter()
                pipe_kwargs = {
                    "prompt": prompt,
                    "image": input_image,
                    "generator": torch.Generator(device=device).manual_seed(seed),
                    "num_inference_steps": steps,
                    "guidance_scale": guidance_scale,
                    "height": height,
                    "width": width,
                    "max_sequence_length": 512,
                }
                if sigmas is not None:
                    pipe_kwargs["sigmas"] = sigmas
                if callback_on_step_end is not None:
                    pipe_kwargs["callback_on_step_end"] = callback_on_step_end
                image = pipe(**pipe_kwargs).images[0]
                timings['diffusion'] = time.perf_counter() - t0
                timings['encoding'] = 0
            else:
                # FLUX.1 needs separate img2img pipeline
                if pipe_img2img is None:
                    # For GGUF models, from_pipe() fails because it tries to cast dtype
                    # on quantized models. Create the pipeline directly instead.
                    if _model_type and _model_type.startswith('gguf'):
                        pipe_img2img = FluxImg2ImgPipeline(
                            scheduler=pipe.scheduler,
                            vae=pipe.vae,
                            text_encoder=pipe.text_encoder,
                            text_encoder_2=pipe.text_encoder_2,
                            tokenizer=pipe.tokenizer,
                            tokenizer_2=pipe.tokenizer_2,
                            transformer=pipe.transformer,
                        )
                    else:
                        pipe_img2img = FluxImg2ImgPipeline.from_pipe(pipe)
                    # Wrap VAE for dtype safety (handles 4-bit/GGUF dtype mismatches)
                    _wrap_vae_for_dtype_safety(pipe_img2img)

                input_image = input_image.resize((width, height))

                t0 = time.perf_counter()
                img2img_kwargs = {
                    "prompt": prompt,
                    "image": input_image,
                    "strength": strength,
                    "generator": torch.Generator(device=device).manual_seed(seed),
                    "num_inference_steps": steps,
                    "guidance_scale": guidance_scale,
                    "max_sequence_length": 512,
                }
                if callback_on_step_end is not None:
                    img2img_kwargs["callback_on_step_end"] = callback_on_step_end
                image = pipe_img2img(**img2img_kwargs).images[0]
                timings['diffusion'] = time.perf_counter() - t0
                timings['encoding'] = 0
        elif local_encoder or _flux_version == 2:
            # Use local text encoder
            t0 = time.perf_counter()
            pipe_kwargs = {
                "prompt": prompt,
                "generator": torch.Generator(device=device).manual_seed(seed),
                "num_inference_steps": steps,
                "guidance_scale": guidance_scale,
                "width": width,
                "height": height,
                "max_sequence_length": 512,
            }
            if sigmas is not None:
                pipe_kwargs["sigmas"] = sigmas
            if callback_on_step_end is not None:
                pipe_kwargs["callback_on_step_end"] = callback_on_step_end
            image = pipe(**pipe_kwargs).images[0]
            timings['diffusion'] = time.perf_counter() - t0
            timings['encoding'] = 0  # Included in diffusion for local
        else:
            # Use remote text encoder - get embeddings first
            t0 = time.perf_counter()
            embeds = remote_text_encoder(prompt)
            timings['encoding'] = time.perf_counter() - t0

            t0 = time.perf_counter()
            remote_pipe_kwargs = {
                "prompt_embeds": embeds,
                "generator": torch.Generator(device=device).manual_seed(seed),
                "num_inference_steps": steps,
                "guidance_scale": guidance_scale,
                "width": width,
                "height": height,
            }
            if callback_on_step_end is not None:
                remote_pipe_kwargs["callback_on_step_end"] = callback_on_step_end
            image = pipe(**remote_pipe_kwargs).images[0]
            timings['diffusion'] = time.perf_counter() - t0

    timings['total'] = timings['encoding'] + timings['diffusion']
    return image, seed, timings

def compile_pipeline():
    """Compile transformer for faster inference (slower first run, faster subsequent)"""
    print("Compiling transformer (this may take a minute)...")
    t0 = time.perf_counter()
    pipe.transformer = torch.compile(pipe.transformer, mode="reduce-overhead")
    print(f"Compilation done in {time.perf_counter() - t0:.1f}s")


def play_completion_sound():
    """Play a pleasant notification sound when image generation is complete."""
    # Try paplay (PulseAudio) with system sounds first
    if shutil.which("paplay"):
        sound_paths = [
            "/usr/share/sounds/freedesktop/stereo/complete.oga",
            "/usr/share/sounds/gnome/default/alerts/glass.ogg",
            "/usr/share/sounds/ubuntu/stereo/message.ogg",
            "/usr/share/sounds/freedesktop/stereo/message.oga",
        ]
        for sound in sound_paths:
            try:
                subprocess.run(["paplay", sound], stderr=subprocess.DEVNULL, timeout=2)
                return
            except (subprocess.SubprocessError, FileNotFoundError):
                continue

    # Try aplay with a beep
    if shutil.which("aplay"):
        try:
            subprocess.run(["aplay", "-q", "/usr/share/sounds/alsa/Front_Center.wav"],
                         stderr=subprocess.DEVNULL, timeout=2)
            return
        except (subprocess.SubprocessError, FileNotFoundError):
            pass

    # Fallback to terminal bell
    print("\a", end="", flush=True)

def save_prompt_file(filepath, raw_prompt, prompt, width, height, seed, steps, timings, guidance_scale=None, strength=None):
    """Save prompt metadata alongside image."""
    prompt_path = filepath.rsplit('.', 1)[0] + '.prompt'
    with open(prompt_path, 'w') as f:
        f.write(f"# Raw input: {raw_prompt}\n")
        f.write(f"# Prompt: {prompt}\n")
        f.write(f"# Dimensions: {width}x{height}\n")
        f.write(f"# Seed: {seed}\n")
        f.write(f"# Steps: {steps}\n")
        f.write(f"# Guidance: {guidance_scale if guidance_scale else 'auto'}\n")
        if strength is not None:
            f.write(f"# Strength: {strength} (img2img)\n")
        f.write(f"# Timings: encoding={timings['encoding']:.2f}s, diffusion={timings['diffusion']:.2f}s, save={timings.get('save', 0):.2f}s\n")
    return prompt_path


def main():
    parser = argparse.ArgumentParser(description="FLUX Image Generator")
    parser.add_argument("--steps", type=int, default=25, help="Number of inference steps (default: 25)")
    parser.add_argument("--guidance", type=float, default=None, help="Guidance scale (default: 4 for normal, 2.5 for turbo). Lower values give more variety.")
    parser.add_argument("--compile", action="store_true", help="Compile model for faster inference (slower startup)")
    parser.add_argument("--local-encoder", action="store_true", help="Use local text encoder instead of remote API (requires more VRAM)")
    parser.add_argument("--full-model", action="store_true", help="Use full FLUX model instead of 4-bit quantized (requires more VRAM)")
    parser.add_argument("--gguf", type=str, choices=["bf16", "q8", "q4"], default=None,
                        help="Use GGUF model (FLUX.1 only, recommended for DGX Spark). Options: bf16 (full quality), q8 (8-bit), q4 (4-bit smallest)")
    parser.add_argument("--flux2", action="store_true", help="Use FLUX.2 model instead of FLUX.1 (requires more VRAM)")
    parser.add_argument("--schnell", action="store_true", help="Use FLUX.1-schnell (fast 4-step model, Apache 2.0 license)")
    parser.add_argument("--image", type=str, default=None, help="Reference image path for img2img generation")
    parser.add_argument("--strength", type=float, default=0.75, help="Denoising strength for img2img (0.0-1.0, default: 0.75). Higher = more change from original")
    args = parser.parse_args()

    # Full model and schnell always use local encoder
    use_local_encoder = args.local_encoder or args.full_model or args.schnell

    # Load the model
    load_model(local_encoder=use_local_encoder, full_model=args.full_model, gguf_quant=args.gguf, flux2=args.flux2, schnell=args.schnell)

    if args.compile:
        compile_pipeline()

    image_count = 0
    current_prompt = None
    last_seed = None

    steps = args.steps
    guidance_scale = args.guidance  # None means use default (4 for normal, 2.5 for turbo)
    strength = args.strength  # Denoising strength for img2img
    input_image = None
    if args.image:
        try:
            input_image = Image.open(args.image).convert("RGB")
            print(f"Loaded reference image: {args.image} ({input_image.size[0]}x{input_image.size[1]})")
        except Exception as e:
            print(f"Warning: Could not load reference image '{args.image}': {e}")
            input_image = None

    # Orientation presets (width, height) at 1K base
    orientations_1k = {
        'square': (1024, 1024),
        'portrait': (768, 1344),
        'landscape': (1360, 768),  # 16:9
        'widescreen': (1568, 672),  # ~21:9 extra-wide
        'extra-tall': (672, 1568),  # ~9:21 mirror of widescreen
    }
    # Size presets
    sizes = {
        '0.75': 0.75,
        '1k': 1.0,
        '2k': 2.0,
        '4k': 4.0,
    }
    orientation = 'landscape'
    size = '1k'
    base_w, base_h = orientations_1k[orientation]
    width, height = int(base_w * sizes[size]), int(base_h * sizes[size])

    flux_name = f"FLUX.{_flux_version}"
    hostname = socket.gethostname()
    print(f"\n=== {flux_name} Image Generator on {hostname} ===")
    if args.gguf:
        model_mode = f"GGUF {args.gguf.upper()}"
    elif args.full_model:
        model_mode = "full model"
    else:
        model_mode = "4-bit BNB"
    encoder_mode = "local encoder" if use_local_encoder else "remote encoder"
    compiled_str = " (compiled)" if args.compile else ""
    print(f"Model: {flux_name} {model_mode}, {encoder_mode}{compiled_str}")
    print(f"Output: {size} {orientation} ({width}x{height}), {steps} steps")
    print("Commands:")
    print("  'quit' or 'q' - Exit the program")
    print("  'same' or 's' - Regenerate with same prompt (uses cached embeddings)")
    print("  'reseed <number>' - Regenerate with specific seed")
    print("  '/steps <number>' - Change inference steps (current: {})".format(steps))
    print("  '/guidance <number>' - Change guidance scale (current: {}, lower=more variety)".format(guidance_scale if guidance_scale else "auto"))
    print("  '/image <path>' - Load reference image for img2img (use '/image clear' to remove)")
    print("  '/strength <number>' - Set img2img denoising strength 0.0-1.0 (current: {})".format(strength))
    print("  '/square' - Set square aspect ratio")
    print("  '/portrait' - Set portrait aspect ratio")
    print("  '/landscape' - Set 16:9 landscape aspect ratio")
    print("  '/widescreen' - Set 21:9 extra-wide aspect ratio")
    print("  '/extra-tall' - Set 9:21 extra-tall aspect ratio")
    print("  '/0.75' - Set 0.75K resolution (smaller/faster)")
    print("  '/1k' - Set 1K resolution (default)")
    print("  '/2k' - Set 2K resolution")
    print("  '/4k' - Set 4K resolution")
    print("  Or enter a prompt (can include modifiers: 'a cat /4k /portrait')\n")

    while True:
        if current_prompt is None:
            user_input = input("Enter your prompt: ").strip()
        else:
            print(f"\nCurrent prompt: {current_prompt[:80]}{'...' if len(current_prompt) > 80 else ''}")
            user_input = input("Enter new prompt, modification, or command: ").strip()

        if not user_input:
            print("Please enter a prompt or command.")
            continue

        lower_input = user_input.lower()

        if lower_input in ('quit', 'q'):
            print("Goodbye!")
            break

        if lower_input.startswith('/steps '):
            try:
                new_steps = int(user_input.split()[1])
                if new_steps < 1:
                    print("Steps must be at least 1.")
                    continue
                steps = new_steps
                print(f"Inference steps set to {steps}")
            except (ValueError, IndexError):
                print("Invalid steps. Usage: /steps 10")
            continue

        if lower_input.startswith('/guidance '):
            try:
                new_guidance = float(user_input.split()[1])
                if new_guidance < 0:
                    print("Guidance scale must be non-negative.")
                    continue
                guidance_scale = new_guidance
                print(f"Guidance scale set to {guidance_scale}")
            except (ValueError, IndexError):
                print("Invalid guidance scale. Usage: /guidance 3.5")
            continue

        if lower_input.startswith('/strength '):
            try:
                new_strength = float(user_input.split()[1])
                if new_strength <= 0 or new_strength > 1:
                    print("Strength must be between 0.01 and 1.0 (0.0 would result in zero pipeline steps).")
                    continue
                strength = new_strength
                print(f"Img2img strength set to {strength}")
            except (ValueError, IndexError):
                print("Invalid strength. Usage: /strength 0.75")
            continue

        if lower_input.startswith('/image '):
            image_arg = user_input.split(maxsplit=1)[1] if len(user_input.split()) > 1 else ""
            if image_arg.lower() == 'clear':
                input_image = None
                print("Reference image cleared. Using text-to-image mode.")
            else:
                try:
                    input_image = Image.open(image_arg).convert("RGB")
                    print(f"Loaded reference image: {image_arg} ({input_image.size[0]}x{input_image.size[1]})")
                except Exception as e:
                    print(f"Could not load image '{image_arg}': {e}")
            continue

        if lower_input in ('/square', '/portrait', '/landscape', '/widescreen', '/extra-tall'):
            orientation = lower_input[1:]  # Remove the leading /
            base_w, base_h = orientations_1k[orientation]
            width, height = int(base_w * sizes[size]), int(base_h * sizes[size])
            print(f"Orientation set to {orientation} ({width}x{height})")
            continue

        if lower_input in ('/0.75', '/1k', '/2k', '/4k'):
            size = lower_input[1:]  # Remove the leading /
            base_w, base_h = orientations_1k[orientation]
            width, height = int(base_w * sizes[size]), int(base_h * sizes[size])
            print(f"Size set to {size} ({width}x{height})")
            continue

        # Parse inline modifiers from prompt (e.g., "a cat /4k /portrait")
        words = user_input.split()
        modifiers_found = []
        prompt_words = []
        for word in words:
            lower_word = word.lower()
            if lower_word in ('/0.75', '/1k', '/2k', '/4k'):
                size = lower_word[1:]
                modifiers_found.append(f"size={size}")
            elif lower_word in ('/square', '/portrait', '/landscape', '/widescreen', '/extra-tall'):
                orientation = lower_word[1:]
                modifiers_found.append(f"orientation={orientation}")
            else:
                prompt_words.append(word)

        if modifiers_found:
            base_w, base_h = orientations_1k[orientation]
            width, height = int(base_w * sizes[size]), int(base_h * sizes[size])
            print(f"Applied: {', '.join(modifiers_found)} -> {width}x{height}")

        # Reconstruct prompt without modifiers
        user_input = ' '.join(prompt_words)
        lower_input = user_input.lower()

        if not user_input:
            # Input was only modifiers, no prompt
            continue

        if lower_input in ('same', 's') and current_prompt:
            prompt = current_prompt
        elif lower_input.startswith('reseed ') and current_prompt:
            try:
                last_seed = int(lower_input.split()[1])
                prompt = current_prompt
            except (ValueError, IndexError):
                print("Invalid seed. Usage: reseed 12345")
                continue
        else:
            prompt = user_input
            current_prompt = prompt
            last_seed = None

        # Build generation info string
        gen_info_parts = [f"{steps} steps", f"{width}x{height}"]
        if guidance_scale:
            gen_info_parts.append(f"guidance={guidance_scale}")
        if input_image:
            gen_info_parts.append(f"img2img strength={strength}")
        gen_info = ", ".join(gen_info_parts)

        print(f"\n[{hostname}] Generating: {gen_info}")
        raw_input = user_input  # Save original input before any processing
        image, last_seed, timings = generate_image(prompt, last_seed if lower_input.startswith('reseed ') else None, steps, width, height, use_local_encoder, input_image=input_image, strength=strength, guidance_scale=guidance_scale)

        image_count += 1
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_id = uuid.uuid4().hex[:8]
        filename = f"flux{_flux_version}_{timestamp}_{unique_id}.png"

        t0 = time.perf_counter()
        image.save(filename)
        timings['save'] = time.perf_counter() - t0

        # Save prompt file alongside image
        save_prompt_file(filename, raw_input, prompt, width, height, last_seed, steps, timings, guidance_scale, strength if input_image else None)

        # Clean summary output
        total_time = timings['total'] + timings['save']
        print(f"[{hostname}] Saved: {filename}")
        print(f"  seed={last_seed}, {total_time:.2f}s total (encode={timings['encoding']:.2f}s, diffuse={timings['diffusion']:.2f}s, save={timings['save']:.2f}s)")
        play_completion_sound()

if __name__ == "__main__":
    main()
