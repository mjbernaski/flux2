import argparse
import torch
import os

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
FLUX1_GGUF_MODELS = {
    "bf16": "https://huggingface.co/city96/FLUX.1-dev-gguf/blob/main/flux1-dev-BF16.gguf",
    "q8": "https://huggingface.co/city96/FLUX.1-dev-gguf/blob/main/flux1-dev-Q8_0.gguf",
    "q4": "https://huggingface.co/city96/FLUX.1-dev-gguf/blob/main/flux1-dev-Q4_K_S.gguf",
}

# FLUX.2 repos
FLUX2_REPO_4BIT = "diffusers/FLUX.2-dev-bnb-4bit"
FLUX2_REPO_FULL = "black-forest-labs/FLUX.2-dev"
device = "cuda:0"
torch_dtype = torch.bfloat16

# Lazy-loaded model components
transformer = None
pipe = None
pipe_img2img = None  # Img2img pipeline (created on-demand from pipe)
_model_type = None  # Track which model is loaded
_flux_version = 1  # Track FLUX version (1 or 2)
_turbo_enabled = False  # Track if turbo LoRA is loaded

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
        print(f"  VAE dtype: {vae_dtype} (matches torch_dtype, no wrapping needed)")
        return

    print(f"  VAE dtype: {vae_dtype} (wrapping for dtype safety)")

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


def load_model(local_encoder=False, full_model=False, gguf_quant=None, flux2=False):
    """Load the FLUX model components. Call this before generating images.

    Args:
        local_encoder: Use local text encoder instead of remote API
        full_model: Use full FLUX model instead of 4-bit quantized
        gguf_quant: GGUF quantization level ('bf16', 'q8', 'q4') - FLUX.1 only
        flux2: Use FLUX.2 model instead of FLUX.1
    """
    global transformer, pipe, _model_type, _flux_version
    if pipe is not None:
        return {}  # Already loaded

    _flux_version = 2 if flux2 else 1
    flux_name = f"FLUX.{_flux_version}"

    # Select repos based on version
    if flux2:
        repo_4bit = FLUX2_REPO_4BIT
        repo_full = FLUX2_REPO_FULL
        if gguf_quant:
            print("Warning: GGUF not available for FLUX.2, using 4-bit instead")
            gguf_quant = None
    else:
        repo_4bit = FLUX1_REPO_4BIT
        repo_full = FLUX1_REPO_FULL

    load_timings = {}
    total_start = time.perf_counter()

    # Determine model type
    if gguf_quant:
        _model_type = f"gguf-{gguf_quant}"
        model_desc = f"GGUF {gguf_quant.upper()} {flux_name}"
    elif full_model:
        _model_type = "full"
        model_desc = f"full {flux_name}-dev"
    else:
        _model_type = "4bit"
        model_desc = f"4-bit quantized {flux_name}"

    print(f"Loading {model_desc}...")

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

    elif full_model:
        # Full model - load transformer and text encoder in parallel for faster startup
        import concurrent.futures

        repo_id = repo_full
        print(f"Loading {flux_name} components in parallel...")
        t0 = time.perf_counter()

        if flux2:
            # FLUX.2 uses Mistral3 text encoder
            from transformers import Mistral3ForConditionalGeneration

            def load_transformer():
                return Flux2Transformer2DModel.from_pretrained(
                    repo_id, subfolder="transformer", torch_dtype=torch_dtype,
                    device_map="cuda", low_cpu_mem_usage=True, use_safetensors=True
                )

            def load_text_encoder():
                # Mistral3 doesn't support low_cpu_mem_usage=True with device_map="cuda"
                # (causes meta tensor dispatch error). Use device_map="auto" instead.
                return Mistral3ForConditionalGeneration.from_pretrained(
                    repo_id, subfolder="text_encoder", torch_dtype=torch_dtype,
                    device_map="auto", use_safetensors=True
                )

            # Load components in parallel
            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
                transformer_future = executor.submit(load_transformer)
                encoder_future = executor.submit(load_text_encoder)

                transformer = transformer_future.result()
                text_encoder = encoder_future.result()

            load_timings['parallel_load'] = time.perf_counter() - t0
            print(f"  Components loaded in parallel in {load_timings['parallel_load']:.2f}s")

            # Assemble FLUX.2 pipeline
            # Note: Use device_map="balanced" (not "cuda" + low_cpu_mem_usage) to avoid
            # meta tensor errors when loading remaining components (VAE, scheduler, etc.)
            print("Assembling pipeline...")
            t0 = time.perf_counter()
            pipe = Flux2Pipeline.from_pretrained(
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
                return FluxTransformer2DModel.from_pretrained(
                    repo_id, subfolder="transformer", torch_dtype=torch_dtype,
                    device_map="cuda", low_cpu_mem_usage=True, use_safetensors=True
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
                from transformers import Mistral3ForConditionalGeneration
                print("Loading local text encoder (Mistral3, requires more VRAM)...")
                text_encoder = Mistral3ForConditionalGeneration.from_pretrained(
                    repo_id, subfolder="text_encoder", torch_dtype=torch_dtype,
                    device_map="auto", use_safetensors=True
                )
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
    print(f"Model loaded successfully in {load_timings['total']:.2f}s")
    print(f"  Summary: transformer={load_timings['transformer']:.2f}s, pipeline={load_timings['pipeline']:.2f}s, to_gpu={load_timings['to_device']:.2f}s")

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

def generate_image(prompt, seed=None, steps=6, width=1024, height=1024, local_encoder=False, input_image=None, strength=0.75, sigmas=None, guidance_scale=None):
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
    """
    global pipe_img2img

    if seed is None:
        seed = torch.randint(0, 2**32, (1,)).item()
    print(f"Using seed: {seed}")

    # Auto-configure for turbo mode if enabled
    if _turbo_enabled and sigmas is None:
        sigmas = TURBO_SIGMAS
        steps = 8  # Turbo uses 8 steps
        print(f"Turbo mode: using 8 steps with custom sigmas")

    if guidance_scale is None:
        guidance_scale = 2.5 if _turbo_enabled else 4

    timings = {}

    # Generate with inference_mode for better performance
    with torch.inference_mode():
        if input_image is not None:
            # Img2img mode
            if _flux_version == 2:
                # FLUX.2 has image conditioning built into the main pipeline
                # (no strength parameter - image is used as reference/conditioning)
                input_image = input_image.resize((width, height))
                print(f"Using image as reference for generation")

                t0 = time.perf_counter()
                pipe_kwargs = {
                    "prompt": prompt,
                    "image": input_image,
                    "generator": torch.Generator(device=device).manual_seed(seed),
                    "num_inference_steps": steps,
                    "guidance_scale": guidance_scale,
                    "height": height,
                    "width": width,
                }
                if sigmas is not None:
                    pipe_kwargs["sigmas"] = sigmas
                image = pipe(**pipe_kwargs).images[0]
                timings['diffusion'] = time.perf_counter() - t0
                timings['encoding'] = 0
            else:
                # FLUX.1 needs separate img2img pipeline
                if pipe_img2img is None:
                    print("Creating img2img pipeline (first use)...")
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
                        print("  Created img2img pipeline directly (GGUF mode)")
                    else:
                        pipe_img2img = FluxImg2ImgPipeline.from_pipe(pipe)
                    # Wrap VAE for dtype safety (handles 4-bit/GGUF dtype mismatches)
                    _wrap_vae_for_dtype_safety(pipe_img2img)

                input_image = input_image.resize((width, height))
                print(f"Using img2img with strength={strength}")

                t0 = time.perf_counter()
                image = pipe_img2img(
                    prompt=prompt,
                    image=input_image,
                    strength=strength,
                    generator=torch.Generator(device=device).manual_seed(seed),
                    num_inference_steps=steps,
                    guidance_scale=guidance_scale,
                ).images[0]
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
            }
            if sigmas is not None:
                pipe_kwargs["sigmas"] = sigmas
            image = pipe(**pipe_kwargs).images[0]
            timings['diffusion'] = time.perf_counter() - t0
            timings['encoding'] = 0  # Included in diffusion for local
        else:
            # Use remote text encoder - get embeddings first
            t0 = time.perf_counter()
            embeds = remote_text_encoder(prompt)
            timings['encoding'] = time.perf_counter() - t0

            t0 = time.perf_counter()
            image = pipe(
                prompt_embeds=embeds,
                generator=torch.Generator(device=device).manual_seed(seed),
                num_inference_steps=steps,
                guidance_scale=guidance_scale,
                width=width,
                height=height,
            ).images[0]
            timings['diffusion'] = time.perf_counter() - t0

    timings['total'] = timings['encoding'] + timings['diffusion']

    print(f"Timing: encoding={timings['encoding']:.2f}s, diffusion={timings['diffusion']:.2f}s, total={timings['total']:.2f}s")
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

def save_prompt_file(filepath, raw_prompt, prompt, width, height, seed, steps, timings, guidance_scale=None):
    """Save prompt metadata alongside image."""
    prompt_path = filepath.rsplit('.', 1)[0] + '.prompt'
    with open(prompt_path, 'w') as f:
        f.write(f"# Raw input: {raw_prompt}\n")
        f.write(f"# Prompt: {prompt}\n")
        f.write(f"# Dimensions: {width}x{height}\n")
        f.write(f"# Seed: {seed}\n")
        f.write(f"# Steps: {steps}\n")
        f.write(f"# Guidance: {guidance_scale if guidance_scale else 'auto'}\n")
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
    args = parser.parse_args()

    # Full model always uses local encoder
    use_local_encoder = args.local_encoder or args.full_model

    # Load the model
    load_model(local_encoder=use_local_encoder, full_model=args.full_model, gguf_quant=args.gguf, flux2=args.flux2)

    if args.compile:
        compile_pipeline()

    image_count = 0
    current_prompt = None
    last_seed = None

    steps = args.steps
    guidance_scale = args.guidance  # None means use default (4 for normal, 2.5 for turbo)

    # Orientation presets (width, height) at 1K base
    orientations_1k = {
        'square': (1024, 1024),
        'portrait': (768, 1344),
        'landscape': (1344, 768),
        '16:9': (1360, 768),
    }
    # Size presets
    sizes = {
        '1k': 1.0,
        '2k': 2.0,
        '4k': 4.0,
    }
    orientation = 'landscape'
    size = '1k'
    base_w, base_h = orientations_1k[orientation]
    width, height = int(base_w * sizes[size]), int(base_h * sizes[size])

    flux_name = f"FLUX.{_flux_version}"
    print(f"\n=== {flux_name} Image Generator ===")
    if args.gguf:
        model_mode = f"GGUF {args.gguf.upper()}"
    elif args.full_model:
        model_mode = "full model"
    else:
        model_mode = "4-bit BNB"
    encoder_mode = "local encoder" if use_local_encoder else "remote encoder"
    print(f"Using {flux_name} {model_mode}, {encoder_mode}, {steps} steps, {size} {orientation} ({width}x{height})" + (" (compiled)" if args.compile else ""))
    print("Commands:")
    print("  'quit' or 'q' - Exit the program")
    print("  'same' or 's' - Regenerate with same prompt (uses cached embeddings)")
    print("  'reseed <number>' - Regenerate with specific seed")
    print("  '/steps <number>' - Change inference steps (current: {})".format(steps))
    print("  '/guidance <number>' - Change guidance scale (current: {}, lower=more variety)".format(guidance_scale if guidance_scale else "auto"))
    print("  '/square' - Set square aspect ratio")
    print("  '/portrait' - Set portrait aspect ratio")
    print("  '/landscape' - Set landscape aspect ratio")
    print("  '/16:9' - Set 16:9 widescreen aspect ratio")
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

        if lower_input in ('/square', '/portrait', '/landscape', '/16:9'):
            orientation = lower_input[1:]  # Remove the leading /
            base_w, base_h = orientations_1k[orientation]
            width, height = int(base_w * sizes[size]), int(base_h * sizes[size])
            print(f"Orientation set to {orientation} ({width}x{height})")
            continue

        if lower_input in ('/1k', '/2k', '/4k'):
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
            if lower_word in ('/1k', '/2k', '/4k'):
                size = lower_word[1:]
                modifiers_found.append(f"size={size}")
            elif lower_word in ('/square', '/portrait', '/landscape', '/16:9'):
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

        guidance_str = f", guidance={guidance_scale}" if guidance_scale else ""
        print(f"\nGenerating image ({steps} steps{guidance_str}, {size} {orientation} {width}x{height})...")
        raw_input = user_input  # Save original input before any processing
        image, last_seed, timings = generate_image(prompt, last_seed if lower_input.startswith('reseed ') else None, steps, width, height, use_local_encoder, guidance_scale=guidance_scale)

        image_count += 1
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_id = uuid.uuid4().hex[:8]
        filename = f"flux{_flux_version}_{timestamp}_{unique_id}.png"

        t0 = time.perf_counter()
        image.save(filename)
        timings['save'] = time.perf_counter() - t0

        # Save prompt file alongside image
        prompt_file = save_prompt_file(filename, raw_input, prompt, width, height, last_seed, steps, timings, guidance_scale)

        print(f"Image saved as: {filename}")
        print(f"Prompt saved as: {prompt_file}")
        print(f"  Steps breakdown: encoding={timings['encoding']:.2f}s, diffusion={timings['diffusion']:.2f}s, save={timings['save']:.2f}s")
        play_completion_sound()

if __name__ == "__main__":
    main()
