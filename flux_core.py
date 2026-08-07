import argparse
import inspect
import torch
import os
import socket
import logging
import warnings
from PIL import Image, ImageFilter

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

# transformers v5 passes extra kwargs (e.g. _is_hf_initialized, via the old
# param's __dict__) when rebuilding bnb quantized params; bitsandbytes only
# tolerates them from v0.50.0 (PR #1900). Mirror that fix until it releases.
import bitsandbytes.nn as _bnb_nn

def _bnb_accept_extra_kwargs(cls):
    orig_new = cls.__new__
    params = inspect.signature(orig_new).parameters
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return  # already fixed upstream
    allowed = set(params) - {"cls"}

    def patched_new(cls, *args, **kwargs):
        return orig_new(cls, *args, **{k: v for k, v in kwargs.items() if k in allowed})

    cls.__new__ = patched_new

_bnb_accept_extra_kwargs(_bnb_nn.Params4bit)
_bnb_accept_extra_kwargs(_bnb_nn.Int8Params)


def _gpu_max_memory():
    """Explicit accelerate memory budget: everything on GPU 0, no CPU offload.

    On unified-memory systems torch.cuda.mem_get_info() mirrors the kernel's
    *free* figure, which excludes reclaimable page cache. Streaming tens of GB
    of weights off disk fills that cache, so by pipeline-assembly time
    accelerate's auto placement (device_map="auto"/"balanced") believes the
    GPU is full and silently offloads modules to CPU — leaving their params on
    meta with dispatch hooks, which later breaks .to() and forward passes.
    Memory is one physical pool here, so budget from total instead; if a
    config genuinely doesn't fit, fail loudly rather than offload.
    """
    total = torch.cuda.mem_get_info()[1]
    return {0: int(total * 0.97), "cpu": 0}

# FLUX.1 classes
from diffusers import FluxPipeline, FluxImg2ImgPipeline, FluxTransformer2DModel, FluxKontextPipeline
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
# FLUX.1 Kontext: instruction-based image editor ("change X, keep the rest").
# No official bnb-4bit repo exists, so we quantize the full repo on the fly.
FLUX1_KONTEXT_REPO = "black-forest-labs/FLUX.1-Kontext-dev"
FLUX1_GGUF_MODELS = {
    "bf16": "https://huggingface.co/city96/FLUX.1-dev-gguf/blob/main/flux1-dev-BF16.gguf",
    "q8": "https://huggingface.co/city96/FLUX.1-dev-gguf/blob/main/flux1-dev-Q8_0.gguf",
    "q4": "https://huggingface.co/city96/FLUX.1-dev-gguf/blob/main/flux1-dev-Q4_K_S.gguf",
}

# FLUX.2 repos
FLUX2_REPO_4BIT = "diffusers/FLUX.2-dev-bnb-4bit"
FLUX2_REPO_FULL = "black-forest-labs/FLUX.2-dev"

# FLUX.2 klein (smaller 9B/4B models, both use Flux2KleinPipeline). Full bf16 only —
# NVFP4 variants are blocked by a diffusers upstream bug in the Flux2 single-file
# converter (qkv-chunking assumes unquantized fused weights; NVFP4 scale tensors
# break it).
FLUX2_KLEIN_REPO_FULL = "black-forest-labs/FLUX.2-klein-9B"
FLUX2_KLEIN_REPO_4B = "black-forest-labs/FLUX.2-klein-4B"
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
_kontext_enabled = False  # Track if using FLUX.1 Kontext (instruction-based editor)
_local_encoder_active = False  # Set when local encoders are loaded (incl. remote-encoder fallback)

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

# Embedding cache - avoids redundant API calls for same prompts.
# Bounded LRU: each entry is a GPU tensor, so an unbounded dict leaks VRAM on
# long-running servers with many unique prompts.
from collections import OrderedDict
_embedding_cache = OrderedDict()
_EMBEDDING_CACHE_MAX = 32


def _stabilize_vae_fp32(pipeline):
    """Run the VAE in float32 to prevent intermittent NaN/black decodes.

    The transformer and text encoder run in bfloat16, but the VAE's deep conv
    stack can overflow in bf16 and emit NaN/Inf (or fully-saturated) values.
    Downstream, ``(image * 255).round().astype("uint8")`` turns NaN into 0 and
    clamps saturated values to 0 -> a pure-black image that gets silently saved
    as a "successful" generation. Upcasting just the VAE to fp32 (~0.6GB) is the
    standard FLUX stability fix; the transformer stays in bf16 for speed/VRAM.

    Latents arrive from the transformer in bf16, so decode/encode are wrapped to
    cast their inputs to fp32. The encode wrapper also casts its *output* back to
    the model dtype: image-conditioning paths (Kontext, img2img) encode a
    reference image through the VAE and feed the result to the bf16 transformer,
    so fp32 latents there cause "mat1 and mat2 must have the same dtype". Decode
    output stays fp32 (it goes straight to the image processor). Idempotent —
    safe to call repeatedly on a shared VAE (e.g. img2img reuses the txt2img VAE).
    """
    if not hasattr(pipeline, 'vae') or pipeline.vae is None:
        return
    vae = pipeline.vae
    if getattr(vae, '_fp32_stabilized', False):
        return
    vae.to(torch.float32)

    original_encode = vae.encode
    original_decode = vae.decode

    def wrapped_encode(x, *args, **kwargs):
        if hasattr(x, 'dtype') and x.dtype != torch.float32:
            x = x.to(torch.float32)
        out = original_encode(x, *args, **kwargs)
        # Cast the encoded distribution back to the model dtype so downstream
        # conditioning (Kontext/img2img) matches the bf16 transformer.
        dist = getattr(out, 'latent_dist', None)
        if dist is not None:
            for attr in ('parameters', 'mean', 'logvar', 'std', 'var'):
                v = getattr(dist, attr, None)
                if v is not None and getattr(v, 'dtype', None) != torch_dtype:
                    setattr(dist, attr, v.to(torch_dtype))
        return out

    def wrapped_decode(z, *args, **kwargs):
        if hasattr(z, 'dtype') and z.dtype != torch.float32:
            z = z.to(torch.float32)
        return original_decode(z, *args, **kwargs)

    vae.encode = wrapped_encode
    vae.decode = wrapped_decode
    vae._fp32_stabilized = True


# Generation can intermittently diverge (bf16 VAE/transformer NaN or saturation)
# and decode to a pure-black frame. We detect that and retry before giving up.
_MAX_GEN_RETRIES = 1  # retries after the first attempt (so up to 2 attempts total)


class DegenerateImageError(RuntimeError):
    """Raised when generation yields an all-black (NaN/saturated) image."""


def _is_degenerate_image(image):
    """True if the image is effectively all-black.

    A NaN/Inf or fully-saturated latent decodes to pure black: NaN -> 0 in the
    uint8 cast, all-negative saturation -> clamped to 0. No real image (even a
    dark night scene) is uniformly black, so the brightest pixel across all
    channels being ~0 is an unambiguous failure signal.
    """
    try:
        extrema = image.convert("RGB").getextrema()
        brightest = max(channel_max for _, channel_max in extrema)
        return brightest < 2
    except Exception:
        return False


def load_model(local_encoder=False, full_model=False, gguf_quant=None, flux2=False, schnell=False, for_lora=False, klein=False, klein_4b=False, kontext=False, quantize_encoder=False):
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
        klein_4b: Use the smaller FLUX.2-klein-4B variant instead of the 9B. Implies klein.
        quantize_encoder: FLUX.2 full/klein only. Quantize the text encoder (Qwen3/
            Mistral3) to 4-bit NF4 on load, bitsandbytes-style, like the Kontext 4-bit
            path — the transformer stays bf16 so image quality is unaffected. Needed on
            discrete-VRAM cards where bf16 transformer + bf16 encoder exceed VRAM and
            the driver's sysmem fallback makes generation ~10x slower (klein-9B is
            17GB + 16GB on a 32GB RTX 5090). Not the single-file NVFP4 path above.
        kontext: Use FLUX.1 Kontext, an instruction-based image editor. FLUX.1 only;
            loaded 4-bit (quantized on the fly) with local T5+CLIP encoders. Implies a
            local encoder; ignores flux2/full_model/gguf/schnell.
    """
    global transformer, pipe, _model_type, _flux_version, _schnell_enabled, _kontext_enabled, _local_encoder_active
    if pipe is not None:
        print(f"Warning: model already loaded ({_model_type}); ignoring load_model() "
              f"request. Restart the process to switch models.")
        return {}  # Already loaded

    # Kontext is a standalone FLUX.1 editor model; it overrides the other variants.
    if kontext:
        flux2 = False
        gguf_quant = None
        schnell = False
        local_encoder = True  # Kontext uses local T5+CLIP (no remote API)
        _kontext_enabled = True
        # full_model passes through: --kontext --full-model loads the editor in
        # full bf16 (no 4-bit quantization) for maximum edit fidelity.

    _local_encoder_active = local_encoder

    # klein_4b implies klein; klein implies flux2 + full (bf16). Klein has no working
    # 4-bit/NVFP4 path in current diffusers.
    if klein_4b:
        klein = True
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
        repo_full = (FLUX2_KLEIN_REPO_4B if klein_4b else
                     FLUX2_KLEIN_REPO_FULL if klein else FLUX2_REPO_FULL)
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
    klein_tag = "-klein-4b" if klein_4b else "-klein" if klein else "-dev"
    if kontext:
        _model_type = "kontext-full" if full_model else "kontext"
        model_desc = ("full bf16 FLUX.1-Kontext (editor)" if full_model
                      else "4-bit FLUX.1-Kontext (editor)")
    elif schnell and not flux2:
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

    if kontext and full_model:
        # Full-precision (bf16) FLUX.1-Kontext editor. No quantization: maximum
        # edit fidelity at the cost of VRAM (~24GB transformer + ~9GB T5).
        # Load the whole pipeline in bf16 and move it to the GPU in one shot.
        # (Pre-loading the transformer and passing it alongside device_map left
        # it on CPU while the rest went to cuda:0 -> "tensors on different
        # devices" at inference. .to(device) keeps everything consistent.)
        repo_id = FLUX1_KONTEXT_REPO
        print("Loading FluxKontextPipeline (full bf16)...")
        t0 = time.perf_counter()
        pipe = FluxKontextPipeline.from_pretrained(
            repo_id, torch_dtype=torch_dtype,
        )
        load_timings['transformer'] = 0  # loaded as part of the pipeline
        load_timings['pipeline'] = time.perf_counter() - t0
        print(f"  Pipeline loaded in {load_timings['pipeline']:.2f}s")

        print("Moving FLUX.1-Kontext to GPU...")
        t0 = time.perf_counter()
        pipe = pipe.to(device)
        load_timings['to_device'] = time.perf_counter() - t0
        print(f"  Moved to GPU in {load_timings['to_device']:.2f}s")

    elif kontext:
        # FLUX.1 Kontext editor. No official bnb-4bit repo, so quantize the full
        # repo on the fly: transformer + T5 to 4-bit NF4, CLIP stays small.
        from diffusers import BitsAndBytesConfig as DiffusersBnbConfig
        from transformers import BitsAndBytesConfig as TransformersBnbConfig, T5EncoderModel

        repo_id = FLUX1_KONTEXT_REPO
        print("Loading FLUX.1-Kontext transformer (4-bit, quantized on load)...")
        t0 = time.perf_counter()
        transformer = FluxTransformer2DModel.from_pretrained(
            repo_id, subfolder="transformer",
            quantization_config=DiffusersBnbConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch_dtype),
            torch_dtype=torch_dtype,
        )
        load_timings['transformer'] = time.perf_counter() - t0
        print(f"  Transformer loaded in {load_timings['transformer']:.2f}s")

        print("Loading FLUX.1-Kontext text encoder (T5, 4-bit)...")
        text_encoder_2 = T5EncoderModel.from_pretrained(
            repo_id, subfolder="text_encoder_2",
            quantization_config=TransformersBnbConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch_dtype),
            torch_dtype=torch_dtype,
        )

        print("Assembling FluxKontextPipeline...")
        t0 = time.perf_counter()
        pipe = FluxKontextPipeline.from_pretrained(
            repo_id, transformer=transformer, text_encoder_2=text_encoder_2,
            torch_dtype=torch_dtype, device_map="balanced",
            max_memory=_gpu_max_memory(),
        )
        load_timings['pipeline'] = time.perf_counter() - t0
        load_timings['to_device'] = 0  # Already on GPU via device_map
        print(f"  Pipeline assembled in {load_timings['pipeline']:.2f}s")

    elif gguf_quant:
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
                device_map="balanced", max_memory=_gpu_max_memory()
            )
        else:
            pipe = FluxPipeline.from_pretrained(
                repo_full, transformer=transformer, text_encoder=None,
                text_encoder_2=None, torch_dtype=torch_dtype,
                device_map="balanced", max_memory=_gpu_max_memory()
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

            param_desc = "4B" if klein_4b else "9B" if klein else "32B"
            print(f"  Loading transformer ({param_desc} params)...")
            t_trans = time.perf_counter()
            transformer = Flux2Transformer2DModel.from_pretrained(
                repo_id, subfolder="transformer", torch_dtype=torch_dtype,
                device_map="auto", max_memory=_gpu_max_memory(),
                low_cpu_mem_usage=True, use_safetensors=True
            )
            load_timings['transformer'] = time.perf_counter() - t_trans
            print(f"    Transformer loaded in {load_timings['transformer']:.2f}s")

            t_enc = time.perf_counter()
            if quantize_encoder:
                from transformers import BitsAndBytesConfig as TransformersBnbConfig
                print(f"  Loading text encoder ({text_encoder_label}, 4-bit NF4)...")
                # bnb models can't be .to(device)-moved after load; pin everything
                # to GPU 0 via device_map instead (avoids the "auto" CPU-embedding
                # placement problem below by construction).
                text_encoder = text_encoder_cls.from_pretrained(
                    repo_id, subfolder="text_encoder", torch_dtype=torch_dtype,
                    quantization_config=TransformersBnbConfig(
                        load_in_4bit=True, bnb_4bit_quant_type="nf4",
                        bnb_4bit_compute_dtype=torch_dtype),
                    device_map={"": 0}, use_safetensors=True
                )
            else:
                print(f"  Loading text encoder ({text_encoder_label})...")
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
                max_memory=_gpu_max_memory(),
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
                    device_map="auto", max_memory=_gpu_max_memory(),
                    use_safetensors=True
                )

            def load_text_encoder():
                # T5 doesn't support low_cpu_mem_usage=True with device_map="cuda"
                # (causes meta tensor dispatch error). Use device_map="auto" instead.
                return T5EncoderModel.from_pretrained(
                    repo_id, subfolder="text_encoder_2", torch_dtype=torch_dtype,
                    device_map="auto", max_memory=_gpu_max_memory(),
                    use_safetensors=True
                )

            def load_text_encoder_clip():
                # CLIP is small enough to load directly to CUDA
                return CLIPTextModel.from_pretrained(
                    repo_id, subfolder="text_encoder", torch_dtype=torch_dtype,
                    device_map="auto", max_memory=_gpu_max_memory(),
                    use_safetensors=True
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
                max_memory=_gpu_max_memory(),
                use_safetensors=True,
            )

        # FLUX.2 records the transformer time above; FLUX.1 counts it inside
        # parallel_load, so only default it when nothing recorded it yet.
        load_timings.setdefault('transformer', 0)
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
                    torch_dtype=torch_dtype, device_map="balanced",
                    max_memory=_gpu_max_memory()
                )
            else:
                pipe = Flux2Pipeline.from_pretrained(
                    repo_id, transformer=transformer, text_encoder=None, torch_dtype=torch_dtype,
                    device_map="balanced", max_memory=_gpu_max_memory()
                )
        else:
            # FLUX.1 pipeline
            use_local = local_encoder
            if not use_local and not remote_encoder_available():
                # The remote text-encoder service is unavailable (HF has a
                # history of decommissioning it). Fall back to loading the
                # local T5+CLIP encoders so 4-bit FLUX.1 still works.
                print("Remote text encoder unavailable - falling back to local encoders.")
                use_local = True
                _local_encoder_active = True

            if use_local:
                print("Loading local text encoders (this requires more VRAM)...")
                pipe = FluxPipeline.from_pretrained(
                    repo_id, transformer=transformer, torch_dtype=torch_dtype,
                    device_map="balanced", max_memory=_gpu_max_memory()
                )
            else:
                pipe = FluxPipeline.from_pretrained(
                    repo_id, transformer=transformer, text_encoder=None, torch_dtype=torch_dtype,
                    device_map="balanced", max_memory=_gpu_max_memory()
                )
        load_timings['pipeline'] = time.perf_counter() - t0
        load_timings['to_device'] = 0  # Already on GPU via device_map
        print(f"  Pipeline loaded in {load_timings['pipeline']:.2f}s")

    # Run the VAE in fp32 to avoid intermittent NaN/black decodes (bf16 VAE
    # overflow). Applies to every config; the transformer stays in bf16.
    _stabilize_vae_fp32(pipe)

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
    _fuse_loaded_lora()
    _turbo_enabled = True


def _fuse_loaded_lora():
    """Fuse the just-loaded LoRA into the base weights.

    Unfused adapters add extra matmuls to every transformer forward (~5-10%
    per step). A server process is locked to one model config for its
    lifetime, so nothing ever needs to detach the adapter — fusing is a pure
    win. Falls back to running unfused if fusing isn't supported (e.g. on a
    quantized base model).
    """
    try:
        t0 = time.perf_counter()
        pipe.fuse_lora()
        print(f"  LoRA fused into base weights in {time.perf_counter() - t0:.2f}s")
    except Exception as e:
        print(f"  Warning: could not fuse LoRA, running unfused: {e}")


_uncensored_enabled = False  # Track if uncensored LoRA is loaded

def load_uncensored_lora():
    """Load the Lustly.ai uncensored NSFW LoRA for FLUX.1.

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

    # lustlyai/Flux_Lustly.ai_Uncensored_nsfw_v1 is the most popular dedicated
    # NSFW FLUX.1-dev LoRA (much stronger than aifeifei798/flux-lora-uncensored,
    # which produced weak results). FLUX.1-dev trained, so it needs the full
    # model + real guidance to take full effect (not schnell).
    print("Loading uncensored LoRA (lustlyai/Flux_Lustly.ai_Uncensored_nsfw_v1)...")
    t0 = time.perf_counter()
    pipe.load_lora_weights("lustlyai/Flux_Lustly.ai_Uncensored_nsfw_v1")
    load_time = time.perf_counter() - t0
    print(f"  Uncensored LoRA loaded in {load_time:.2f}s")
    _fuse_loaded_lora()
    _uncensored_enabled = True


REMOTE_ENCODER_URL = "https://remote-text-encoder-flux-2.huggingface.co/predict"


class RemoteEncoderUnavailable(RuntimeError):
    """Raised when the remote text-encoder endpoint does not return embeddings."""


def remote_text_encoder(prompt, use_cache=True):
    if use_cache and prompt in _embedding_cache:
        _embedding_cache.move_to_end(prompt)
        return _embedding_cache[prompt]

    response = _session.post(
        REMOTE_ENCODER_URL,
        json={"prompt": prompt},
        headers={
            "Authorization": f"Bearer {get_token()}",
            "Content-Type": "application/json"
        },
        timeout=(10, 60),  # (connect timeout, read timeout) in seconds
    )
    response.raise_for_status()

    # The endpoint has a history of being decommissioned and silently serving
    # the Hugging Face website (HTTP 206, text/html) instead of embeddings,
    # which produces a cryptic unpickling error. Detect that explicitly so the
    # caller can fall back to the local encoder.
    ctype = response.headers.get("content-type", "")
    if "text/html" in ctype or response.content[:1] in (b"<",):
        raise RemoteEncoderUnavailable(
            f"Remote text encoder returned non-tensor content "
            f"(status {response.status_code}, content-type '{ctype}'). "
            f"The HF remote encoder service appears to be unavailable."
        )

    # PyTorch 2.6 flipped torch.load's `weights_only` default to True, which
    # rejects the pickled tensor payload returned by the remote encoder. The
    # endpoint is HF's official, bearer-token-authenticated service, so the
    # source is trusted and weights_only=False is safe here.
    prompt_embeds = torch.load(io.BytesIO(response.content), weights_only=False)
    result = prompt_embeds.to(device)

    if use_cache:
        _embedding_cache[prompt] = result
        while len(_embedding_cache) > _EMBEDDING_CACHE_MAX:
            _embedding_cache.popitem(last=False)

    return result


def remote_encoder_available():
    """Best-effort probe of the remote text-encoder endpoint.

    Returns True only if it responds with an actual embedding tensor. Used at
    load time to decide whether to fall back to the local encoder.
    """
    try:
        remote_text_encoder("ping", use_cache=False)
        return True
    except Exception as e:
        print(f"  Remote text encoder probe failed: {e}")
        return False

def encode_prompt_once(prompt):
    """Pre-encode ``prompt`` and return pipeline kwargs reusable across calls.

    Generating a batch re-runs the text encoder for every image even though
    the prompt is identical — and FLUX.2's encoders are LLM-sized (Mistral3 /
    Qwen3), so that's seconds of redundant work per image. Callers with a
    multi-image job encode once here and pass the result to generate_image()
    as ``prompt_embeds_kwargs``.

    Returns None when pre-encoding doesn't apply (the FLUX.1 remote-encoder
    path, which already has its own embedding cache) or fails (callers fall
    back to normal per-call encoding).
    """
    if pipe is None:
        raise RuntimeError("Model must be loaded before encoding prompts")
    if _flux_version == 1 and not (_local_encoder_active or _kontext_enabled):
        return None
    try:
        with torch.inference_mode():
            if _flux_version == 2:
                prompt_embeds, _ = pipe.encode_prompt(
                    prompt=prompt, device=device, max_sequence_length=512)
                kwargs = {"prompt_embeds": prompt_embeds}
                # Klein-style pipelines run true CFG with an empty negative
                # prompt unless the checkpoint is distilled (klein-9B is);
                # pre-encode the negative too so CFG runs skip both passes.
                if ("negative_prompt_embeds" in inspect.signature(pipe.__call__).parameters
                        and not getattr(pipe.config, "is_distilled", True)):
                    neg_embeds, _ = pipe.encode_prompt(
                        prompt="", device=device, max_sequence_length=512)
                    kwargs["negative_prompt_embeds"] = neg_embeds
                return kwargs
            # FLUX.1 / Kontext: T5 + CLIP
            prompt_embeds, pooled_prompt_embeds, _ = pipe.encode_prompt(
                prompt=prompt, prompt_2=None, device=device, max_sequence_length=512)
            return {"prompt_embeds": prompt_embeds,
                    "pooled_prompt_embeds": pooled_prompt_embeds}
    except Exception as e:
        print(f"Warning: prompt pre-encoding failed ({e}); using per-image encoding")
        return None


def _apply_prompt_embeds(pipe_kwargs, prompt_embeds_kwargs):
    """Swap a kwargs dict from raw-prompt to precomputed-embeddings form."""
    if prompt_embeds_kwargs:
        pipe_kwargs.pop("prompt", None)
        pipe_kwargs.update(prompt_embeds_kwargs)
    return pipe_kwargs


def _infer_latent_grid(seq, height, width, cell):
    """Recover the true (grid_h, grid_w) token grid for ``seq`` packed latents.

    The pipeline snaps the requested height/width to a multiple of ``cell``
    (it logs "height and width have been adjusted to ..."), so the dimensions
    the preview decoder is handed can disagree with the actual latents by a row
    or column. Trusting them makes the unpack reshape fail. Instead, search a
    small neighborhood around the requested grid for the row count that divides
    ``seq`` exactly (the column count then follows), preferring the candidate
    closest to the requested aspect ratio. A no-op when dimensions already
    match, so non-adjusted FLUX.1/FLUX.2 previews are unchanged.
    """
    gh0 = max(1, round(int(height) / cell))
    gw0 = max(1, round(int(width) / cell))
    if gh0 * gw0 == seq:
        return gh0, gw0
    # Consider every exact factor pair of seq and pick the one whose grid is
    # closest to the requested one. Kontext can rebucket to a quite different
    # resolution, so we don't assume the adjustment is small.
    best = None
    for gh in range(1, int(seq ** 0.5) + 1):
        if seq % gh:
            continue
        for a, b in ((gh, seq // gh), (seq // gh, gh)):
            dist = abs(a - gh0) + abs(b - gw0)
            if best is None or dist < best[0]:
                best = (dist, a, b)
    if best is None:
        raise ValueError(
            f"cannot factor latent seq {seq} near grid {gh0}x{gw0}"
        )
    return best[1], best[2]


def _shrink_latents_for_preview(lat, vae_scale_factor, max_size):
    """Spatially downsample latents so the decoded preview is ~max_size px.

    VAE decode cost scales with output area; a full-resolution fp32 decode of
    a 1-2MP image costs hundreds of ms, which at the preview interval can add
    30-50% to generation time. Previews get downscaled to max_size anyway, so
    decode small instead: shrinking the latents first cuts decode work by the
    square of the scale (4x for 1MP, ~8x for 2MP) for a slightly softer
    preview. No hardcoded per-model constants — works for any conv VAE.
    """
    if not max_size:
        return lat
    out_px = max(lat.shape[-2:]) * vae_scale_factor
    if out_px <= max_size:
        return lat
    return torch.nn.functional.interpolate(
        lat, scale_factor=max_size / out_px, mode="area")


def decode_latents_to_preview(pipe_obj, latents, height, width, max_size=512):
    """Decode intermediate Flux latents into a PIL preview image.

    Handles both FLUX.1 (scaling_factor/shift_factor + _unpack_latents) and
    FLUX.2 (batch-norm stats + _unpatchify_latents). Latents are spatially
    downsampled before decoding so the preview never pays for a full-res
    decode. Returns None on any failure so callers can treat previews as
    best-effort.
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
                B, seq, ch = latents.shape
                # Recover the true grid from the latents; the pipeline may have
                # adjusted the requested height/width to fit model requirements.
                patch_h, patch_w = _infer_latent_grid(
                    seq, height, width, vae_scale_factor * 2
                )
                lat = latents.view(B, patch_h, patch_w, ch).permute(0, 3, 1, 2).contiguous()
                vae = pipe_obj.vae
                bn_mean = vae.bn.running_mean.view(1, -1, 1, 1).to(lat.device, lat.dtype)
                bn_std = torch.sqrt(
                    vae.bn.running_var.view(1, -1, 1, 1) + vae.config.batch_norm_eps
                ).to(lat.device, lat.dtype)
                lat = lat * bn_std + bn_mean
                lat = unpatchify(lat)
                lat = _shrink_latents_for_preview(lat, vae_scale_factor, max_size)
                decoded = vae.decode(lat, return_dict=False)[0]
            else:
                # FLUX.1
                unpack = getattr(pipe_obj, "_unpack_latents", None)
                if unpack is not None:
                    # Use the latents' true grid (the pipeline may have adjusted
                    # the requested height/width), expressed back as pixel dims.
                    cell = vae_scale_factor * 2
                    gh, gw = _infer_latent_grid(latents.shape[1], height, width, cell)
                    lat = unpack(latents, gh * cell, gw * cell, vae_scale_factor)
                else:
                    lat = latents
                if lat.dim() == 4:
                    lat = _shrink_latents_for_preview(lat, vae_scale_factor, max_size)
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


def _prepare_flux2_inpaint(pipe_obj, init_image, mask_image, width, height, seed):
    """Build the tensors needed to inpaint a region of ``init_image`` with FLUX.2.

    FLUX.2 ships no inpainting pipeline, so we re-create the classic masked
    flow-matching approach on top of the standard ``Flux2Pipeline``: encode the
    original image to clean packed latents (``x0``), build a packed latent-space
    mask (1 = regenerate, 0 = keep), and sample a fixed noise tensor used to
    re-noise the kept region at each step. The actual blending happens in the
    step callback returned by :func:`_flux2_inpaint_callback`.

    Returns ``(x0, noise, mask)`` — each a packed ``(1, seq, C)`` tensor on the
    transformer device/dtype (``mask`` is ``(1, seq, 1)``).
    """
    import numpy as np
    from diffusers.utils.torch_utils import randn_tensor

    multiple_of = pipe_obj.vae_scale_factor * 2  # latent token covers a 16px block
    width = (int(width) // multiple_of) * multiple_of
    height = (int(height) // multiple_of) * multiple_of

    # Encode the original image to clean, packed latents (x0).
    img = init_image.convert("RGB").resize((width, height), Image.Resampling.LANCZOS)
    img_t = pipe_obj.image_processor.preprocess(img, height=height, width=width)
    img_t = img_t.to(device=device, dtype=pipe_obj.vae.dtype)
    gen = torch.Generator(device=device).manual_seed(seed)
    x0 = pipe_obj._encode_vae_image(image=img_t, generator=gen)  # (1, C, h16, w16)
    x0 = pipe_obj._pack_latents(x0)                              # (1, seq, C)

    # Build a latent-resolution mask. White (255) = regenerate this token.
    h16, w16 = height // multiple_of, width // multiple_of
    m = mask_image.convert("L").resize((w16, h16), Image.Resampling.BILINEAR)
    m = m.filter(ImageFilter.GaussianBlur(radius=0.5))  # soften latent edges
    m_np = np.asarray(m, dtype=np.float32) / 255.0      # (h16, w16), row-major
    mask = torch.from_numpy(m_np).reshape(1, h16 * w16, 1)  # matches token order

    dtype = pipe_obj.transformer.dtype
    x0 = x0.to(device=device, dtype=dtype)
    mask = mask.to(device=device, dtype=dtype)
    noise = randn_tensor(
        x0.shape,
        generator=torch.Generator(device=device).manual_seed(seed),
        device=torch.device(device),
        dtype=dtype,
    )
    return x0, noise, mask


def _flux2_inpaint_callback(x0, noise, mask, user_callback):
    """Step callback that pins the unmasked region to the original latents.

    After each scheduler step the kept region (mask==0) is overwritten with the
    original latents re-noised to the *next* sigma on the flow-matching path, so
    the model only ever generates inside the painted region while attending to
    consistent surroundings. ``user_callback`` (e.g. live preview) still runs.
    """
    def cb(p, i, t, kwargs):
        latents = kwargs["latents"]
        sigmas = p.scheduler.sigmas
        sigma_next = sigmas[i + 1] if (i + 1) < len(sigmas) else sigmas.new_zeros(())
        sigma_next = sigma_next.to(latents.device, latents.dtype)
        noised_orig = (1.0 - sigma_next) * x0 + sigma_next * noise
        kwargs["latents"] = mask * latents + (1.0 - mask) * noised_orig
        if user_callback is not None:
            res = user_callback(p, i, t, kwargs)
            if res is not None:
                kwargs = res
        return kwargs
    return cb


MAX_REFERENCE_IMAGES = 3


def _stitch_references(images, gap=16):
    """Combine multiple reference images into one side-by-side canvas.

    FLUX.1 Kontext conditions on a single image (diffusers' pipeline treats a
    list as a batch and only uses the first), so multi-reference editing uses
    the standard stitching technique: normalize heights, lay the images out
    left-to-right with a thin white divider, and let the edit instruction
    refer to them positionally ("the person in the left image", "the style of
    the right image"). FLUX.2 doesn't need this — its pipeline conditions on a
    list of images natively.
    """
    imgs = [im.convert("RGB") for im in images]
    target_h = min(max(im.height for im in imgs), 1024)
    resized = []
    for im in imgs:
        w = max(1, round(im.width * target_h / im.height))
        resized.append(im.resize((w, target_h), Image.Resampling.LANCZOS))
    total_w = sum(im.width for im in resized) + gap * (len(resized) - 1)
    canvas = Image.new("RGB", (total_w, target_h), (255, 255, 255))
    x = 0
    for im in resized:
        canvas.paste(im, (x, 0))
        x += im.width + gap
    return canvas


def generate_image(prompt, seed=None, steps=6, width=1024, height=1024, local_encoder=False, input_image=None, strength=0.75, sigmas=None, guidance_scale=None, callback_on_step_end=None, mask_image=None, prompt_embeds_kwargs=None, _retry_depth=0):
    """Generate an image from a text prompt.

    Args:
        prompt: Text description of the image to generate
        seed: Random seed for reproducibility
        steps: Number of inference steps
        width: Output image width
        height: Output image height
        local_encoder: Use local text encoder instead of remote API
        input_image: Optional PIL Image — or a list of up to MAX_REFERENCE_IMAGES
            of them — used as reference(s). The first image is the primary (it
            drives the output aspect ratio). Multiple references need FLUX.2
            (conditions on the list natively) or Kontext (references are
            stitched into one canvas; address them as left/middle/right in the
            instruction). FLUX.1 img2img takes a single image.
        strength: Denoising strength for img2img (0.0-1.0, higher = more change)
        sigmas: Custom noise schedule (for turbo LoRA, use TURBO_SIGMAS)
        guidance_scale: Classifier-free guidance scale (default: 4, turbo uses 2.5)
        callback_on_step_end: Optional callback called after each inference step.
            Signature: callback(pipe, step_index, timestep, callback_kwargs) -> callback_kwargs
        mask_image: Optional PIL Image (L/RGB) for inpainting. White pixels mark
            the region to regenerate; black pixels are preserved. Requires
            input_image and FLUX.2 (the FLUX.2 family has no inpaint pipeline, so
            this is done via masked latent re-injection on Flux2Pipeline).
        prompt_embeds_kwargs: Optional dict from encode_prompt_once(prompt) —
            precomputed text embeddings for this same prompt, used instead of
            re-encoding. Callers generating batches pass this to skip a
            text-encoder forward per image. Ignored on the remote-encoder path.
    """
    global pipe_img2img

    if mask_image is not None and _flux_version != 2:
        raise ValueError(
            "Inpainting (mask_image) is only supported on FLUX.2. "
            "Start the server/CLI with --flux2."
        )

    # Normalize the reference input to a list. The first image is the primary
    # reference; extra images are only meaningful for FLUX.2 (native
    # multi-reference) and Kontext (stitched into one conditioning canvas).
    if isinstance(input_image, (list, tuple)):
        ref_images = [im for im in input_image if im is not None]
    elif input_image is not None:
        ref_images = [input_image]
    else:
        ref_images = []
    input_image = ref_images[0] if ref_images else None

    if len(ref_images) > MAX_REFERENCE_IMAGES:
        raise ValueError(f"At most {MAX_REFERENCE_IMAGES} reference images are supported.")
    if len(ref_images) > 1:
        if mask_image is not None:
            raise ValueError("Inpainting (mask_image) uses exactly one input image.")
        if not (_kontext_enabled or _flux_version == 2):
            raise ValueError(
                "Multiple reference images require the Kontext editor or FLUX.2; "
                "FLUX.1 img2img takes a single image."
            )

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

    # Ensure strength won't result in zero pipeline steps for img2img.
    # Kontext ignores strength (it edits from the instruction), so skip the check.
    if input_image is not None and strength is not None and not _kontext_enabled:
        min_strength = 1.0 / steps
        if strength < min_strength:
            print(f"Warning: strength {strength} too low for {steps} steps (minimum {min_strength:.2f}). Using {min_strength:.2f}.")
            strength = min_strength

    if guidance_scale is None:
        # Turbo and Kontext use 2.5, others use 4
        if _turbo_enabled or _kontext_enabled:
            guidance_scale = 2.5
        else:
            guidance_scale = 4

    timings = {}

    # Generate with inference_mode for better performance
    with torch.inference_mode():
        if input_image is not None:
            # Img2img / editing mode
            if _kontext_enabled:
                # FLUX.1 Kontext: instruction-based editing. The prompt is an
                # edit instruction ("make the car red") and the image is the
                # source; Kontext keeps everything the instruction doesn't touch.
                # No strength — the model decides what to change.
                #
                # The Kontext pipeline snaps the *conditioning* image to its
                # nearest preferred-resolution bucket, but if we don't pass
                # height/width it defaults the *output* latent to 1024x1024
                # (square) regardless of the input aspect ratio — so every edit
                # came out square. Replicate the pipeline's bucket selection here
                # and pass the matching height/width so the output keeps the
                # reference image's aspect ratio (and stays consistent with the
                # resized conditioning).
                try:
                    from diffusers.pipelines.flux.pipeline_flux_kontext import (
                        PREFERRED_KONTEXT_RESOLUTIONS,
                    )
                except ImportError:
                    PREFERRED_KONTEXT_RESOLUTIONS = None

                kontext_dims = {}
                if PREFERRED_KONTEXT_RESOLUTIONS:
                    src = input_image.convert("RGB")
                    in_w, in_h = src.size
                    aspect_ratio = in_w / in_h
                    _, best_w, best_h = min(
                        (abs(aspect_ratio - w / h), w, h)
                        for w, h in PREFERRED_KONTEXT_RESOLUTIONS
                    )
                    multiple_of = 16
                    kontext_dims = {
                        "width": (best_w // multiple_of) * multiple_of,
                        "height": (best_h // multiple_of) * multiple_of,
                    }

                # Multiple references get stitched into one canvas; the output
                # dims above still come from the primary image, so the edit
                # result keeps the primary's aspect ratio.
                cond_image = (_stitch_references(ref_images)
                              if len(ref_images) > 1 else input_image.convert("RGB"))

                t0 = time.perf_counter()
                kontext_kwargs = {
                    "prompt": prompt,
                    "image": cond_image,
                    "generator": torch.Generator(device=device).manual_seed(seed),
                    "num_inference_steps": steps,
                    "guidance_scale": guidance_scale,
                    "max_sequence_length": 512,
                    **kontext_dims,
                }
                if callback_on_step_end is not None:
                    kontext_kwargs["callback_on_step_end"] = callback_on_step_end
                image = pipe(**_apply_prompt_embeds(kontext_kwargs, prompt_embeds_kwargs)).images[0]
                timings['diffusion'] = time.perf_counter() - t0
                timings['encoding'] = 0
            elif _flux_version == 2 and mask_image is not None:
                # FLUX.2 inpainting: no dedicated pipeline exists, so we run the
                # standard txt2img loop and re-inject the original latents outside
                # the painted mask at every step (flow-matching masked diffusion).
                multiple_of = pipe.vae_scale_factor * 2
                width = (width // multiple_of) * multiple_of
                height = (height // multiple_of) * multiple_of

                x0, noise, mask = _prepare_flux2_inpaint(
                    pipe, input_image, mask_image, width, height, seed)
                inpaint_cb = _flux2_inpaint_callback(x0, noise, mask, callback_on_step_end)

                t0 = time.perf_counter()
                pipe_kwargs = {
                    "prompt": prompt,
                    "generator": torch.Generator(device=device).manual_seed(seed),
                    "num_inference_steps": steps,
                    "guidance_scale": guidance_scale,
                    "height": height,
                    "width": width,
                    "max_sequence_length": 512,
                    "callback_on_step_end": inpaint_cb,
                }
                if sigmas is not None:
                    pipe_kwargs["sigmas"] = sigmas
                image = pipe(**_apply_prompt_embeds(pipe_kwargs, prompt_embeds_kwargs)).images[0]

                # Paste the untouched original back outside the mask: pinning the
                # latents preserves structure, but the VAE round-trip can still
                # shift kept pixels slightly. A feathered pixel composite keeps
                # everything outside the painted region byte-for-byte original.
                orig_rgb = input_image.convert("RGB").resize((width, height), Image.Resampling.LANCZOS)
                mask_px = mask_image.convert("L").resize((width, height), Image.Resampling.BILINEAR)
                feather = max(1, round(max(width, height) / 256))
                mask_px = mask_px.filter(ImageFilter.GaussianBlur(radius=feather))
                image = Image.composite(image, orig_rgb, mask_px)

                timings['diffusion'] = time.perf_counter() - t0
                timings['encoding'] = 0
            elif _flux_version == 2:
                # FLUX.2 has image conditioning built into the main pipeline
                # (no strength parameter - image is used as reference/conditioning).
                # It accepts a list of references natively: each is VAE-encoded
                # separately (the pipeline caps them at ~1MP) and their tokens
                # are concatenated, so pass multiple images through as-is. A
                # single reference keeps the resize-to-output behavior.
                if len(ref_images) > 1:
                    pipe_image = [im.convert("RGB") for im in ref_images]
                else:
                    pipe_image = input_image.resize((width, height), Image.Resampling.LANCZOS)

                t0 = time.perf_counter()
                pipe_kwargs = {
                    "prompt": prompt,
                    "image": pipe_image,
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
                image = pipe(**_apply_prompt_embeds(pipe_kwargs, prompt_embeds_kwargs)).images[0]
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
                    # Keep the img2img VAE in fp32 too (shares the txt2img VAE,
                    # so this is a no-op when already stabilized).
                    _stabilize_vae_fp32(pipe_img2img)

                input_image = input_image.resize((width, height), Image.Resampling.LANCZOS)

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
                image = pipe_img2img(**_apply_prompt_embeds(img2img_kwargs, prompt_embeds_kwargs)).images[0]
                timings['diffusion'] = time.perf_counter() - t0
                timings['encoding'] = 0
        elif local_encoder or _local_encoder_active or _flux_version == 2:
            # Use local text encoder (incl. remote-encoder fallback)
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
            image = pipe(**_apply_prompt_embeds(pipe_kwargs, prompt_embeds_kwargs)).images[0]
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

    # A degenerate (all-black) result means the latents/VAE diverged (NaN or
    # saturation). Retry with a perturbed seed; if it still fails, raise so the
    # caller surfaces an error instead of silently saving a black image.
    if _is_degenerate_image(image):
        if _retry_depth < _MAX_GEN_RETRIES:
            retry_seed = (seed + 1) % (2 ** 32)
            print(f"Warning: degenerate (all-black) image from seed {seed}; "
                  f"retrying with seed {retry_seed} "
                  f"(attempt {_retry_depth + 2}/{_MAX_GEN_RETRIES + 1})", flush=True)
            return generate_image(
                prompt, seed=retry_seed, steps=steps, width=width, height=height,
                local_encoder=local_encoder,
                input_image=(ref_images if len(ref_images) > 1 else input_image),
                strength=strength,
                sigmas=sigmas, guidance_scale=guidance_scale,
                callback_on_step_end=callback_on_step_end, mask_image=mask_image,
                prompt_embeds_kwargs=prompt_embeds_kwargs,
                _retry_depth=_retry_depth + 1)
        raise DegenerateImageError(
            f"Generation produced an all-black image after {_MAX_GEN_RETRIES + 1} "
            f"attempts (likely a VAE/transformer NaN); not saving.")

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
        # guidance_scale == 0 is a real value (schnell forces it), not "auto"
        f.write(f"# Guidance: {guidance_scale if guidance_scale is not None else 'auto'}\n")
        if strength is not None:
            f.write(f"# Strength: {strength} (img2img)\n")
        f.write(f"# Timings: encoding={timings['encoding']:.2f}s, diffusion={timings['diffusion']:.2f}s, save={timings.get('save', 0):.2f}s\n")
    return prompt_path
