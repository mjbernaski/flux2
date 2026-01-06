import torch
import time

# Performance optimizations for Blackwell/DGX Spark GPUs
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True

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
print("Applied safetensors unified memory patch")

from diffusers import Flux2Pipeline

# Pre-shifted custom sigmas for 8-step turbo inference
TURBO_SIGMAS = [1.0, 0.6509, 0.4374, 0.2932, 0.1893, 0.1108, 0.0495, 0.00031]

timings = {}
total_start = time.perf_counter()

# Phase 1: Load pipeline from pretrained
print("=" * 60)
print("Phase 1: Loading FLUX.2-dev pipeline...")
t0 = time.perf_counter()
pipe = Flux2Pipeline.from_pretrained(
    "black-forest-labs/FLUX.2-dev",
    torch_dtype=torch.bfloat16,
)
timings['load_pipeline'] = time.perf_counter() - t0
print(f"  Pipeline loaded in {timings['load_pipeline']:.2f}s")

# Phase 2: Move to GPU
print("=" * 60)
print("Phase 2: Moving pipeline to CUDA...")
t0 = time.perf_counter()
pipe = pipe.to("cuda")
timings['to_cuda'] = time.perf_counter() - t0
print(f"  Moved to CUDA in {timings['to_cuda']:.2f}s")

# Phase 3: Load LoRA weights
print("=" * 60)
print("Phase 3: Loading turbo LoRA weights...")
t0 = time.perf_counter()
pipe.load_lora_weights(
    "fal/FLUX.2-dev-Turbo",
    weight_name="flux.2-turbo-lora.safetensors"
)
timings['load_lora'] = time.perf_counter() - t0
print(f"  LoRA loaded in {timings['load_lora']:.2f}s")

prompt = "Industrial product shot of a chrome turbocharger with glowing hot exhaust manifold, engraved text 'FLUX.2 [dev] Turbo by fal' on the compressor housing and 'fal' on the turbine wheel, gradient heat glow from orange to electric blue , studio lighting with dramatic shadows, shallow depth of field, engineering blueprint pattern in background."

# Phase 4: Generation (encoding + diffusion)
print("=" * 60)
print("Phase 4: Generating image (encoding + 8 diffusion steps)...")
t0 = time.perf_counter()
with torch.inference_mode():
    image = pipe(
        prompt=prompt,
        sigmas=TURBO_SIGMAS,
        guidance_scale=2.5,
        height=1024,
        width=1024,
        num_inference_steps=8,
    ).images[0]
timings['generation'] = time.perf_counter() - t0
print(f"  Generation completed in {timings['generation']:.2f}s")

# Phase 5: Save image
print("=" * 60)
print("Phase 5: Saving image...")
t0 = time.perf_counter()
image.save("output.png")
timings['save'] = time.perf_counter() - t0
print(f"  Image saved in {timings['save']:.2f}s")

# Summary
timings['total'] = time.perf_counter() - total_start
print("=" * 60)
print("TIMING SUMMARY")
print("=" * 60)
print(f"  1. Load pipeline:  {timings['load_pipeline']:>7.2f}s")
print(f"  2. Move to CUDA:   {timings['to_cuda']:>7.2f}s")
print(f"  3. Load LoRA:      {timings['load_lora']:>7.2f}s")
print(f"  4. Generation:     {timings['generation']:>7.2f}s")
print(f"  5. Save image:     {timings['save']:>7.2f}s")
print("-" * 60)
print(f"  TOTAL:             {timings['total']:>7.2f}s")
print("=" * 60)
print("Saved to output.png")
