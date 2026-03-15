# FLUX Server Options

Run with `./run_server.sh` for an interactive menu, or `./run_server.sh <number>` to launch directly.

## FLUX.1 Models

| # | Name | Flags | Description |
|---|------|-------|-------------|
| **1** | FLUX.1 4-bit BNB | *(none)* | 4-bit quantized FLUX.1-dev (`diffusers/FLUX.1-dev-bnb-4bit`). Lowest VRAM usage. Uses remote Hugging Face text encoder API for prompt embeddings. Good balance of quality and resource usage. |
| **2** | FLUX.1 Full | `--full-model` | Full-precision FLUX.1-dev (`black-forest-labs/FLUX.1-dev`). Highest quality for FLUX.1. Requires significantly more VRAM. Uses local text encoder. |
| **3** | FLUX.1 GGUF Q8 | `--gguf q8 --local-encoder` | GGUF Q8 quantized FLUX.1 (`city96/FLUX.1-dev-gguf`). Optimized for DGX Spark / unified memory systems. Sets `TORCH_CUDA_ARCH_LIST=12.1` for Blackwell GPUs. Uses local text encoder. |
| **4** | FLUX.1-schnell | `--schnell --local-encoder` | FLUX.1-schnell (`black-forest-labs/FLUX.1-schnell`). Distilled model that generates in just 4 steps with `guidance_scale=0` (no classifier-free guidance). Much faster but lower quality. Apache 2.0 licensed. Uses local text encoder. |
| **8** | FLUX.1 + Uncensored | `--uncensored` | FLUX.1 4-bit with the Flux-Uncensored-V2 LoRA (`enhanceaiteam/Flux-Uncensored-V2`) applied. Removes content filters from generation. FLUX.1 only. |

## FLUX.2 Models

| # | Name | Flags | Description |
|---|------|-------|-------------|
| **5** | FLUX.2 4-bit BNB | `--flux2` | 4-bit quantized FLUX.2-dev (`diffusers/FLUX.2-dev-bnb-4bit`). FLUX.2 is a 32B parameter model (vs 12B for FLUX.1). Automatically uses local Mistral3 text encoder (no remote API available for FLUX.2). |
| **6** | FLUX.2 Full | `--flux2 --full-model` | Full-precision FLUX.2-dev (`black-forest-labs/FLUX.2-dev`). Highest quality available. Very high VRAM requirement. Turbo LoRA enabled by default. |
| **7** | FLUX.2 Full + Turbo | `--flux2 --full-model --turbo` | Full-precision FLUX.2 with turbo LoRA explicitly enabled (`fal/FLUX.2-dev-Turbo`). Uses custom 8-step noise schedule for ~3x faster inference with minimal quality loss. Guidance scale defaults to 2.5 (vs 4.0 standard). |
| **9** | FLUX.2 Full (no Turbo) | `--flux2 --full-model --no-turbo` | Full-precision FLUX.2 without turbo LoRA. Maximum quality at the cost of slower generation (25 steps default vs 8). Use this when quality matters more than speed. |

## Notes

- All servers launch on **port 2222** with a web UI
- Auto-restart is enabled (up to 5 retries on crash, including OOM kills)
- FLUX.2 models require substantially more VRAM than FLUX.1
- The turbo LoRA is enabled by default for FLUX.2 full models (option 6 and 7 behave the same); use option 9 to disable it
- GGUF and schnell modes are FLUX.1 only; uncensored LoRA is also FLUX.1 only
