# FLUX Server Options

Run with `./run_server.sh` for an interactive menu, or `./run_server.sh <number>` to launch directly.

> Numbers below match `run_server.sh`. Keep this table in sync with the `case`
> statement in that script if the menu is ever renumbered.

## FLUX.1 Models (12B)

| # | Name | Flags | Description |
|---|------|-------|-------------|
| **1** | FLUX.1 4-bit BNB | *(none)* | 4-bit quantized FLUX.1-dev (`diffusers/FLUX.1-dev-bnb-4bit`). Lowest VRAM usage. Uses remote Hugging Face text encoder API for prompt embeddings. Good balance of quality and resource usage. |
| **2** | FLUX.1 Full | `--full-model` | Full-precision FLUX.1-dev (`black-forest-labs/FLUX.1-dev`). Highest quality for FLUX.1. Requires significantly more VRAM. Uses local text encoder. |
| **3** | FLUX.1 GGUF Q8 | `--gguf q8 --local-encoder` | GGUF Q8 quantized FLUX.1 (`city96/FLUX.1-dev-gguf`). Optimized for DGX Spark / unified memory systems. Sets `TORCH_CUDA_ARCH_LIST=12.1` for Blackwell GPUs. Uses local text encoder. |
| **4** | FLUX.1-schnell | `--schnell --local-encoder` | FLUX.1-schnell (`black-forest-labs/FLUX.1-schnell`). Distilled model that generates in just 4 steps with `guidance_scale=0` (no classifier-free guidance). Much faster but lower quality. Apache 2.0 licensed. Uses local text encoder. |
| **5** | FLUX.1 + Uncensored | `--uncensored` | Full FLUX.1-dev with an uncensored LoRA (`lustlyai/Flux_Lustly.ai_Uncensored_nsfw_v1`) applied (`--uncensored` implies the full model). Removes content filters from generation. FLUX.1 only. |

## FLUX.2 Models (32B · klein 9B)

| # | Name | Flags | Description |
|---|------|-------|-------------|
| **6** | FLUX.2 4-bit BNB | `--flux2` | 4-bit quantized FLUX.2-dev (`diffusers/FLUX.2-dev-bnb-4bit`). FLUX.2 is a 32B parameter model (vs 12B for FLUX.1). Automatically uses local Mistral3 text encoder (no remote API available for FLUX.2). |
| **7** | FLUX.2 Full + Turbo | `--flux2 --full-model --turbo` | Full-precision FLUX.2 with turbo LoRA enabled (`fal/FLUX.2-dev-Turbo`). Uses custom 8-step noise schedule for ~3x faster inference with minimal quality loss. Guidance scale defaults to 2.5 (vs 4.0 standard). |
| **8** | FLUX.2 Full (no Turbo) | `--flux2 --full-model --no-turbo` | Full-precision FLUX.2-dev (`black-forest-labs/FLUX.2-dev`) without the turbo LoRA. Maximum quality at the cost of slower generation (25 steps default vs 8). Very high VRAM requirement; slowest to load. |
| **9** | FLUX.2-klein-9B | `--klein` | FLUX.2-klein, a faster 9B variant. Default menu selection. Lower VRAM and faster than the full 32B FLUX.2 models. |

## Editing

| # | Name | Flags | Description |
|---|------|-------|-------------|
| **10** | FLUX.1 Kontext | `--kontext` | Instruction-based image editor (`black-forest-labs/FLUX.1-Kontext-dev`, quantized to 4-bit on load). Upload a reference image and write the prompt as an **edit instruction** (e.g. "make the car red", "remove the sign") — it changes only what you ask and keeps the rest. Unlike plain img2img (the reference-image field on the other models), it does true targeted edits. The strength slider does not apply; output keeps the source aspect ratio. First launch downloads the full model (~24GB) one time. |
| **11** | FLUX.1 Kontext Full | `--kontext --full-model` | The Kontext editor in full bf16 (no quantization) for maximum edit fidelity. Substantially more VRAM (~24GB transformer + ~9GB T5). |
| **12** | Kontext Full + Uncensored | `--kontext --full-model --uncensored` | Kontext Full with the uncensored LoRA loaded, for editing reference images without content filters. |

## Stable Diffusion (SDXL)

| # | Name | Flags | Description |
|---|------|-------|-------------|
| **13** | SDXL Uncensored | `--sdxl` | Uncensored Stable Diffusion XL via the `sd_core.py` backend (default checkpoint `John6666/lustify-sdxl-nsfwsfw-v2-sdxl`, the photorealistic LUSTIFY! NSFW/SFW merge, bf16, ~7GB). Unlike FLUX, SDXL checkpoints have no instruction-refusal behavior — but they also have **no instruction understanding**: prompts must *describe the desired final image*, not command an edit. Enables the **negative prompt** field in the UI (a quality-boilerplate default applies when left empty); guidance defaults to 6.0 (real CFG); dimensions snap to SDXL's 64-px training buckets. The shipped scheduler config of auto-converted Civitai repos is corrected at load (EDM → Euler Ancestral; the EDM config produces pure noise). Supports txt2img, single-image img2img via the strength slider, and **masked inpainting** — the paint-a-mask UI works here too, lazy-loading the dedicated LUSTIFY v2.0 INPAINTING checkpoint (`SD_INPAINT_MODEL`, default `andro-flock/LUSTIFY-SDXL-NSFW-checkpoint-v2-0-INPAINTING`, ~7GB on first masked job); in inpaint mode the strength slider is the denoise level for the painted region (~0.4–0.7 edits, higher replaces). No Kontext-style instruction editing. Override the base checkpoint with `SD_MODEL=<hf-repo-or-path>` or `./run_server.sh 13 --sdxl /path/to/checkpoint.safetensors` (single-file Civitai downloads work). Outputs are named `sdxl_*.png`. |

## Notes

- All servers launch on **port 2222** with a web UI
- Extra flags pass through the launcher, e.g. `./run_server.sh 9 --compile`:
  torch.compile makes the first generation per resolution much slower but
  later ones ~10-25% faster — best for sessions at a consistent resolution
- Batch jobs encode the prompt once and reuse the embeddings for every image;
  LoRAs (turbo/uncensored) are fused into the base weights at load
- Up to **3 reference images** per generation: FLUX.2 (6-9) uses them natively;
  Kontext (10-12) stitches them side-by-side (address them as left/middle/right
  in the instruction); FLUX.1 img2img (1-5) takes a single reference
- Auto-restart is enabled (up to 5 retries on crash, including OOM kills)
- FLUX.2 models require substantially more VRAM than FLUX.1
- GGUF and schnell modes are FLUX.1 only; the uncensored LoRA is also FLUX.1 only
- The full FLUX.2 models (7 and 8) are 32B and can take ~10+ minutes to cold-load from disk
