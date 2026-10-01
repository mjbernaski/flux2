# FLUX Server Options

Run with `./run_server.sh` for an interactive menu, or `./run_server.sh <number>` to launch directly.

> Numbers below match `run_server.sh`. Keep this table in sync with the
> `set_config_args` case statement in that script — and the `SERVER_CONFIGS`
> table in `web_server.py` — if the menu is ever renumbered.

## FLUX.1 Models (12B)

| # | Name | Flags | Description |
|---|------|-------|-------------|
| **1** | FLUX.1 4-bit BNB | *(none)* | 4-bit quantized FLUX.1-dev (`diffusers/FLUX.1-dev-bnb-4bit`). Lowest VRAM usage. Uses remote Hugging Face text encoder API for prompt embeddings. Good balance of quality and resource usage. |
| **2** | FLUX.1 Full | `--full-model` | Full-precision FLUX.1-dev (`black-forest-labs/FLUX.1-dev`). Highest quality for FLUX.1. Requires significantly more VRAM. Uses local text encoder. |
| **3** | FLUX.1 GGUF Q8 | `--gguf q8 --local-encoder` | GGUF Q8 quantized FLUX.1 (`city96/FLUX.1-dev-gguf`). Optimized for DGX Spark / unified memory systems. Sets `TORCH_CUDA_ARCH_LIST=12.1` for Blackwell GPUs. Uses local text encoder. |
| **4** | FLUX.1-schnell | `--schnell --local-encoder` | FLUX.1-schnell (`black-forest-labs/FLUX.1-schnell`). Distilled model that generates in just 4 steps with `guidance_scale=0` (no classifier-free guidance). Much faster but lower quality. Apache 2.0 licensed. Uses local text encoder. |
| **5** | FLUX.1 + U-LoRA | `--uncensored` | Full FLUX.1-dev with an uncensored LoRA (`lustlyai/Flux_Lustly.ai_Uncensored_nsfw_v1`) applied (`--uncensored` implies the full model). Removes content filters from generation. FLUX.1 only. |

## FLUX.2 Models (32B · klein 9B/4B)

| # | Name | Flags | Description |
|---|------|-------|-------------|
| **6** | FLUX.2 4-bit BNB | `--flux2` | 4-bit quantized FLUX.2-dev (`diffusers/FLUX.2-dev-bnb-4bit`). FLUX.2 is a 32B parameter model (vs 12B for FLUX.1). Automatically uses local Mistral3 text encoder (no remote API available for FLUX.2). |
| **7** | FLUX.2 Full + Turbo | `--flux2 --full-model --turbo` | Full-precision FLUX.2 with turbo LoRA enabled (`fal/FLUX.2-dev-Turbo`). Uses custom 8-step noise schedule for ~3x faster inference with minimal quality loss. Guidance scale defaults to 2.5 (vs 4.0 standard). |
| **8** | FLUX.2 Full (no Turbo) | `--flux2 --full-model --no-turbo` | Full-precision FLUX.2-dev (`black-forest-labs/FLUX.2-dev`) without the turbo LoRA. Maximum quality at the cost of slower generation (25 steps default vs 8). Very high VRAM requirement; slowest to load. |
| **9** | FLUX.2-klein-9B | `--klein` | FLUX.2-klein, a faster 9B variant. Default menu selection. Lower VRAM and faster than the full 32B FLUX.2 models. |
| **14** | FLUX.2-klein-4B | `--klein-4b` | FLUX.2-klein-4B (`black-forest-labs/FLUX.2-klein-4B`), the smallest FLUX.2 variant. Same `Flux2KleinPipeline` + Qwen3 text encoder as the 9B, fewer transformer params. ~13GB VRAM full bf16 for the transformer — but the Qwen3 text encoder is a further ~16GB in bf16, the same as the 9B's, so total residency is ~30GB and the transformer size is misleading on a discrete card. Add `--quantize-encoder` (NF4 encoder, ~5GB) when VRAM is tight; `run_server.ps1` does this by default, `run_server.sh` does not (unified memory). Still local-encoder only, no remote API. |

## Editing

| # | Name | Flags | Description |
|---|------|-------|-------------|
| **10** | FLUX.1 Kontext | `--kontext` | Instruction-based image editor (`black-forest-labs/FLUX.1-Kontext-dev`, quantized to 4-bit on load). Upload a reference image and write the prompt as an **edit instruction** (e.g. "make the car red", "remove the sign") — it changes only what you ask and keeps the rest. Unlike plain img2img (the reference-image field on the other models), it does true targeted edits. The strength slider does not apply; output keeps the source aspect ratio. First launch downloads the full model (~24GB) one time. |
| **11** | FLUX.1 Kontext Full | `--kontext --full-model` | The Kontext editor in full bf16 (no quantization) for maximum edit fidelity. Substantially more VRAM (~24GB transformer + ~9GB T5). |
| **12** | Kontext Full + U-LoRA | `--kontext --full-model --uncensored` | Kontext Full with the uncensored LoRA loaded, for editing reference images without content filters. |

## Stable Diffusion (SDXL)

| # | Name | Flags | Description |
|---|------|-------|-------------|
| **13** | SDXL (photoreal) | `--sdxl` | Uncensored Stable Diffusion XL via the `sd_core.py` backend (default checkpoint `John6666/lustify-sdxl-nsfwsfw-v2-sdxl`, a photorealistic merge, bf16, ~7GB). UI-facing names for this config (menu, model switcher, model-name hover) deliberately avoid the checkpoint's brand name. Unlike FLUX, SDXL checkpoints have no instruction-refusal behavior — but they also have **no instruction understanding**: prompts must *describe the desired final image*, not command an edit. Enables the **negative prompt** field in the UI (a quality-boilerplate default applies when left empty); guidance defaults to 6.0 (real CFG); dimensions snap to SDXL's 64-px training buckets. The shipped scheduler config of auto-converted Civitai repos is corrected at load (EDM → Euler Ancestral; the EDM config produces pure noise). Supports txt2img, single-image img2img via the strength slider, and **masked inpainting** — the paint-a-mask UI works here too, lazy-loading the dedicated inpainting checkpoint (`SD_INPAINT_MODEL`, default `andro-flock/LUSTIFY-SDXL-NSFW-checkpoint-v2-0-INPAINTING`, ~7GB on first masked job); in inpaint mode the strength slider is the denoise level for the painted region (~0.4–0.7 edits, higher replaces). No Kontext-style instruction editing. Override the base checkpoint with `SD_MODEL=<hf-repo-or-path>` or `./run_server.sh 13 --sdxl /path/to/checkpoint.safetensors` (single-file Civitai downloads work). Outputs are named `sdxl_*.png`. |

## Qwen

| # | Name | Flags | Description |
|---|------|-------|-------------|
| **15** | Qwen-Image 2.1 | `--qwen` | Qwen-Image 2.1 (`Qwen/Qwen-Image-2.1`, ~31GB) via the `qwen_core.py` backend: a 7B single-stream DiT with a Qwen3-VL 8B text encoder and a 64-channel RGBA VAE. **One model does both jobs** — with no reference it is text-to-image, with references it is an instruction editor, so unlike the FLUX split there is no separate editor config. Takes **up to 10 reference images** (the UI raises its own limit from `/model-info`), and because the *text encoder* sees each reference as vision context, they can be referred to in the prompt ("the hat from the second image"). **Native transparency**: a Transparent background checkbox appears in the UI and returns a real RGBA PNG — alpha is kept only when the result actually uses it, so ordinary jobs stay RGB. **Masked editing** works through the paint-a-mask UI, but not as diffusion inpainting: the pipeline has no mask input, so the painted region is ringed in magenta *just outside the mask* (the model is trained to act on annotated images) and everything outside the mask is composited back from the source, which is what makes unpainted pixels pixel-exact. Enables the **negative prompt** field, which switches on true CFG at the guidance value (default 4.0 via `QWEN_TRUE_CFG`); with the field empty guidance is 1.0 — the model is meant to be sampled without it, and CFG costs a second transformer pass per step. The **strength** slider does nothing here (conditioning is by token, not by partial noising) and is dropped from the request like it is on FLUX.2. One consequence for the **spectrum grid**, which sweeps guidance against strength: with the negative prompt empty neither axis is live, so the cells come back identical — fill in a negative prompt first, or use a batch instead. Dimensions snap to 32px; the model is 2K-native, so `2mp` sizes are in its comfort zone. Prefix KV caching is on (`QWEN_KV_CACHE=0` disables it; a seed only reproduces at a fixed setting). Steps default to 40 in the core. Other env knobs: `QWEN_MODEL` (checkpoint), `QWEN_LORA` (fused at load), `QWEN_CONDITION_RESOLUTION` (reference token budget, default 1024). Outputs are named `qwen_*.png`. **VRAM**: 7B transformer + 8B encoder is ~30GB bf16 — `run_server.ps1` adds `--quantize-encoder` (NF4 encoder, ~20GB total) for 32GB cards, and `--cpu-offload` is the fallback that fits anywhere. Requires a diffusers build with `QwenImage21Pipeline` (merged upstream 2026-09-18, after the 0.40.0 release). |

## Notes

- All servers launch on **port 2222** with a web UI
- Extra flags pass through the launcher, e.g. `./run_server.sh 9 --compile`:
  torch.compile makes the first generation per resolution much slower but
  later ones ~10-25% faster — best for sessions at a consistent resolution
- Batch jobs encode the prompt once and reuse the embeddings for every image;
  LoRAs (turbo/uncensored) are fused into the base weights at load
- Reference images per generation are a property of the backend, published by
  `/model-info` and followed by the UI: **10** on Qwen (15), which reads them
  through its text encoder; **3** on FLUX.2 (6-9), which uses them natively,
  and on Kontext (10-12), which stitches them side-by-side (address them as
  left/middle/right in the instruction); **1** on FLUX.1 img2img (1-5)
- Auto-restart is enabled (up to 5 retries on crash, including OOM kills)
- **Switching models on the fly**: when launched via `run_server.sh`, the web
  UI shows a Model dropdown. Picking another config restarts the server under
  the supervisor with that config's flags (queue must be idle — interrupt or
  cancel jobs first). A running generation can be stopped mid-step with the
  UI's Interrupt button.
- FLUX.2 models require substantially more VRAM than FLUX.1
- GGUF and schnell modes are FLUX.1 only; the uncensored LoRA is also FLUX.1 only
- The full FLUX.2 models (7 and 8) are 32B and can take ~10+ minutes to cold-load from disk
