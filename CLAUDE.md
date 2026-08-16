# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

FLUX Image Generator — self-hosted image generation on FLUX.1/FLUX.2 diffusion
models via `diffusers`, tuned for DGX Spark / unified-memory systems. Three
entry points share one model core:

- **`flux_core.py`** — the model core (loading, generation, LoRAs, latent
  previews, FLUX.2 inpainting). `web_server.py`, `flux_cli.py`, and the smoke
  test import from it; functions like `load_turbo_lora`,
  `decode_latents_to_preview`, and the inpaint helpers are only exercised by
  the server, so don't assume code unused by the CLI is dead.
- **`flux_cli.py`** — the interactive CLI REPL front end for `flux_core`.
- **`web_server.py`** — Flask API + single-worker generation queue on port
  2222. Auth via `FLUX_API_KEY` (X-API-Key header / `api_key` param). The UI is
  static files in `static/` (index.html, app.css, app.js) — no build step.
- **`rest_api.py`** — the `/api/v1` REST layer (a blueprint registered by
  `web_server.py`). It holds no logic: both it and the legacy flat routes call
  the `_api_*` core functions in `web_server.py`, so behavior lives in exactly
  one place. Adding an endpoint means extracting an `_api_*` function and
  wiring both dialects to it. See REST_API.md and `examples/`.
- **`image_manager.py`** — separate Flask gallery/crop tool on port 2223 over
  the same `web-generated/` tree.

## Running

```bash
./run_server.sh          # interactive menu of 12 model configs (see SERVER_OPTIONS.md)
./run_server.sh 9        # launch config 9 (FLUX.2-klein, the default) directly
./kill_flux.sh           # stop the supervisor and server
python flux_cli.py [--flux2|--gguf q8|--schnell|--kontext|--full-model] [--image path]
```

`run_server.sh` is also a supervisor: logs to `server.log`, auto-restarts on
crash/OOM (max 5 retries). SERVER_OPTIONS.md documents each menu number and
must stay in sync with the `case` statement in `run_server.sh`.

## Testing

```bash
python smoke_test_servers.py        # all 12 configs, isolated subprocesses
python smoke_test_servers.py 9 10   # subset
python test_rest_api.py             # /api/v1 contract checks, no GPU needed
```

`smoke_test_servers.py` writes a live-updating HTML tracker to
`server_smoke_test/index.html` and requires GPU + model downloads.
`test_rest_api.py` runs against Flask's test client with no model loaded, so it
covers routing/auth/status codes/error envelopes in seconds but never reaches
the GPU — it is the one test to run after touching either API dialect.

## Dependencies

Uses `uv pip` against `.venv` with `requirements.txt`. Key deps: torch (CUDA),
diffusers, transformers, flask, python-dotenv, huggingface_hub, requests.

## Architecture notes

- **Model variants** (all selected via flags on `load_model`): FLUX.1
  4-bit/full/GGUF/schnell, FLUX.1-Kontext editor (4-bit or full bf16), FLUX.2
  4-bit/full (32B), FLUX.2-klein (9B). Turbo LoRA (FLUX.2-dev only) and
  uncensored LoRA (FLUX.1 only) load on top.
- **VLM endpoint**: critique/describe/boost talk to the vision model through
  `edit_loop.py`, which speaks two dialects. `OLLAMA_URL` (the endpoint, name
  kept for compatibility) may point at a local ollama daemon — `/api/chat`,
  base64 `images`, `format` schema — or at an OpenAI-compatible server such as
  vLLM — `/v1/chat/completions`, `image_url` data-URL parts, `response_format`
  guided decoding. `vlm_dialect()` probes `/api/version` once per URL and
  caches the answer (`VLM_API=ollama|openai` forces it); `_openai_body()` does
  the translation, and the reply is reshaped back into ollama's envelope so
  every caller stays dialect-agnostic. `CRITIQUE_MODEL`/`DESCRIBE_MODEL` name
  the model, and default to `auto` — `resolve_vlm_model()` asks the endpoint
  what it serves (`/v1/models`, or ollama's `/api/ps` then `/api/tags`) and
  takes the first, preferring a resident one, so restarting the serving host
  on a different checkpoint doesn't need a config edit. The answer is cached
  per URL, refreshed by the `/status` probe (so a swap is noticed within one
  poll and the UI badge names the real model), and dropped by
  `forget_vlm_model()` on any HTTP error, which makes `_chat_text`'s existing
  retry re-discover in place. An explicit name still pins — do that when the
  endpoint serves several models and only one has vision. Serving the VLM
  off-box is the point of this: it takes the vision model's VRAM off the
  generating card entirely, which makes the note below moot for that setup
  (the status badge reads
  `/v1/models` instead of `/api/ps`, and startup warm-up is skipped — a remote
  server owns its own residency).
- **VLM VRAM**: a *local* ollama vision model (critique/describe/boost) competes
  with the diffusion pipeline for the same card — a resident qwen3.6 is ~5GB.
  `edit_loop.KEEP_ALIVE` (env `VLM_KEEP_ALIVE`, default `"0"`) is sent as
  ollama's `keep_alive` on every call, so the model unloads as soon as a call
  returns and the startup warm-up is skipped. Set `VLM_KEEP_ALIVE=15m` to keep
  it warm when the edit loop, not generation, is the main workload.
- **Text encoding**: FLUX.1 4-bit can use HF's remote text-encoder API (with
  bounded LRU embedding cache and automatic fallback to local encoders when the
  endpoint is down); everything else uses local encoders. FLUX.2 has no remote
  API (Mistral3/Qwen3 encoders).
- **Stability**: the VAE always runs in fp32 (`_stabilize_vae_fp32`) to prevent
  bf16 NaN → black-image decodes; generation retries once on a degenerate
  (all-black) result. That fp32 upcast doubles the activations of the
  full-resolution final decode, which is the peak-memory moment of a
  generation — `--vae-tiling` (`_apply_vae_tiling`) decodes in overlapping
  tiles to bound it, opt-in because tiling can leave faint seams on smooth
  gradients. Symptom it addresses: above ~1MP the job stalls on its last step
  with no error, the driver having silently paged the decode to system RAM.
- **Reference images**: `generate_image` accepts one PIL image or a list of up
  to `MAX_REFERENCE_IMAGES` (3). FLUX.2 pipelines take the list natively;
  Kontext stitches multiple refs side-by-side (`_stitch_references`) since its
  diffusers pipeline conditions on a single image; FLUX.1 img2img rejects >1.
  The API field is `input_images` (list); legacy single `input_image` and
  `input_paths` (server-side file paths, resolved against `web-generated/`
  when relative) are normalized into it at validation. `/fetch-image-path`
  serves the UI's "server path → data URL" import (mirrors `/fetch-image-url`).
- **Web queue**: one worker thread, `QUEUE_MAX_SIZE=10`, jobs carry progress
  state polled by the UI via `/status`. `/generate` validates all params at the
  API boundary and returns 400s.
- **Prompt expansion**: `{a|b}` in a prompt queues one job per alternative,
  cartesian across groups (`expand_prompt` in `web_server.py`, applied inside
  `_api_enqueue_generation` so both API dialects and every client get it).
  Nesting and `\{` escaping are supported; a braced run without a top-level
  `|` stays literal. All-or-nothing: past `QUEUE_MAX_SIZE` it's a 400, and a
  product that won't fit beside the current queue is a 429. The group's jobs
  carry an `expansion` dict; `_expansion_record` (called for every member, in
  the worker and in the queued-job cancel path) tiles them into a
  `_expansion_grid.png` contact sheet when the last one lands. A seedless
  group draws one seed at enqueue and shares it across members, so the
  alternatives are comparable (`expansion_same_seed: false` opts out).
- **Multi-model runs**: `/multi-run` generates one prompt (same seed) on a
  subset of the server configs sequentially. Each model switch is a supervised
  restart (the `/switch-model` exit-86 flow), so run state lives in
  `.multi_run.json`, not memory: `_multi_run_advance()` — called on
  model-ready and after every finished job — queues the current config's job,
  restarts into the next config once the queue is idle, or marks the run
  finished. Text-to-image only. Both endings go through `_multi_run_finish()`,
  which tiles the run's images into a `_multi_run_grid.png` comparison sheet
  (each cell captioned with the model that made it) and records its filename as
  `composite` on the run state, so a canceled run still gets a sheet of the
  models that did finish.
- **Performance**: batch jobs pre-encode the prompt once
  (`encode_prompt_once` → `prompt_embeds_kwargs`) instead of re-running the
  LLM-sized FLUX.2 encoders per image; LoRAs are fused after loading
  (`_fuse_loaded_lora`); live previews decode spatially-downsampled latents
  (`_shrink_latents_for_preview`), not full resolution; `web_server.py
  --compile` torch.compiles the transformer (lazy — first generation per
  resolution is slow).
- **Status note**: a daemon thread (`_note_reporter`) posts a short status line
  — config, model state, image/job counters, queue depth, last error — to
  `NOTE_URL` (default `http://127.0.0.1:9999/note`) every `NOTE_INTERVAL`
  seconds, default 300. The board holds *one* note and a POST replaces it, so
  the post is a full status line, not a log entry; its text is capped at 500
  characters server-side (a longer post is rejected outright and the stale note
  survives), so `_note_text` truncates. **Never put prompt text in it** — the
  board is shared and visible. Counters live in `_stats`, incremented in the
  queue worker's `finally`, and reset on restart, which means they are
  per-config totals (a model switch is a restart). Post failures are non-fatal
  and logged once. `NOTE_URL=off` disables it.
- **Output convention**: `flux{1|2}_{YYYYMMDD_HHMMSS}_{8hex}.png` plus a
  `.prompt` sidecar with the generation metadata, in `web-generated/` (server)
  or the CWD (CLI). `image_manager.py` parses the `# Prompt:` sidecar line —
  keep the format stable.

## Requirements

- CUDA GPU; Hugging Face token (`huggingface-cli login` or `HF_TOKEN`)
- `FLUX_API_KEY` env var (or `.env`) for the web server
- FLUX.2-dev (32B) needs far more VRAM than FLUX.1 (12B) or klein (9B)
