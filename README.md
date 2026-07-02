# FLUX Image Generator

Self-hosted image generation on FLUX.1 / FLUX.2 diffusion models (via
`diffusers`), tuned for DGX Spark / unified-memory NVIDIA systems. One codebase,
three entry points:

| Entry point | What it is | Port |
|---|---|---|
| `web_server.py` | Web UI + JSON API with a generation queue, live latent previews, img2img, multi-reference editing (up to 3 images), inpainting (FLUX.2), spectrum grids, history | 2222 |
| `flux_cli.py` | Interactive CLI (REPL) on top of `flux_core.py`, the shared model-loading/generation core the other apps import | — |
| `image_manager.py` | Browser tool to browse/crop/organize everything under `web-generated/` | 2223 |

## Quick start

```bash
# One-time setup
python -m venv .venv && source .venv/bin/activate
uv pip install -r requirements.txt
huggingface-cli login          # or export HF_TOKEN=...
cp .env.example .env           # set FLUX_API_KEY=<your secret> in .env

# Launch the web server (interactive menu of 12 model configs)
./run_server.sh

# Or launch a specific config directly, e.g. FLUX.2-klein (the default):
./run_server.sh 9
```

Then open `http://<host>:2222`, enter your `FLUX_API_KEY` in the UI, and
generate. Images and their `.prompt` sidecar metadata land in `web-generated/`.

The 12 server configurations (FLUX.1 4-bit/full/GGUF/schnell/uncensored,
FLUX.2 4-bit/full±turbo/klein, and the Kontext instruction editor) are
documented in **[SERVER_OPTIONS.md](SERVER_OPTIONS.md)**. `run_server.sh` is a
supervisor: it logs to `server.log` and auto-restarts the server on crashes
(including OOM kills), up to 5 times. Stop everything with `./kill_flux.sh`.

## CLI

```bash
python flux_cli.py                # FLUX.1 4-bit, remote text encoder
python flux_cli.py --flux2        # FLUX.2
python flux_cli.py --gguf q8      # GGUF Q8 (DGX Spark recommended)
python flux_cli.py --kontext --image photo.png   # instruction editing
```

The REPL supports `/help`, `same`, `reseed <n>`, `/steps`, `/guidance`,
`/strength`, `/image <path> [path2] [path3]`, aspect presets (`/square`,
`/portrait`, `/landscape`, `/widescreen`, `/extra-tall`) and sizes (`/0.75`,
`/1k`, `/2k`, `/4k`), inline in prompts too: `a cat /4k /portrait`.

## Reference images (up to 3)

Both the web UI and the CLI accept up to 3 reference images per generation
(`input_images` in the JSON API). The first image is the primary — it sets the
output aspect ratio. How they're used depends on the model:

- **FLUX.2 (incl. klein)** conditions on all references natively (each is
  encoded separately and attended to), e.g. "put the object from the first
  image into the scene from the second".
- **Kontext** conditions on a single image, so multiple references are
  stitched side-by-side into one canvas — write the instruction positionally:
  "give the person in the left image the jacket from the right image".
- **FLUX.1 img2img** takes a single reference; multiple are rejected with a 400.

Strength only applies to single-image FLUX.1 img2img; inpainting requires
exactly one reference.

## Web frontend

The UI is plain static files in `static/` (`index.html`, `app.css`, `app.js`)
served by Flask — no build step. The API is authenticated with the
`X-API-Key` header (or `api_key` query param).

## Testing

```bash
python smoke_test_servers.py          # sweep all 12 configs
python smoke_test_servers.py 9 10     # just klein + kontext
```

Each config runs in an isolated subprocess (a crash/OOM fails just that
config) and generates one sample image. While the sweep runs,
`server_smoke_test/index.html` is a live tracking page (auto-refreshes,
shows pending/running/pass/fail with phase and elapsed time); when it
finishes it becomes the final report. Runs merge, so re-testing one option
updates only its card.

## Requirements

- CUDA-capable GPU (FLUX.2-dev is 32B parameters — the full model wants
  ~64GB+; klein-9B and the 4-bit/GGUF variants are much lighter)
- Hugging Face token with access to the FLUX repos
- `FLUX_API_KEY` set (env var or `.env`) for the web server

## Layout

```
flux_core.py          model loading + generation core (imported by everything)
flux_cli.py           interactive CLI REPL
web_server.py         Flask API + queue worker
static/               web UI (index.html, app.css, app.js)
image_manager.py      gallery/crop/organize tool for web-generated/
run_server.sh         launcher menu + crash-restart supervisor
kill_flux.sh          stop supervisor + server
smoke_test_servers.py per-config smoke test with live HTML report
SERVER_OPTIONS.md     the 12 server configs, flags, and VRAM notes
web-generated/        generated images + .prompt sidecars (gitignored)
```
