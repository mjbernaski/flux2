# REST API (`/api/v1`)

A resource-oriented HTTP API covering everything the port-2222 server can do.
Sample code for every call is in [`examples/`](examples/); a running server
serves a browsable version of this reference at `GET /api/v1/docs` and the
OpenAPI 3.1 document at `GET /api/v1/openapi.json`.

This sits **alongside** the original flat routes (`/generate`, `/status`,
`/delete`, ...), which `static/app.js` still uses and which are unchanged. Both
dialects call the same `_api_*` core functions in `web_server.py`, so there is
one implementation of each behavior; only the routing and the response envelope
differ. New clients should use `/api/v1`.

## Conventions

**Authentication.** Every endpoint except `/api/v1`, `/health`, `/openapi.json`
and `/docs` requires the server's API key, either as an `X-API-Key` header or
an `api_key` query parameter. The query form exists for consumers that cannot
set headers, notably `<img src>`.

**Errors.** Always the same shape, with the HTTP status carrying the category:

```json
{"error": {"code": "queue_full", "message": "Queue is full (10 max). Cancel a queued job or wait."}}
```

Branch on `code` — it is stable. Messages are written for humans and may change.

| code | status | meaning |
| --- | --- | --- |
| `unauthorized` | 401 | Missing or wrong API key |
| `invalid_request` | 400 | A parameter failed validation |
| `not_found` | 404 | No such job, image, run, or path |
| `method_not_allowed` | 405 | Wrong verb for the resource |
| `busy` | 409 | Jobs in flight block this operation |
| `queue_full` | 429 | 10 jobs already pending |
| `model_loading` | 503 | The model is not up yet — poll `/health` |
| `no_supervisor` | 400 | Started without `run_server`, so it cannot restart itself |
| `undecodable_image` | 400 | A reference image or mask could not be decoded |
| `fetch_failed` | 400 | A remote URL could not be retrieved |
| `too_large` | 400/413 | Body or fetched image over the 64MB cap |
| `vlm_failed` | 502 | The vision model failed (usually ollama not running) |
| `io_error` | 500 | A filesystem operation failed |
| `internal_error` | 500 | Unhandled server-side failure |

**Asynchrony.** Generation and vision-model calls are always jobs. `POST /jobs`
returns `201` with an id even when the queue is empty, because a single worker
thread runs everything in turn; a vision call can run for minutes, well past a
browser's per-request timeout. Poll the job resource in both cases.

## Endpoints

### Discovery

| | |
| --- | --- |
| `GET /api/v1` | Service index and resource map. Public. |
| `GET /api/v1/health` | Model-load readiness. Public. `200` when ready, `503` while loading. |
| `GET /api/v1/openapi.json` | OpenAPI 3.1 document. Public. |
| `GET /api/v1/docs` | Human-readable endpoint list. Public. |

`/health` returns `{ready, status, error, elapsed_s, server_version}`. `status`
narrates the load (`loading FLUX.2 model`, `loading turbo LoRA`, ...), which is
worth surfacing since a cold 32B load takes minutes.

### Model and configuration

| | |
| --- | --- |
| `GET /model` | Capabilities of the loaded backend |
| `GET /models` | The launcher's config menu (ids 1–14) and which is live |
| `GET /models/current` | The running config |
| `PUT /models/current` | Switch config — **restarts the server** |
| `GET /telemetry` | GPU power draw, vision-model residency |

Check `GET /model` before sending optional fields: `negative_prompt` needs the
SDXL backend, `mask_image` needs FLUX.2 or SDXL, and more than one reference
image needs Kontext or FLUX.2. Sending an unsupported field is a `400`.

`GET /model` also reports `vae_tiling`, which says whether the server was
started with `--vae-tiling`. That decodes the final image in overlapping tiles
instead of one allocation — worth turning on if generations above roughly 1 MP
stall on their last step, which is the fp32 VAE decode spilling out of VRAM.

`PUT /models/current` answers `202`, not `200`: the models are far too large to
hot-swap, so switching writes the target config to a file and exits with a code
the supervisor interprets as "relaunch me". The response means the restart was
scheduled. Poll `/health` — expect connection failures while the process is
down. Without the supervisor (`GET /models` reports `switchable: false`) this is
a `400 no_supervisor`.

### Generation

| | |
| --- | --- |
| `POST /jobs` | Queue a job → `201` + `Location` |
| `GET /queue` | The current queue, ordered, with positions and a wait estimate |
| `GET /queue.html` | Self-refreshing HTML view of the queue |
| `GET /jobs` | Running, queued, and recently finished jobs |
| `GET /jobs/{id}` | One job |
| `DELETE /jobs/{id}` | Cancel a queued job, or interrupt the running one |
| `GET /jobs/{id}/previews` | Intermediate images: the live frame plus saved per-step frames |
| `GET /jobs/{id}/previews.html` | Self-refreshing HTML view of those images |
| `DELETE /jobs/recent` | Clear the finished-jobs list → `204` |

Request body — only `prompt` is required:

| field | type | default | notes |
| --- | --- | --- | --- |
| `prompt` | string | — | Required |
| `steps` | int | 25 | 1–200 |
| `batch` | int | 1 | 1–128; the prompt is encoded once for the whole batch |
| `seed` | int\|null | null | Omit for a fresh random seed per image |
| `guidance` | float\|null | null | ≥ 0 |
| `strength` | float | 0.5 | 0–1; img2img only |
| `orientation` | string | `landscape` | `square`, `portrait`, `landscape`, `widescreen`, `extra-tall` |
| `size` | string | `1mp` | `0.25mp` … `2mp` |
| `negative_prompt` | string | — | SDXL only |
| `input_images` | string[] | `[]` | Base64 or data URLs, up to 3 |
| `input_paths` | string[] | — | Server-side paths, folded into `input_images` |
| `mask_image` | string | — | Inpainting; needs exactly one input image |
| `aspect_mode` | string | `keep` | `keep` derives output dims from the reference |
| `show_preview` | bool | false | Decode latent previews while generating |
| `save_previews` | bool | false | Also write each frame to `steps/` |
| `spectrum_grid` | bool | false | Sweep guidance/strength into a matrix |

`GET /queue` is the queue-centric view, for answering "where is my job in line
and when will it run":

```json
{
  "running": { "id": "a1b2c3d4e5f6", "current": 2, "batch": 4, "step": 18, "total_steps": 30 },
  "waiting": [ { "position": 1, "id": "9f8e7d6c5b4a", "prompt": "...", "batch": 2 } ],
  "depth": 1, "capacity": 10, "accepting": true, "busy": true,
  "images_pending": 4, "seconds_per_image": 6.4, "estimated_wait_s": 25.6
}
```

`accepting` is what `POST /jobs` will do: `false` means the next submit is
rejected with `queue_full`. The running job holds no pending slot, so a full
queue can still have one generating. `estimated_wait_s` extrapolates from
recently completed jobs and is `null` until at least one has finished.

A job's `state` goes `queued` → `running` → `done` | `failed` | `canceled`.
While running, `current`/`batch` track the image and `step`/`total_steps` the
diffusion progress; `total_steps` is re-read from the live scheduler on the
first step, because img2img, turbo and schnell denoise fewer steps than
requested. When done, `images[]` carries `{filename, seed, timings}` per output.

Cancelling a running job keeps whatever batch images already finished.

`GET /jobs/{id}/previews` returns the intermediate images from a generation.
Two distinct things live there:

```json
{
  "id": "a1b2c3d4e5f6", "state": "running",
  "live":  { "path": "_preview_current.png", "step": 12, "total_steps": 30,
             "image": 1, "ts": 1786225620454,
             "url": "/api/v1/images/_preview_current.png?t=1786225620454" },
  "frames": [ { "path": "steps/a1b2c3d4e5f6_img01_step005.png", "image": 1, "step": 5,
                "url": "/api/v1/images/steps/a1b2c3d4e5f6_img01_step005.png" } ],
  "count": 1, "saving": true
}
```

`live` is the latent being denoised right now, decoded at reduced resolution
and overwritten in place — it exists only while the job runs with
`show_preview`, and its `ts` changes on every frame, so use the supplied `url`
rather than caching the path. `frames` are written only when `save_previews` is
also set, but they persist after the job ends, so a finished run can be
replayed step by step.

### Watching in a browser

Two endpoints render the same data as a self-refreshing page, for when you want
to watch rather than integrate: `GET /queue.html` and
`GET /jobs/{id}/previews.html`. A plain browser navigation cannot send an
`X-API-Key` header, so open them with the key in the query string:

```
http://localhost:2222/api/v1/queue.html?api_key=YOUR_KEY
http://localhost:2222/api/v1/jobs/a1b2c3d4e5f6/previews.html?api_key=YOUR_KEY
```

The queue page polls every 2 seconds and shows the running job's live preview
and progress bar alongside the ordered wait list. The previews page shows the
frame being denoised at full width and every saved frame below it, grouped by
batch image, and stops polling once the job settles. Thumbnails load from the
unauthenticated legacy `/images/` route, so no key ends up in an image URL.

### Images

| | |
| --- | --- |
| `GET /images` | Today's images, newest first |
| `GET /images/{filename}` | Image bytes |
| `DELETE /images/{filename}` | Delete one of today's images and its sidecar |
| `DELETE /images` | Delete **all** of today's output |
| `POST /images/{filename}/save` | Copy into `.saved/`, beyond archive and delete |
| `POST /archive` | Move today's output into `archive/` |
| `POST /filmstrips` | Compose an edit-loop film strip |

`GET /images/{filename}` requires the key, unlike the legacy `/images/` route
which is deliberately open so the UI's `<img>` tags work. Use
`?api_key=...` for that case here.

Both delete endpoints are permanent. `POST /archive` is the non-destructive
way to clear the day.

### Reference images

| | |
| --- | --- |
| `POST /imports/raw` | Camera RAW → JPEG data URL (`multipart/form-data`, field `file`) |
| `POST /imports/url` | Fetch a remote image server-side |
| `POST /imports/path` | Load from the server's filesystem |
| `GET /files` | Browse a server-side directory |
| `GET /files/thumbnail` | JPEG thumbnail of a server-side image |

All three import routes return `{image, width, height}` where `image` is a
JPEG data URL bounded to 2048px — pass it straight into `input_images`. RAW
support needs `rawpy` on the server (`501` if absent).

For files already on the server, `input_paths` on `POST /jobs` skips the
round trip entirely.

> These endpoints read arbitrary paths and fetch arbitrary URLs by design —
> the server owner reaching their own disk and network. The API key is the only
> gate, so treat it as a host credential and don't expose the server to
> untrusted networks.

### Vision-model jobs

| | |
| --- | --- |
| `POST /vlm/jobs` | Start a `describe`, `boost`, or `critique` job → `202` |
| `GET /vlm/jobs/{id}` | Poll it |

One resource for all three tasks, discriminated by `task` — the legacy API
exposes these as three endpoints with three different id fields.

- **describe** — `images[]`, `think` → `result.prompt`. Photo(s) in, a prompt
  that would recreate them out.
- **boost** — `prompt`, `level` 1–5, `has_image`, `negative_prompt`,
  `variant_index`/`variant_count` → `result.prompt`, `result.negative_prompt`.
  Rewrites a draft into the loaded model's idiom; `level` sets how far it may
  depart from the original.
- **critique** — `direction`, `ref_image`, `output_filename`, `history` →
  `result.critique`, `result.revised_prompt`, `result.score`, `result.metrics`.
  Compares an edit's output against its reference.

While running: `{"id", "done": false, "result": null}`. On success, `done: true`
with `result`. If the model itself failed, the status is `502 vlm_failed` —
the request was fine, the vision model was not.

### Multi-model runs

| | |
| --- | --- |
| `POST /multi-runs` | Run one prompt across several configs → `201` |
| `GET /multi-runs/current` | The active or last-finished run |
| `DELETE /multi-runs/current` | Cancel or dismiss → `204` |

Body: `configs` (ids from `GET /models`), `prompt`, and optionally
`orientation`, `size`, `steps`, `seed`, `guidance`, `batch`, `show_preview`.
Text-to-image only — references, masks and negative prompts are per-model
capabilities. Unless you supply a seed, one is drawn and shared by every model
so the outputs are comparable.

Each config change is a supervised restart, so run state lives in a file, not
memory, and survives them. At most one run is active at a time; expect
connection failures while polling across a restart.

## Quick start

```bash
curl -s localhost:2222/api/v1/health | jq

JOB=$(curl -s -X POST -H "X-API-Key: $FLUX_API_KEY" -H 'Content-Type: application/json' \
  localhost:2222/api/v1/jobs -d '{"prompt":"a red fox in snow","steps":30}' | jq -r .id)

until [ "$(curl -s -H "X-API-Key: $FLUX_API_KEY" \
  localhost:2222/api/v1/jobs/$JOB | jq -r .state)" != "running" ]; do sleep 2; done

curl -s -H "X-API-Key: $FLUX_API_KEY" localhost:2222/api/v1/jobs/$JOB | jq '.images[].filename'
```

Or in Python:

```python
from examples.flux_client import FluxClient

flux = FluxClient()
flux.wait_until_ready()
job = flux.generate("a red fox in falling snow", steps=30)
print(flux.download(job["images"][0]["filename"]))
```

## Testing

```bash
python test_rest_api.py
```

61 contract checks — routing, auth, status codes, the error envelope, and that
the legacy routes still answer in their own dialect. Runs against Flask's test
client with no model loaded, so it needs no GPU and takes seconds. It does not
generate anything; for that, run `examples/flux_client.py` against a live
server.
