# API examples

Sample code for every major function of the FLUX server's REST API
(`/api/v1`). See [../REST_API.md](../REST_API.md) for the endpoint reference,
or `GET /api/v1/docs` on a running server for a browsable version.

| File | What it covers |
| --- | --- |
| [`flux_client.py`](flux_client.py) | Full Python client + a runnable CLI for each capability |
| [`flux_client.js`](flux_client.js) | The same surface in JavaScript (browser and Node 18+) |
| [`curl_examples.sh`](curl_examples.sh) | Every call as a copy-pasteable curl command |

## Setup

```bash
export FLUX_API_KEY=your_key          # the key the server was started with
export FLUX_URL=http://localhost:2222 # optional; this is the default
```

The Python client needs only `requests`, which the server already depends on,
so the project venv works as-is:

```bash
.venv/bin/python examples/flux_client.py health
```

## The five-line version

```python
from flux_client import FluxClient

flux = FluxClient()
flux.wait_until_ready()
job = flux.generate("a red fox in falling snow", steps=30)
print(flux.download(job["images"][0]["filename"]))
```

## Try it

```bash
python examples/flux_client.py health                       # readiness + capabilities
python examples/flux_client.py generate "a red fox" --steps 30
python examples/flux_client.py batch "a lighthouse" --count 4
python examples/flux_client.py img2img photo.jpg "make it winter"
python examples/flux_client.py inpaint photo.jpg mask.png "a brass telescope"
python examples/flux_client.py progress "a mountain range"  # live latent previews
python examples/flux_client.py describe photo.jpg           # photo -> prompt
python examples/flux_client.py boost "a castle" --level 4   # prompt -> better prompt
python examples/flux_client.py history
python examples/flux_client.py models
python examples/flux_client.py multirun "a red fox" 9 6 1   # one prompt, three models
python examples/flux_client.py errors                       # how failures surface
python examples/flux_client.py demo                         # every read-only call
```

## Three things that trip people up

**Generation is always asynchronous.** `POST /jobs` returns 201 with a job id
even when the queue is empty, because one worker thread runs every job. Poll
`GET /jobs/{id}` until `state` is `done`, `failed`, or `canceled`.
`FluxClient.generate()` does that for you; `submit()` doesn't, if you'd rather
drive the loop.

**Capabilities depend on the loaded config.** Negative prompts need the SDXL
backend, inpainting needs FLUX.2 or SDXL, and more than one reference image
needs Kontext or FLUX.2. Check `GET /model` rather than assuming — sending an
unsupported field is a 400.

**Branch on the error code, not the message.** Failures are always
`{"error": {"code", "message"}}`. The codes (`model_loading`, `queue_full`,
`not_found`, `busy`, `invalid_request`, `vlm_failed`, ...) are stable; the
messages are written for humans and may change.
