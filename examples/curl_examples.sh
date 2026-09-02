#!/usr/bin/env bash
# Every major FLUX REST API call as a copy-pasteable curl command.
#
#   export FLUX_API_KEY=your_key
#   export FLUX_URL=http://localhost:2222
#
# This file is a reference, not a script to run start to finish — some commands
# restart the server or delete files. Copy the one you need.
#
# Errors always come back as {"error": {"code": "...", "message": "..."}}.
# Branch on `code`; the message text is free to change.

: "${FLUX_URL:=http://localhost:2222}"
API="$FLUX_URL/api/v1"
AUTH=(-H "X-API-Key: $FLUX_API_KEY")
JSON=(-H 'Content-Type: application/json')


# --- discovery ---------------------------------------------------------------

# What this API offers (public — no key needed).
curl -s "$API" | jq

# Readiness. 200 when the model is loaded, 503 while it is still loading.
# Public, so a client can check the server is alive before it has a key.
curl -s "$API/health" | jq

# Machine-readable spec, and a browsable endpoint list.
curl -s "$API/openapi.json" | jq '.paths | keys'
open "$API/docs"   # or just visit it


# --- model and configs -------------------------------------------------------

# Capabilities of the loaded backend — check these before sending a mask or a
# negative prompt, which only some backends support.
curl -s "${AUTH[@]}" "$API/model" | jq

# The launcher's config menu; `switchable` is false without the supervisor.
curl -s "${AUTH[@]}" "$API/models" | jq

# Switch config. 202 means the restart was scheduled, not that the model is up:
# poll /health until ready afterwards.
curl -s -X PUT "${AUTH[@]}" "${JSON[@]}" "$API/models/current" \
  -d '{"config": 9}' | jq

# GPU draw and vision-model residency.
curl -s "${AUTH[@]}" "$API/telemetry" | jq


# --- generation --------------------------------------------------------------

# Queue a job. Returns 201 immediately with an id — generation is always
# asynchronous, even when the queue is empty.
curl -s -X POST "${AUTH[@]}" "${JSON[@]}" "$API/jobs" \
  -d '{"prompt": "a red fox in falling snow, golden hour", "steps": 30}' | jq

# Full parameter set.
curl -s -X POST "${AUTH[@]}" "${JSON[@]}" "$API/jobs" -d '{
  "prompt": "a lighthouse in a storm",
  "steps": 30,
  "batch": 4,
  "seed": 42,
  "guidance": 3.5,
  "size": "1.5mp",
  "orientation": "widescreen",
  "show_preview": true
}' | jq

# A {a|b} prompt queues one job per alternative — this is four jobs, and the
# response lists them all in `expanded`. The group shares one seed so the
# prompt is the only variable; "expansion_same_seed": false opts out.
curl -s -X POST "${AUTH[@]}" "${JSON[@]}" "$API/jobs" -d '{
  "prompt": "a {red|blue} car in {rain|snow}",
  "expansion_same_seed": true
}' | jq '.expanded[] | {id, prompt}'

# Poll one job. `state` settles at done | failed | canceled.
JOB=abc123def456
curl -s "${AUTH[@]}" "$API/jobs/$JOB" | jq '{state, current, batch, step, total_steps}'

# Wait for it from the shell.
until [ "$(curl -s "${AUTH[@]}" "$API/jobs/$JOB" | jq -r .state)" != "running" ]; do
  sleep 2
done

# Filenames and seeds of the results.
curl -s "${AUTH[@]}" "$API/jobs/$JOB" | jq '.images[] | {filename, seed}'

# The whole queue.
curl -s "${AUTH[@]}" "$API/jobs" | jq '{running: .running.id, queued: (.queued|length)}'

# Cancel a queued job, or interrupt the running one (finished batch images are
# kept).
curl -s -X DELETE "${AUTH[@]}" "$API/jobs/$JOB" | jq

# Forget the finished-jobs list (204, no body).
curl -s -X DELETE "${AUTH[@]}" "$API/jobs/recent" -o /dev/null -w '%{http_code}\n'

# Saved per-step preview frames (jobs run with save_previews).
curl -s "${AUTH[@]}" "$API/jobs/$JOB/frames" | jq


# --- reference images --------------------------------------------------------

# Generating from a reference means base64 in the JSON body. Build it inline:
IMG="data:image/jpeg;base64,$(base64 -w0 photo.jpg)"
jq -n --arg img "$IMG" '{
  prompt: "make it winter",
  input_images: [$img],
  strength: 0.55,
  aspect_mode: "keep"
}' | curl -s -X POST "${AUTH[@]}" "${JSON[@]}" "$API/jobs" -d @- | jq

# Or skip the upload entirely when the file is already on the server.
curl -s -X POST "${AUTH[@]}" "${JSON[@]}" "$API/jobs" \
  -d '{"prompt": "make it winter", "input_paths": ["archive/photo.jpg"]}' | jq

# Inpainting: exactly one reference plus a mask (FLUX.2 or SDXL only).
jq -n --arg img "$IMG" --arg mask "data:image/png;base64,$(base64 -w0 mask.png)" '{
  prompt: "a brass telescope",
  input_images: [$img],
  mask_image: $mask
}' | curl -s -X POST "${AUTH[@]}" "${JSON[@]}" "$API/jobs" -d @- | jq

# Convert a camera RAW file the browser cannot decode.
curl -s -X POST "${AUTH[@]}" "$API/imports/raw" -F 'file=@DSC_0001.NEF' \
  | jq '{width, height}'

# Fetch a remote image server-side (no CORS limits).
curl -s -X POST "${AUTH[@]}" "${JSON[@]}" "$API/imports/url" \
  -d '{"url": "https://example.com/photo.jpg"}' | jq '{width, height}'

# Load one from the server's own filesystem.
curl -s -X POST "${AUTH[@]}" "${JSON[@]}" "$API/imports/path" \
  -d '{"path": "archive/photo.png"}' | jq '{width, height}'

# Browse server-side folders to find those paths.
curl -s "${AUTH[@]}" "$API/files?dir=archive" | jq '{dir, dirs, files: (.files|length)}'
curl -s "${AUTH[@]}" "$API/files/thumbnail?path=archive/photo.png" -o thumb.jpg


# --- images ------------------------------------------------------------------

# Today's output, newest first.
curl -s "${AUTH[@]}" "$API/images" | jq '.images[] | {time, filename}'

# Download one. Unlike the legacy /images/ route this needs the key; use the
# query param when the consumer is an <img> tag that cannot set headers.
curl -s "${AUTH[@]}" "$API/images/flux2_20260808_143022_a1b2c3d4.png" -o out.png
echo "$API/images/out.png?api_key=$FLUX_API_KEY"

# Preserve one from the day's housekeeping.
curl -s -X POST "${AUTH[@]}" "$API/images/flux2_20260808_143022_a1b2c3d4.png/save" | jq

# Move today's output to archive/.
curl -s -X POST "${AUTH[@]}" "$API/archive" | jq

# Delete. Both are permanent.
curl -s -X DELETE "${AUTH[@]}" "$API/images/flux2_20260808_143022_a1b2c3d4.png" | jq
curl -s -X DELETE "${AUTH[@]}" "$API/images" | jq


# --- vision-model jobs -------------------------------------------------------

# One resource for all three tasks, discriminated by "task". These can run for
# minutes, which is why they are jobs: POST returns 202 with an id to poll.

# Photo -> prompt.
VLM=$(jq -n --arg img "$IMG" '{task: "describe", images: [$img]}' \
  | curl -s -X POST "${AUTH[@]}" "${JSON[@]}" "$API/vlm/jobs" -d @- | jq -r .id)
until curl -s "${AUTH[@]}" "$API/vlm/jobs/$VLM" | jq -e .done >/dev/null; do sleep 2; done
curl -s "${AUTH[@]}" "$API/vlm/jobs/$VLM" | jq -r .result.prompt

# Draft prompt -> stronger prompt in the loaded model's idiom (level 1-5).
curl -s -X POST "${AUTH[@]}" "${JSON[@]}" "$API/vlm/jobs" \
  -d '{"task": "boost", "prompt": "a castle", "level": 4}' | jq

# Critique an edit against its reference.
jq -n --arg img "$IMG" '{
  task: "critique",
  direction: "make the sky more dramatic",
  ref_image: $img,
  output_filename: "flux2_20260808_143022_a1b2c3d4.png"
}' | curl -s -X POST "${AUTH[@]}" "${JSON[@]}" "$API/vlm/jobs" -d @- | jq

# A failed vision job polls as 502 with code "vlm_failed" — usually ollama not
# running, or the model not pulled.


# --- multi-model runs --------------------------------------------------------

# One prompt on several configs in turn, sharing a seed so results compare.
# The server restarts between models, so the run is file-backed and survives.
curl -s -X POST "${AUTH[@]}" "${JSON[@]}" "$API/multi-runs" \
  -d '{"configs": [9, 6, 1], "prompt": "a red fox in snow", "steps": 28}' | jq

# Follow it. Expect connection failures mid-restart; that is normal.
curl -s "${AUTH[@]}" "$API/multi-runs/current" \
  | jq '{active, done: (.run.results|length), total: (.run.configs|length)}'

# Cancel or dismiss (204 either way).
curl -s -X DELETE "${AUTH[@]}" "$API/multi-runs/current" -o /dev/null -w '%{http_code}\n'


# --- edit-loop film strip ----------------------------------------------------

curl -s -X POST "${AUTH[@]}" "${JSON[@]}" "$API/filmstrips" -d '{
  "filenames": ["flux2_20260808_143022_a1b2c3d4.png",
                "flux2_20260808_143512_e5f6a7b8.png"],
  "direction": "make the sky more dramatic",
  "prompts": ["iteration 1 prompt", "iteration 2 prompt"]
}' | jq
