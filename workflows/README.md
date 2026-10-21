# Workflows for `video_server.py`

`video_server.py` does not contain an H3 node graph. It submits **ComfyUI's
own** workflow with a handful of values overwritten, which takes two files per
workflow:

| File | What it is |
|---|---|
| `<name>.json` | ComfyUI's workflow in **API format** (`Workflow → Export (API)`) |
| `<name>.map.json` | which node input each request field writes to |

The default workflow name is `h3_t2va`, so the server looks for
`workflows/h3_t2va.json` and `workflows/h3_t2va.map.json`. Post
`{"workflow": "other_name"}` to `/generate` to use another pair.

## Why it works this way

H3's graph is a moving target. The base T2V/I2V/R2V templates need ComfyUI
0.30, Multiframe Reference needs 0.34, Fun ControlNet and sparse attention need
0.35, FastH3 needs 0.36, and the `pruned_w6a8` build needs 0.38 — and the node
set changes along the way. A graph pinned in Python would need editing on every
one of those steps, and the failure would be a confusing `KeyError` rather than
"your map is stale". Exporting the template you actually use and saying where
the five knobs live is both less code and less to keep in sync.

It also means every ComfyUI feature stays reachable: turbo LoRAs, Sage
Attention patching, the latent upscaler, I2V continuation. Build the graph you
want in ComfyUI, export it, point the map at its prompt/size/length/steps/seed
inputs, and this server drives it.

## What is already here

| Workflow | From template | Nodes | Takes |
|---|---|---|---|
| `h3_t2va` | `video_minimax_h3_t2v` | 20 | prompt only |
| `h3_fl2va` | `video_minimax_h3_i2v` | 21 | prompt + a first frame |

Both are generated and working. `h3_t2va` is the default; post
`{"workflow": "h3_fl2va"}` with a `start_image` file to animate a still. The
extra node in `h3_fl2va` is the `LoadImage` that `start_image` writes into —
`video_server.py` uploads the image to ComfyUI's `/upload/image` and puts the
returned name in that node.

### Regenerating them

```bash
python workflow_from_template.py --list                              # the H3 templates
python workflow_from_template.py video_minimax_h3_t2v --out h3_t2va
python workflow_from_template.py video_minimax_h3_i2v --out h3_fl2va
```

That writes both the graph and the map. It does the same job as the browser's
`Export (API)` — including flattening the subgraph — and then derives the map
from the graph, so node ids that moved in a ComfyUI upgrade are picked up
rather than hand-corrected. It needs ComfyUI running, because `/object_info` is
the source of truth for widget names. `--dry-run` prints without writing;
`--size`, `--frames`, `--turbo`, `--turbo-steps`, `--unet` and `--lora` set the
baked-in defaults.

Three things about the generated graphs are worth knowing before editing them:

- **It is the template's graph, with the checkpoint swapped.** The template
  defaults to `minimax_h3_fl2va_pruned_int8_convrot` (19.5 GiB), which does not
  fit beside the 14.6 GiB NVFP4 text encoder on a 32GB card.
  `minimax_h3_fl2va_pruned_w6a8` (14.9 GiB) does, and needing it is why the
  checkout is ComfyUI 0.38+.
- **Turbo is on, so `steps` maps to the turbo branch of a switch.** The
  template routes the step count through a `ComfySwitchNode` pair driven by a
  boolean: with it `true`, one switch selects the LoRA-patched model and the
  other selects the turbo step count; the non-turbo `20` is wired but unused.
  `workflow_from_template.py` follows that switch rather than guessing, which
  is why the map lands on `138.value` for `h3_t2va` and `125.value` for
  `h3_fl2va`, and why `--turbo off` re-points it at the other branch by
  itself. The `_steps_note` key in each map records which branch was taken
  (keys starting with `_` are notes, not request fields).
- **The seconds→frames chain is gone.** The template computed `length` from a
  `PrimitiveFloat` of seconds through a `ComfyMathExpression` running
  `max(5, round(a*24)) + (5 - (max(5, round(a*24)) % 17)) % 17`. That is the
  same 17k+5 snap `video_server.py` does in `_snap_frames`, so `length` is a
  literal and `frames` maps straight to it. (It also removed the one node whose
  API spelling was unclear — its `values` input is a `COMFY_AUTOGROW_V3` group
  whose UI slot name is the dotted `values.a`.)

**H3 has no negative prompt.** `MiniMaxH3ImageToVideo` takes `prompt`, `width`,
`height`, `length`, `clip`, `vae`, `first_frame` and `last_frame` — nothing
else. The map therefore has no `negative` entry, and supplying one is a 400
rather than a silent drop, because a clip that ignored your negative prompt
with nothing saying so is the worse outcome.

## Setting one up

1. In ComfyUI, open the H3 template (`Workflow → Browse Templates → Video →
   MiniMax H3`), pick the checkpoint build you downloaded, and **run it once**.
   A graph that does not work in ComfyUI will not work through here, and
   debugging it in the ComfyUI UI is far easier.
2. `Workflow → Export (API)`. Save as `workflows/h3_t2va.json` in this repo.
   Not plain `Export` — that writes the UI format, with `nodes` and `links`
   arrays, which the server rejects by name because the two look similar enough
   to waste an afternoon.

   Note that ComfyUI's bundled H3 templates wrap the whole graph in a
   **subgraph** (the node's `type` is a UUID and the real 21 nodes live under
   `definitions.subgraphs`). `Export (API)` flattens that; copying the template
   JSON straight out of
   `comfy-venv/Lib/site-packages/comfyui_workflow_templates_json/templates/`
   does not, and `video_server.py` will reject it as UI format.
3. List the graph:

   ```bash
   python video_server.py --probe h3_t2va
   ```

   which prints every node, its class, its title and its literal inputs — the
   same data as `GET /workflow/nodes?name=h3_t2va`.
4. Write `workflows/h3_t2va.map.json` from that listing. Values are
   `"<node_id>.<input_name>"`:

   ```json
   {
     "prompt":   "6.text",
     "negative": "7.text",
     "width":    "40.width",
     "height":   "40.height",
     "frames":   "40.length",
     "steps":    "17.steps",
     "seed":     "16.noise_seed",
     "start_image": "52.image"
   }
   ```

   **Those node ids are an example, not a recipe.** Read yours off the probe
   output — they differ between template versions and between the `fl2va` and
   `ref2va` graphs.

`prompt`, `width`, `height`, `frames`, `steps` and `seed` are required; a
missing one is a 400 that names it. `negative` and `start_image` are optional —
leave `start_image` out for a text-to-video-only graph and the server will say
so if a first frame is uploaded anyway.

The server refuses to write over an input that is wired to another node (those
appear as `linked_inputs` in the probe), because overwriting a link silently
detaches the graph and the result is a clip that ignored half the settings.

## Suggested pairs

- `h3_t2va` — the T2V/I2V template on the `fl2va` checkpoint. One graph covers
  both: with `start_image` mapped, supplying a first frame makes it
  image-to-video, and omitting one makes it text-to-video.
- `h3_ref2va` — the R2V template on the `ref2va` checkpoint, for locking a
  character, style, camera move or voice to references. Needs its own export
  because it is a different checkpoint and a different graph.

## Gotchas worth knowing before you debug

- A `Reference to Video` node dropped onto a T2V/I2V canvas is **ignored
  without an error**. Reference conditioning only applies in a graph built from
  the R2V template.
- The graph needs a saving node (`SaveVideo`, or a VHS combine node). Without
  one ComfyUI completes the job and produces no file, and the server reports
  exactly that rather than hanging.
- Audio needs 12+ steps. At the turbo 8 it is usually the weakest part of the
  output, so the 4- and 8-step LoRAs trade audio quality for time.
- Keep both dimensions on a multiple of 32 and inside 768×1344. The server
  snaps and bounds them anyway, but a graph with its own hardcoded latent size
  will quietly disagree with what the request asked for if `width`/`height` are
  mapped to the wrong node.
