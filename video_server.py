"""Video Server - MiniMax H3 text/image-to-video, fronting ComfyUI.

Run:
    python video_server.py --port 2224

This is the third of the Flask apps over `web-generated/` (web_server.py on
2222 generates images, image_manager.py on 2223 browses them), and the only one
that holds no model and touches no GPU. That is deliberate, and it is the whole
reason this is its own app rather than a `/video` endpoint inside
`web_server.py`: H3 peaks at ~27GB and the image generator sits at 21-26GB, so
the two cannot be resident on one 32GB card at once. Switching modes therefore
means stopping one of them - and a switch button inside a process that the
switch kills can only ever go one way. This app stays up in both modes, which
is what lets it be asked to go either way.

What it does not do is generate. ComfyUI (port 8188) owns the H3 graph, its
quantized checkpoints and its own queue; everything here is a thin translation
of a small JSON request into a patched ComfyUI workflow, plus the bookkeeping
that lands finished clips in `web-generated/video/` under this repo's filename
convention so the rest of the tooling can see them.

The workflow is not hardcoded. `workflows/h3_t2va.json` is an API-format export
of ComfyUI's own H3 template and `workflows/h3_t2va.map.json` says which node
input each request field writes to, because H3's node graph is a moving target
across ComfyUI releases (0.30 for the base templates, 0.35 for sparse
attention, 0.38 for the w6a8 build) and a graph pinned in Python would break on
the next one. `GET /workflow/nodes` dumps the loaded graph so the map can be
written by reading it rather than by guessing.

The mode lives in `.mode` (`flux` | `video`; absent means `flux`, so a box that
has never switched behaves exactly as it did before this file existed), and
`switch_mode.ps1` is its only writer. Nothing here edits it: POST /mode
launches that script detached and returns, because the switch outlives the
request by minutes.
"""
import argparse
import hmac
import json
import os
import random
import shutil
import socket
import subprocess
import sys
import threading
import time
from datetime import datetime

import requests
from dotenv import load_dotenv
from flask import Flask, Response, jsonify, request, send_from_directory

load_dotenv()

HERE = os.path.abspath(os.path.dirname(__file__))
ROOT = os.path.join(HERE, "web-generated")
# Finished clips go in a subfolder rather than loose in web-generated/: an mp4
# is not something the image tools can thumbnail, so keeping them together
# makes `web-generated/video` one path to include or exclude wholesale.
VIDEO_DIR = os.path.join(ROOT, "video")
WORKFLOW_DIR = os.path.join(HERE, "workflows")
os.makedirs(VIDEO_DIR, exist_ok=True)

API_KEY = (os.environ.get("FLUX_API_KEY") or "").strip()

PORT = int(os.environ.get("VIDEO_SERVER_PORT", 2224))
FLUX_PORT = int(os.environ.get("PORT", 2222))
COMFY_URL = (os.environ.get("COMFY_URL") or "http://127.0.0.1:8188").rstrip("/")
COMFY_PORT = int(os.environ.get("COMFY_PORT", 8188))

MODE_FILE = os.path.join(HERE, ".mode")
SWITCHING_FILE = os.path.join(HERE, ".switching")
# A switch is a process stop plus a cold model load. Past this we assume the
# script died rather than leave the UI claiming a switch forever.
SWITCH_STALE_SECONDS = 15 * 60

OUTPUT_PREFIX = "h3"
VIDEO_EXTS = (".mp4", ".webm", ".mkv", ".mov", ".webp", ".gif")

# H3's own limits, from the model card: 4-15s at 24fps, 768px short side for
# the open weights. The frame grid is 17k+5, which is what makes 124 frames 5s.
FPS = 24
FRAME_STEP = 17
FRAME_BASE = 5
MIN_FRAMES = 107    # 17*6+5, ~4.5s
MAX_FRAMES = 362    # 17*21+5, ~15.1s
MAX_STEPS = 60
# Dimensions land on a multiple of 32 and stay inside the 768x1344 cap the open
# 768p stage was trained for; 1.0MP exceeds it, which is why 0.98MP is the
# practical 16:9 ceiling.
DIM_MULTIPLE = 32
MAX_DIM = 1344
MAX_PIXELS = 768 * 1344

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 64 * 1024 * 1024

# Jobs submitted in this process, keyed by ComfyUI's prompt_id. The durable
# record is the .prompt sidecar next to each finished clip; this is only what
# the page needs to follow work in flight, and it is lost on restart the same
# way web_server's queue is.
_jobs = {}
_jobs_lock = threading.Lock()


# ---------------------------------------------------------------------------
# Auth - same key and the same two ways of presenting it as the other two apps.
# ---------------------------------------------------------------------------
def check_auth():
    provided = request.headers.get("X-API-Key") or request.args.get("api_key")
    if not provided:
        return False
    return hmac.compare_digest(provided, API_KEY)


@app.before_request
def require_auth():
    # /mode is public for the reason web_server's /ready is: the page must be
    # able to say which mode the box is in before a key has been entered, and
    # that answer is not a secret. Clips are served with random filenames,
    # which is the same bet /images/ already makes.
    if request.endpoint in ("index", "api_mode_get", "serve_video"):
        return
    if not API_KEY:
        return jsonify({"success": False, "error": "Server misconfigured: FLUX_API_KEY not set"}), 500
    if not check_auth():
        return jsonify({"success": False, "error": "Unauthorized"}), 401


# ---------------------------------------------------------------------------
# Mode
# ---------------------------------------------------------------------------
def port_is_up(port, host="127.0.0.1"):
    """Whether something is listening. The PowerShell side uses netstat; from
    here a connect attempt is cheaper and needs no parsing."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.35)
        return sock.connect_ex((host, port)) == 0


def powershell_exe():
    """The PowerShell to run switch_mode.ps1 with.

    The real install is preferred over whatever `pwsh` resolves to on PATH,
    because on this box that is the WindowsApps App Execution Alias - a
    reparse point that behaves unlike a normal executable when launched from
    a service-like process. Windows PowerShell 5.1 is the last resort and only
    just works: kill_flux.ps1 and kill_video.ps1 use Process.Parent, which
    needs PowerShell 6+, so a switch run under 5.1 would fail to find the
    process trees it has to stop.
    """
    candidates = []
    for root in (os.environ.get("ProgramFiles", r"C:\Program Files"),
                 os.environ.get("ProgramW6432", r"C:\Program Files")):
        base = os.path.join(root, "PowerShell")
        if os.path.isdir(base):
            # Highest stable version first: a plain "7" beats "7-preview",
            # which a reverse alphabetical sort would otherwise pick.
            def rank(entry):
                prerelease = any(tag in entry.lower()
                                 for tag in ("preview", "rc", "beta", "alpha"))
                major = int(entry.split("-", 1)[0]) if entry.split("-", 1)[0].isdigit() else 0
                return (prerelease, -major, entry)
            for entry in sorted(os.listdir(base), key=rank):
                candidates.append(os.path.join(base, entry, "pwsh.exe"))
    found = shutil.which("pwsh")
    if found and "WindowsApps" not in found:
        candidates.append(found)
    candidates.append(shutil.which("powershell"))
    for candidate in candidates:
        if candidate and os.path.exists(candidate):
            return candidate
    return None


def read_mode():
    try:
        with open(MODE_FILE, "r", encoding="utf-8", errors="replace") as fh:
            value = fh.read().strip().lower()
    except OSError:
        return "flux"
    return value if value in ("flux", "video") else "flux"


def switching_target():
    """The mode a switch in flight is heading for, or None."""
    try:
        if time.time() - os.path.getmtime(SWITCHING_FILE) > SWITCH_STALE_SECONDS:
            return None
        with open(SWITCHING_FILE, "r", encoding="utf-8", errors="replace") as fh:
            target = fh.read().strip().lower().split()[0]
    except (OSError, IndexError):
        return None
    return target if target in ("flux", "video") else None


def mode_state():
    return {
        "mode": read_mode(),
        "switching": switching_target(),
        "flux": {"port": FLUX_PORT, "up": port_is_up(FLUX_PORT)},
        "comfy": {"port": COMFY_PORT, "up": port_is_up(COMFY_PORT)},
        "video_server_port": PORT,
    }


@app.route("/mode", methods=["GET"])
def api_mode_get():
    return jsonify({"success": True, **mode_state()})


@app.route("/mode", methods=["POST"])
def api_mode_post():
    """Ask for a mode. Launches switch_mode.ps1 detached and returns: the
    switch takes minutes (a process stop, then a cold model load), so the
    caller polls GET /mode to watch it land."""
    target = ((request.json or {}).get("mode") or "").strip().lower()
    if target not in ("flux", "video"):
        return jsonify({"success": False, "error": "mode must be 'flux' or 'video'"}), 400

    in_flight = switching_target()
    if in_flight:
        return jsonify({"success": False, "error": f"a switch to {in_flight} is already in flight"}), 409

    state = mode_state()
    settled = state["comfy"]["up"] if target == "video" else state["flux"]["up"]
    if state["mode"] == target and settled:
        return jsonify({"success": True, "already": True, **state})

    script = os.path.join(HERE, "switch_mode.ps1")
    if not os.path.exists(script):
        return jsonify({"success": False, "error": "switch_mode.ps1 not found"}), 500
    shell = powershell_exe()
    if shell is None:
        return jsonify({"success": False, "error": "no PowerShell found to run the switch"}), 500

    try:
        # Not DETACHED_PROCESS. A detached pwsh launched from here exits 0
        # without ever running the script - the switch silently did nothing
        # and the POST still answered 200. CREATE_NO_WINDOW gives the same
        # "no console in the user's face" without that, and
        # CREATE_NEW_PROCESS_GROUP keeps a Ctrl+C here from reaching it.
        #
        # Output goes to a log rather than DEVNULL for the same reason: this
        # child is never waited on, so discarding its output is discarding the
        # only evidence that it failed.
        launch_log = open(os.path.join(HERE, "switch_launch.log"), "ab")
        proc = subprocess.Popen(
            [shell, "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", script, target],
            cwd=HERE,
            stdin=subprocess.DEVNULL,
            stdout=launch_log,
            stderr=subprocess.STDOUT,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0)
            | getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0),
        )
    except OSError as exc:
        return jsonify({"success": False, "error": f"could not launch switch: {exc}"}), 500
    finally:
        try:
            launch_log.close()
        except (NameError, OSError):
            pass

    # Wait just long enough to see it take hold. switch_mode.ps1 writes
    # .switching before it stops anything, so this turns "launched something
    # that did nothing" - which is exactly how this broke - into an error the
    # page can show, instead of a 200 and a UI that never changes.
    deadline = time.time() + 6
    while time.time() < deadline:
        if switching_target() == target:
            break
        code = proc.poll()
        if code is not None:
            if code == 0:
                break
            return jsonify({"success": False,
                            "error": f"the switch exited with code {code} before starting; "
                                     f"see switch_launch.log and switch_mode.log"}), 500
        time.sleep(0.2)
    else:
        return jsonify({"success": False,
                        "error": "the switch was launched but never started; "
                                 "see switch_launch.log"}), 500

    return jsonify({"success": True, "switching_to": target, **mode_state()})


# ---------------------------------------------------------------------------
# Workflow loading and patching
# ---------------------------------------------------------------------------
class WorkflowError(Exception):
    """A workflow or map that cannot serve the request as asked."""


def _workflow_paths(name):
    safe = os.path.basename(name)
    return (os.path.join(WORKFLOW_DIR, safe + ".json"),
            os.path.join(WORKFLOW_DIR, safe + ".map.json"))


def load_workflow(name):
    """Return (graph, map) for a workflow name, or raise WorkflowError.

    `graph` is ComfyUI's API format: {node_id: {class_type, inputs, _meta}}.
    `map` is {field: "<node_id>.<input>"} and is what keeps this file free of
    any particular H3 node layout.
    """
    graph_path, map_path = _workflow_paths(name)
    if not os.path.exists(graph_path):
        raise WorkflowError(
            f"no workflow at {os.path.relpath(graph_path, HERE)} - export ComfyUI's "
            f"H3 template with 'Export (API)' and save it there (see workflows/README.md)")
    try:
        with open(graph_path, "r", encoding="utf-8") as fh:
            graph = json.load(fh)
    except (OSError, ValueError) as exc:
        raise WorkflowError(f"could not read {name}.json: {exc}") from exc
    if not isinstance(graph, dict) or not graph:
        raise WorkflowError(f"{name}.json is not a ComfyUI API-format graph")
    if "nodes" in graph and "links" in graph:
        raise WorkflowError(
            f"{name}.json looks like a UI-format workflow (it has 'nodes'/'links'). "
            f"Re-export it with 'Export (API)', which keys nodes by id.")

    field_map = {}
    if os.path.exists(map_path):
        try:
            with open(map_path, "r", encoding="utf-8") as fh:
                field_map = json.load(fh)
        except (OSError, ValueError) as exc:
            raise WorkflowError(f"could not read {name}.map.json: {exc}") from exc
        if not isinstance(field_map, dict):
            raise WorkflowError(f"{name}.map.json must be an object of field -> \"node.input\"")
        # Keys beginning with _ are notes for whoever reads the file (see
        # workflow_from_template.py's _steps_note), not request fields.
        field_map = {k: v for k, v in field_map.items() if not k.startswith("_")}
    return graph, field_map


def _assign(graph, target, value, field):
    """Write one value into the graph at a "<node_id>.<input>" target."""
    if not isinstance(target, str) or "." not in target:
        raise WorkflowError(f"map entry for '{field}' must look like \"6.text\", got {target!r}")
    node_id, _, input_name = target.partition(".")
    node = graph.get(node_id)
    if node is None:
        raise WorkflowError(f"map entry for '{field}' names node {node_id}, which is not in the graph")
    inputs = node.setdefault("inputs", {})
    if input_name not in inputs:
        # Not fatal on its own - a template can legitimately leave an optional
        # input unset - but it is nearly always a stale map after a re-export,
        # so name the inputs that do exist rather than failing mutely.
        have = ", ".join(sorted(inputs)) or "(none)"
        raise WorkflowError(
            f"map entry for '{field}' points at {node_id}.{input_name}, but node {node_id} "
            f"({node.get('class_type', '?')}) has inputs: {have}")
    if isinstance(inputs[input_name], list):
        # A list here is a link to another node's output, not a literal. Writing
        # over it would silently detach the graph.
        raise WorkflowError(
            f"map entry for '{field}' points at {node_id}.{input_name}, which is wired to "
            f"another node rather than being a literal value")
    inputs[input_name] = value


def patch_workflow(graph, field_map, values):
    """Apply request values to the graph. Fields with no map entry are skipped
    unless they are required; a required field with no entry is an error the
    caller can fix in the map file, so it says so plainly."""
    required = ("prompt", "width", "height", "frames", "steps", "seed")
    missing = [f for f in required if f not in field_map]
    if missing:
        raise WorkflowError(
            "map is missing required field(s): " + ", ".join(missing) +
            " - add them as \"<node_id>.<input>\" (GET /workflow/nodes lists the graph)")

    applied = {}
    for field, value in values.items():
        if value is None:
            continue
        target = field_map.get(field)
        if target is None:
            if field in required:
                raise WorkflowError(f"map has no entry for required field '{field}'")
            # A value was supplied for something this graph cannot express.
            # Dropping it quietly is worse than refusing: the clip would come
            # back having ignored it with nothing to say so. H3's
            # MiniMaxH3ImageToVideo takes no negative prompt, for instance.
            raise WorkflowError(
                f"this workflow has no '{field}' input, so that value cannot be applied "
                f"(the map has: {', '.join(sorted(field_map))})")
        _assign(graph, target, value, field)
        applied[field] = target
    return applied


@app.route("/workflow/nodes")
def api_workflow_nodes():
    """Dump a workflow's nodes so the map can be written from what is actually
    in the graph. This is the tool that replaces guessing at node ids."""
    name = request.args.get("name", "h3_t2va")
    try:
        graph, field_map = load_workflow(name)
    except WorkflowError as exc:
        return jsonify({"success": False, "error": str(exc)}), 400
    nodes = []
    for node_id, node in graph.items():
        literals, links = {}, {}
        for key, value in (node.get("inputs") or {}).items():
            (links if isinstance(value, list) else literals)[key] = value
        nodes.append({
            "id": node_id,
            "class_type": node.get("class_type"),
            "title": (node.get("_meta") or {}).get("title"),
            "literal_inputs": literals,
            "linked_inputs": sorted(links),
        })
    nodes.sort(key=lambda n: (n["class_type"] or "", n["id"]))
    return jsonify({"success": True, "workflow": name, "map": field_map, "nodes": nodes})


@app.route("/workflows")
def api_workflows():
    try:
        names = sorted({os.path.splitext(f)[0].replace(".map", "")
                        for f in os.listdir(WORKFLOW_DIR) if f.endswith(".json")})
    except OSError:
        names = []
    out = []
    for name in names:
        graph_path, map_path = _workflow_paths(name)
        out.append({"name": name,
                    "has_graph": os.path.exists(graph_path),
                    "has_map": os.path.exists(map_path)})
    return jsonify({"success": True, "workflows": out})


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
def _snap_dim(value, field):
    try:
        number = int(value)
    except (TypeError, ValueError):
        raise WorkflowError(f"{field} must be an integer")
    if number < DIM_MULTIPLE:
        raise WorkflowError(f"{field} must be at least {DIM_MULTIPLE}")
    if number > MAX_DIM:
        raise WorkflowError(f"{field} must be at most {MAX_DIM}")
    return int(round(number / DIM_MULTIPLE)) * DIM_MULTIPLE


def _snap_frames(value):
    """Snap to the 17k+5 grid H3 is trained on. Off-grid lengths are accepted
    by the sampler and then produce a clip that drifts at the tail."""
    try:
        number = int(value)
    except (TypeError, ValueError):
        raise WorkflowError("frames must be an integer")
    k = max(0, round((number - FRAME_BASE) / FRAME_STEP))
    snapped = FRAME_STEP * k + FRAME_BASE
    if snapped < MIN_FRAMES or snapped > MAX_FRAMES:
        raise WorkflowError(
            f"frames must land between {MIN_FRAMES} and {MAX_FRAMES} "
            f"({MIN_FRAMES / FPS:.1f}-{MAX_FRAMES / FPS:.1f}s at {FPS}fps)")
    return snapped


def validate_request(payload):
    prompt = (payload.get("prompt") or "").strip()
    if not prompt:
        raise WorkflowError("prompt is required")
    if len(prompt) > 4000:
        raise WorkflowError("prompt must be at most 4000 characters")

    width = _snap_dim(payload.get("width", 864), "width")
    height = _snap_dim(payload.get("height", 480), "height")
    if width * height > MAX_PIXELS:
        raise WorkflowError(
            f"{width}x{height} is {width * height / 1e6:.2f}MP; the open 768p stage caps at "
            f"{MAX_PIXELS / 1e6:.2f}MP (768x1344)")

    frames = _snap_frames(payload.get("frames", 124))

    try:
        steps = int(payload.get("steps", 10))
    except (TypeError, ValueError):
        raise WorkflowError("steps must be an integer")
    if not 1 <= steps <= MAX_STEPS:
        raise WorkflowError(f"steps must be between 1 and {MAX_STEPS}")

    seed = payload.get("seed")
    if seed in (None, "", "random"):
        seed = random.randint(0, 2 ** 32 - 1)
    else:
        try:
            seed = int(seed)
        except (TypeError, ValueError):
            raise WorkflowError("seed must be an integer")
        if not 0 <= seed < 2 ** 63:
            raise WorkflowError("seed out of range")

    negative = (payload.get("negative") or "").strip() or None

    return {
        "prompt": prompt,
        "negative": negative,
        "width": width,
        "height": height,
        "frames": frames,
        "steps": steps,
        "seed": seed,
    }


# ---------------------------------------------------------------------------
# ComfyUI
# ---------------------------------------------------------------------------
def comfy_post_prompt(graph, client_id):
    response = requests.post(f"{COMFY_URL}/prompt",
                             json={"prompt": graph, "client_id": client_id},
                             timeout=30)
    if response.status_code >= 400:
        # ComfyUI reports a graph it will not run as a structured
        # node_errors/error body; surfacing it verbatim is the difference
        # between a fixable message and "500".
        try:
            detail = response.json()
        except ValueError:
            detail = {"error": response.text[:2000]}
        raise WorkflowError(f"ComfyUI rejected the workflow: {json.dumps(detail)[:2000]}")
    return response.json()


def comfy_upload_image(file_storage):
    """Put a start/end frame where ComfyUI's LoadImage can see it, and return
    the name to write into the graph."""
    files = {"image": (file_storage.filename or "start.png",
                       file_storage.stream,
                       file_storage.mimetype or "application/octet-stream")}
    response = requests.post(f"{COMFY_URL}/upload/image",
                             files=files, data={"overwrite": "true"}, timeout=60)
    if response.status_code >= 400:
        raise WorkflowError(f"ComfyUI rejected the image upload: {response.text[:500]}")
    body = response.json()
    name = body.get("name")
    if not name:
        raise WorkflowError("ComfyUI returned no filename for the uploaded image")
    subfolder = body.get("subfolder") or ""
    return f"{subfolder}/{name}" if subfolder else name


def _outputs_to_files(outputs):
    """Pull every file reference out of a /history outputs block.

    Which key holds a video depends on the saving node (`gifs` for the
    VideoHelperSuite lineage, `videos` for Comfy's own, `images` for a node
    that writes a webp), so every list of dicts carrying a filename counts and
    extension decides what it is.
    """
    found = []
    for node_output in (outputs or {}).values():
        if not isinstance(node_output, dict):
            continue
        for entries in node_output.values():
            if not isinstance(entries, list):
                continue
            for entry in entries:
                if isinstance(entry, dict) and entry.get("filename"):
                    found.append(entry)
    return found


def comfy_fetch_file(entry):
    params = {
        "filename": entry.get("filename"),
        "subfolder": entry.get("subfolder") or "",
        "type": entry.get("type") or "output",
    }
    response = requests.get(f"{COMFY_URL}/view", params=params, timeout=300)
    response.raise_for_status()
    return response.content


def _output_name(ext):
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{OUTPUT_PREFIX}_{stamp}_{os.urandom(4).hex()}{ext}"


def _write_sidecar(path, job, entry):
    """A `.prompt` sidecar in the same shape the image outputs use, so the
    `# Prompt:` line stays the one format every tool here parses."""
    params = job["params"]
    lines = [
        f"# Prompt: {params['prompt']}",
        f"# Model: MiniMax H3 ({job.get('workflow', 'h3_t2va')})",
        f"# Seed: {params['seed']}",
        f"# Steps: {params['steps']}",
        f"# Size: {params['width']}x{params['height']}",
        f"# Frames: {params['frames']} ({params['frames'] / FPS:.2f}s at {FPS}fps)",
        f"# Submitted: {datetime.fromtimestamp(job['submitted']).isoformat(timespec='seconds')}",
        f"# ComfyUI: {entry.get('subfolder') or '.'}/{entry.get('filename')}",
    ]
    if params.get("negative"):
        lines.insert(1, f"# Negative: {params['negative']}")
    if job.get("start_image"):
        lines.append(f"# Start image: {job['start_image']}")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")


def harvest(prompt_id):
    """Copy a finished job's clip out of ComfyUI into web-generated/video/.

    The clip is pulled over HTTP from ComfyUI's /view rather than read off
    disk, so this app never needs to know where that install keeps its output
    directory - which matters because the whole point of the second checkout
    is that the path can change.
    """
    response = requests.get(f"{COMFY_URL}/history/{prompt_id}", timeout=30)
    response.raise_for_status()
    record = (response.json() or {}).get(prompt_id)
    if not record:
        return None

    status = (record.get("status") or {})
    if status.get("status_str") == "error" or status.get("completed") is False:
        messages = status.get("messages") or []
        return {"state": "error", "detail": json.dumps(messages)[:2000]}

    entries = _outputs_to_files(record.get("outputs"))
    videos = [e for e in entries
              if os.path.splitext(e.get("filename", ""))[1].lower() in VIDEO_EXTS]
    if not videos:
        if status.get("completed"):
            return {"state": "error",
                    "detail": "ComfyUI finished but produced no video file - does the "
                              "workflow have a saving node (SaveVideo / VHS Combine)?"}
        return None

    with _jobs_lock:
        job = _jobs.get(prompt_id)
    if job is None:
        return None

    saved = []
    for entry in videos:
        ext = os.path.splitext(entry["filename"])[1].lower() or ".mp4"
        name = _output_name(ext)
        path = os.path.join(VIDEO_DIR, name)
        try:
            data = comfy_fetch_file(entry)
        except requests.RequestException as exc:
            return {"state": "error", "detail": f"could not fetch {entry['filename']}: {exc}"}
        with open(path, "wb") as fh:
            fh.write(data)
        _write_sidecar(os.path.splitext(path)[0] + ".prompt", job, entry)
        saved.append(name)
    return {"state": "done", "files": saved}


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------
@app.route("/generate", methods=["POST"])
def api_generate():
    """Submit one clip to ComfyUI's queue.

    Multipart with a `start_image` file is how first-frame conditioning
    arrives; everything else is JSON, or the same fields as form values
    alongside the file.
    """
    state = mode_state()
    if state["switching"]:
        return jsonify({"success": False, "error": f"switching to {state['switching']}; try again shortly",
                        **state}), 409
    if state["flux"]["up"]:
        # The hard guard, independent of .mode: both resident at once is the
        # sysmem-fallback stall, which has no error and no log line.
        return jsonify({"success": False,
                        "error": f"the image generator is still up on port {FLUX_PORT}; "
                                 f"POST /mode {{\"mode\":\"video\"}} first",
                        **state}), 409
    if not state["comfy"]["up"]:
        return jsonify({"success": False,
                        "error": f"ComfyUI is not listening on port {COMFY_PORT}; "
                                 f"POST /mode {{\"mode\":\"video\"}} to start it",
                        **state}), 503

    if request.content_type and request.content_type.startswith("multipart/"):
        payload = dict(request.form)
    else:
        payload = request.json or {}

    workflow_name = (payload.get("workflow") or "h3_t2va").strip()

    try:
        params = validate_request(payload)
        graph, field_map = load_workflow(workflow_name)
        values = dict(params)
        start_image = request.files.get("start_image")
        uploaded = None
        if start_image is not None and start_image.filename:
            if "start_image" not in field_map:
                raise WorkflowError(
                    "this workflow's map has no 'start_image' entry, so it cannot take a "
                    "first frame - use the fl2va (image-to-video) template for that")
            uploaded = comfy_upload_image(start_image)
            values["start_image"] = uploaded
        applied = patch_workflow(graph, field_map, values)
    except WorkflowError as exc:
        return jsonify({"success": False, "error": str(exc)}), 400
    except requests.RequestException as exc:
        return jsonify({"success": False, "error": f"ComfyUI unreachable: {exc}"}), 502

    client_id = f"video_server-{os.getpid()}"
    try:
        result = comfy_post_prompt(graph, client_id)
    except WorkflowError as exc:
        return jsonify({"success": False, "error": str(exc)}), 400
    except requests.RequestException as exc:
        return jsonify({"success": False, "error": f"ComfyUI unreachable: {exc}"}), 502

    prompt_id = result.get("prompt_id")
    if not prompt_id:
        return jsonify({"success": False, "error": f"ComfyUI returned no prompt_id: {result}"}), 502

    with _jobs_lock:
        _jobs[prompt_id] = {
            "prompt_id": prompt_id,
            "params": params,
            "workflow": workflow_name,
            "start_image": uploaded,
            "submitted": time.time(),
            "state": "queued",
            "files": [],
            "applied": applied,
        }

    return jsonify({"success": True, "prompt_id": prompt_id, "params": params,
                    "number": result.get("number")})


def _job_view(job):
    params = job["params"]
    return {
        "prompt_id": job["prompt_id"],
        "state": job["state"],
        "prompt": params["prompt"],
        "seed": params["seed"],
        "steps": params["steps"],
        "width": params["width"],
        "height": params["height"],
        "frames": params["frames"],
        "seconds": round(params["frames"] / FPS, 2),
        "submitted": job["submitted"],
        "elapsed": round(time.time() - job["submitted"], 1),
        "files": job.get("files") or [],
        "error": job.get("error"),
    }


@app.route("/status")
def api_status():
    """Everything the page needs in one poll: mode, ComfyUI's own queue, and
    this process's jobs, harvesting any that finished since the last call."""
    state = mode_state()
    queue = {"running": 0, "pending": 0}
    if state["comfy"]["up"]:
        try:
            body = requests.get(f"{COMFY_URL}/queue", timeout=5).json()
            queue = {"running": len(body.get("queue_running") or []),
                     "pending": len(body.get("queue_pending") or [])}
        except (requests.RequestException, ValueError):
            pass

    with _jobs_lock:
        pending_ids = [pid for pid, job in _jobs.items()
                       if job["state"] in ("queued", "running")]

    for prompt_id in pending_ids:
        try:
            outcome = harvest(prompt_id)
        except (requests.RequestException, ValueError, OSError) as exc:
            outcome = {"state": "error", "detail": str(exc)}
        if not outcome:
            continue
        with _jobs_lock:
            job = _jobs.get(prompt_id)
            if job is None:
                continue
            job["state"] = outcome["state"]
            if outcome["state"] == "done":
                job["files"] = outcome["files"]
            else:
                job["error"] = outcome.get("detail")

    with _jobs_lock:
        jobs = [_job_view(job) for job in
                sorted(_jobs.values(), key=lambda j: j["submitted"], reverse=True)[:25]]

    return jsonify({"success": True, "queue": queue, "jobs": jobs, **state})


@app.route("/cancel", methods=["POST"])
def api_cancel():
    """Interrupt what ComfyUI is running, or clear its pending queue."""
    scope = ((request.json or {}).get("scope") or "running").lower()
    if not port_is_up(COMFY_PORT):
        return jsonify({"success": False, "error": "ComfyUI is not running"}), 503
    try:
        if scope == "all":
            requests.post(f"{COMFY_URL}/queue", json={"clear": True}, timeout=10).raise_for_status()
        requests.post(f"{COMFY_URL}/interrupt", timeout=10).raise_for_status()
    except requests.RequestException as exc:
        return jsonify({"success": False, "error": f"ComfyUI unreachable: {exc}"}), 502
    return jsonify({"success": True, "scope": scope})


# ---------------------------------------------------------------------------
# Finished clips
# ---------------------------------------------------------------------------
def _safe_video_name(name):
    if not name or "/" in name or "\\" in name or name.startswith("."):
        return None
    if os.path.splitext(name)[1].lower() not in VIDEO_EXTS:
        return None
    return name


@app.route("/videos")
def api_videos():
    try:
        names = [n for n in os.listdir(VIDEO_DIR) if _safe_video_name(n)]
    except OSError:
        names = []
    out = []
    for name in names:
        path = os.path.join(VIDEO_DIR, name)
        try:
            stat = os.stat(path)
        except OSError:
            continue
        prompt = ""
        try:
            with open(os.path.splitext(path)[0] + ".prompt", "r", encoding="utf-8",
                      errors="replace") as fh:
                for line in fh:
                    if line.startswith("# Prompt:"):
                        prompt = line[len("# Prompt:"):].strip()
                        break
        except OSError:
            pass
        out.append({"name": name, "size": stat.st_size, "mtime": stat.st_mtime,
                    "prompt": prompt})
    out.sort(key=lambda v: v["mtime"], reverse=True)
    return jsonify({"success": True, "videos": out[:200], "dir": "web-generated/video"})


@app.route("/videos/<path:name>")
def serve_video(name):
    safe = _safe_video_name(name)
    if not safe:
        return jsonify({"success": False, "error": "invalid filename"}), 400
    # conditional=True so the browser can range-request: without it a <video>
    # element cannot seek in an mp4 served by Flask.
    return send_from_directory(VIDEO_DIR, safe, conditional=True)


@app.route("/delete", methods=["POST"])
def api_delete():
    safe = _safe_video_name(((request.json or {}).get("name") or ""))
    if not safe:
        return jsonify({"success": False, "error": "invalid filename"}), 400
    path = os.path.join(VIDEO_DIR, safe)
    if not os.path.exists(path):
        return jsonify({"success": False, "error": "not found"}), 404
    os.remove(path)
    sidecar = os.path.splitext(path)[0] + ".prompt"
    if os.path.exists(sidecar):
        os.remove(sidecar)
    return jsonify({"success": True, "deleted": safe})


# ---------------------------------------------------------------------------
# Prompt boost
# ---------------------------------------------------------------------------
# Longer than this is a draft prompt in the wrong field; same cap web_server
# uses so the two behave alike.
BOOST_STEER_MAX = 2000
# What the VLM sees of a first frame. It resamples to its own patch grid at
# 896px anyway (edit_loop.img_b64), so sending generation-sized pixels is
# upload time for detail nothing reads.
BOOST_REF_EDGE = 1024
VLM_URL = (os.environ.get("OLLAMA_URL") or "http://127.0.0.1:11434").rstrip("/")
BOOST_MODEL = (os.environ.get("BOOST_MODEL")
               or os.environ.get("CRITIQUE_MODEL") or "auto")


@app.route("/boost", methods=["POST"])
def api_boost():
    """Rewrite a draft prompt into one written for H3.

    Unlike web_server's /boost this answers synchronously. There is no Boost
    xN or Evolve here to make a job id worth it, and one call is ~10-15s
    against a remote VLM.

    It talks to edit_loop directly rather than proxying to web_server,
    because web_server is *down* in video mode - which is exactly when this
    is wanted. The VLM lives at OLLAMA_URL, normally another box, so it costs
    this card nothing. Pointed at a local ollama it would compete with H3 for
    VRAM, which edit_loop.KEEP_ALIVE="0" mitigates by unloading after each
    call.
    """
    try:
        import edit_loop
    except ImportError as exc:
        return jsonify({"success": False, "error": f"edit_loop unavailable: {exc}"}), 503

    data = request.json or {}
    prompt = (data.get("prompt") or "").strip()
    if not prompt:
        return jsonify({"success": False, "error": "prompt is required"}), 400
    try:
        level = int(data.get("level", 3))
    except (TypeError, ValueError):
        return jsonify({"success": False, "error": "level must be an integer 1-5"}), 400
    if not 1 <= level <= 5:
        return jsonify({"success": False, "error": "level must be an integer 1-5"}), 400

    steer = (data.get("steer") or "").strip() or None
    if steer and len(steer) > BOOST_STEER_MAX:
        return jsonify({"success": False,
                        "error": f"steer must be at most {BOOST_STEER_MAX} characters"}), 400

    # has_image picks the prompting idiom and stays true whether or not the
    # caller chose to show the picture; input_images is the separate decision
    # to spend a vision pass grounding the rewrite in it.
    has_image = bool(data.get("has_image"))
    images = []
    if data.get("input_images"):
        try:
            import base64
            import io as _io
            from PIL import Image
        except ImportError as exc:
            return jsonify({"success": False, "error": f"Pillow unavailable: {exc}"}), 503
        for i, b64 in enumerate(data["input_images"][:1]):
            if not isinstance(b64, str):
                return jsonify({"success": False,
                                "error": f"input_images[{i}] is not a base64 image"}), 400
            try:
                payload = b64.split(",", 1)[1] if "," in b64 else b64
                image = Image.open(_io.BytesIO(base64.b64decode(payload))).convert("RGB")
            except Exception:
                return jsonify({"success": False,
                                "error": f"input_images[{i}] is not a decodable base64 image"}), 400
            image.thumbnail((BOOST_REF_EDGE, BOOST_REF_EDGE), Image.LANCZOS)
            images.append(image)
        has_image = has_image or bool(images)

    # Clip shape, so the storyboard it writes fits the seconds that exist.
    try:
        frames = _snap_frames(data.get("frames", 124))
        width = _snap_dim(data.get("width", 864), "width")
        height = _snap_dim(data.get("height", 480), "height")
    except WorkflowError as exc:
        return jsonify({"success": False, "error": str(exc)}), 400
    seconds = frames / FPS
    model_desc = (f"MiniMax H3, which generates picture and synchronized stereo audio "
                  f"together, for a {seconds:.1f} second clip at {FPS}fps, {width}x{height}")

    model = data.get("model") or BOOST_MODEL
    try:
        resolved = edit_loop.resolve_vlm_model(VLM_URL, model)
    except Exception as exc:
        return jsonify({"success": False, "error": f"no VLM at {VLM_URL}: {exc}"}), 502
    if not resolved:
        return jsonify({"success": False,
                        "error": f"no vision model available at {VLM_URL}"}), 502

    started = time.time()
    try:
        result = edit_loop.vlm_boost(
            resolved, prompt,
            family="h3-i2v" if has_image else "h3",
            model_desc=model_desc,
            level=level,
            think=bool(data.get("think", False)),
            ollama_url=VLM_URL,
            images=images or None,
            steer=steer,
            medium="video",
        )
    except Exception as exc:
        return jsonify({"success": False, "error": f"boost failed: {exc}"}), 502

    if not result or not result.get("prompt"):
        # edit_loop returns None on any failure, having already logged it.
        return jsonify({"success": False,
                        "error": f"no rewrite from the VLM at {VLM_URL} ({resolved})"}), 502

    return jsonify({
        "success": True,
        "prompt": result["prompt"],
        "original": prompt,
        "model": resolved,
        "family": "h3-i2v" if has_image else "h3",
        "saw_images": len(images),
        "level": level,
        "elapsed": round(time.time() - started, 1),
    })


@app.route("/vlm")
def api_vlm():
    """What the boost button will talk to, for the page's status line."""
    try:
        import edit_loop
    except ImportError as exc:
        return jsonify({"success": False, "error": str(exc), "url": VLM_URL}), 200
    try:
        resolved = edit_loop.resolve_vlm_model(VLM_URL, BOOST_MODEL)
    except Exception as exc:
        return jsonify({"success": False, "error": str(exc), "url": VLM_URL}), 200
    return jsonify({"success": bool(resolved), "url": VLM_URL, "model": resolved})


@app.route("/ready")
def ready():
    return jsonify({"success": True, "ready": True, **mode_state()})


@app.route("/")
def index():
    return Response(PAGE, mimetype="text/html")


PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>FLUX Video</title>
<style>
:root{--bg:#14161a;--panel:#1d2026;--line:#2e333c;--fg:#e7e9ee;--dim:#949aa6;
--accent:#6aa9ff;--warn:#f0a44a;--bad:#e2605a;--good:#5fbf7f}
*{box-sizing:border-box}
body{margin:0;padding:16px;background:var(--bg);color:var(--fg);
font:14px/1.5 system-ui,-apple-system,Segoe UI,sans-serif}
h1{font-size:19px;margin:0 0 12px}
.wrap{max-width:1100px;margin:0 auto}
.panel{background:var(--panel);border:1px solid var(--line);border-radius:10px;
padding:14px;margin-bottom:14px}
label{display:block;font-size:12px;color:var(--dim);margin:10px 0 4px}
input,select,textarea,button{font:inherit;color:var(--fg);background:#262a32;
border:1px solid var(--line);border-radius:7px;padding:8px 10px}
textarea{width:100%;min-height:74px;resize:vertical}
input,select{width:100%}
button{cursor:pointer;background:#2f3640;border-color:#3a4150}
button:hover:not(:disabled){border-color:var(--accent)}
button:disabled{opacity:.45;cursor:not-allowed}
button.primary{background:var(--accent);color:#07101f;border-color:var(--accent);font-weight:600}
.row{display:flex;gap:10px;flex-wrap:wrap;align-items:flex-end}
.row>div{flex:1 1 120px;min-width:0}
.bar{display:flex;gap:10px;align-items:center;flex-wrap:wrap}
.pill{font-size:12px;padding:3px 9px;border-radius:999px;border:1px solid var(--line)}
.pill.on{background:#18351f;border-color:#2c5a38;color:var(--good)}
.pill.off{background:#2a1c1c;border-color:#53302e;color:var(--bad)}
.pill.busy{background:#332715;border-color:#5d4523;color:var(--warn)}
.muted{color:var(--dim);font-size:12px}
.job{border-top:1px solid var(--line);padding:9px 0;font-size:13px}
.job:first-child{border-top:0}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(260px,1fr));gap:12px}
.card video{width:100%;border-radius:8px;background:#000;display:block}
.card p{margin:6px 0 0;font-size:12px;color:var(--dim);
overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.err{color:var(--bad);font-size:12px;white-space:pre-wrap;word-break:break-word}
@media(max-width:640px){body{padding:10px}}
</style></head><body><div class="wrap">
<h1>FLUX Video &mdash; MiniMax H3</h1>

<div class="panel">
  <div class="bar">
    <span id="mode" class="pill">mode &hellip;</span>
    <span id="pFlux" class="pill">flux</span>
    <span id="pComfy" class="pill">comfy</span>
    <span id="pQueue" class="pill">queue</span>
    <span style="flex:1"></span>
    <button id="toVideo">Stop flux &rarr; start video</button>
    <button id="toFlux">Stop video &rarr; start flux</button>
  </div>
  <p class="muted" id="modeNote" style="margin:10px 0 0">
    One GPU, one tenant: the image generator and H3 cannot be resident at the same time.</p>
</div>

<div class="panel">
  <label for="key">API key</label>
  <input id="key" type="password" placeholder="X-API-Key" autocomplete="off">
  <label for="prompt">Prompt</label>
  <textarea id="prompt" placeholder="A fox trotting through tall grass at golden hour, handheld camera"></textarea>
  <!-- No negative prompt: MiniMaxH3ImageToVideo has no such input, so a field
       here could only mislead. The server refuses one rather than drop it. -->
  <div class="bar" style="margin-top:8px">
    <button id="boost" style="flex:0 0 auto">&#10022; Boost</button>
    <select id="blevel" style="width:auto" title="How far the rewrite may depart from your draft">
      <option value="1">1 &mdash; polish</option>
      <option value="2">2 &mdash; tighten</option>
      <option value="3" selected>3 &mdash; flesh out</option>
      <option value="4">4 &mdash; rework</option>
      <option value="5">5 &mdash; reimagine</option>
    </select>
    <button id="steerChip" style="flex:0 0 auto" title="Say how to rewrite it, not what to depict">steer</button>
    <label id="refChip" class="muted" style="display:none;margin:0">
      <input type="checkbox" id="boostRef" style="width:auto;vertical-align:-2px"> show first frame</label>
    <button id="undoBoost" style="flex:0 0 auto;display:none">undo</button>
    <span style="flex:1"></span>
    <span class="muted" id="vlm"></span>
  </div>
  <input id="steer" style="display:none;margin-top:6px"
         placeholder="how to rewrite it &mdash; &quot;set it at night&quot;, &quot;keep it under 40 words&quot;, &quot;no dialogue&quot;">
  <div class="row">
    <div><label for="size">Size</label><select id="size">
      <option value="864x480" selected>864&times;480 &mdash; 0.41MP, fastest</option>
      <option value="1024x576">1024&times;576 &mdash; 0.59MP</option>
      <option value="1152x640">1152&times;640 &mdash; 0.74MP</option>
      <option value="1344x768">1344&times;768 &mdash; 0.98MP, native 16:9</option>
      <option value="640x640">640&times;640 &mdash; 1:1</option>
      <option value="768x1344">768&times;1344 &mdash; 9:16 vertical</option>
    </select></div>
    <div><label for="frames">Length</label><select id="frames">
      <option value="124">124 frames &mdash; 5.2s</option>
      <option value="192">192 frames &mdash; 8.0s</option>
      <option value="243">243 frames &mdash; 10.1s</option>
      <option value="362">362 frames &mdash; 15.1s</option>
    </select></div>
    <div><label for="steps">Steps</label><input id="steps" type="number" value="10" min="1" max="60"></div>
    <div><label for="seed">Seed</label><input id="seed" placeholder="random"></div>
  </div>
  <label for="start">First frame (optional &mdash; image to video)</label>
  <input id="start" type="file" accept="image/*">
  <div class="row" style="margin-top:12px">
    <button id="go" class="primary" style="flex:0 0 auto">Generate</button>
    <button id="cancel" style="flex:0 0 auto">Interrupt</button>
    <span class="muted" id="hint">~175s for 10s at 864&times;480, 10 steps on a 5090.</span>
  </div>
  <p id="err" class="err"></p>
</div>

<div class="panel"><strong style="font-size:13px">Jobs</strong>
  <div id="jobs"><p class="muted">Nothing submitted yet.</p></div></div>

<div class="panel"><strong style="font-size:13px">Clips</strong>
  <div id="clips" class="grid" style="margin-top:10px"></div></div>
</div>
<script>
const $ = id => document.getElementById(id);
const KEY = 'flux_video_key';
$('key').value = localStorage.getItem(KEY) || '';
$('key').addEventListener('change', () => localStorage.setItem(KEY, $('key').value.trim()));

const hdrs = (extra = {}) => Object.assign({'X-API-Key': $('key').value.trim()}, extra);
const setErr = m => { $('err').textContent = m || ''; };

function pill(el, up, label) {
  el.textContent = label;
  el.className = 'pill ' + (up ? 'on' : 'off');
}

async function poll() {
  let s;
  try {
    const r = await fetch('/status', {headers: hdrs()});
    s = await r.json();
    if (!s.success) { s = await (await fetch('/mode')).json(); }
  } catch (e) { return; }

  const sw = s.switching;
  $('mode').textContent = sw ? 'switching to ' + sw : 'mode: ' + s.mode;
  $('mode').className = 'pill ' + (sw ? 'busy' : 'on');
  pill($('pFlux'), s.flux.up, 'flux ' + s.flux.port + (s.flux.up ? ' up' : ' down'));
  pill($('pComfy'), s.comfy.up, 'comfy ' + s.comfy.port + (s.comfy.up ? ' up' : ' down'));
  if (s.queue) {
    $('pQueue').textContent = 'queue ' + s.queue.running + ' running / ' + s.queue.pending + ' pending';
    $('pQueue').className = 'pill' + (s.queue.running ? ' busy' : '');
  }

  const ready = s.comfy.up && !s.flux.up && !sw;
  $('go').disabled = !ready;
  $('toVideo').disabled = !!sw || (s.comfy.up && !s.flux.up);
  $('toFlux').disabled = !!sw || (s.flux.up && !s.comfy.up);
  $('modeNote').textContent = sw
    ? 'Switching to ' + sw + '. This takes a couple of minutes: a process stop, then a cold model load.'
    : ready ? 'Video mode. The image generator is down; H3 has the card.'
    : s.flux.up ? 'Image mode. Stop flux first to free the ~27GB H3 needs.'
    : 'Neither service is up.';

  if (s.jobs) renderJobs(s.jobs);
}

function renderJobs(jobs) {
  if (!jobs.length) { $('jobs').innerHTML = '<p class="muted">Nothing submitted yet.</p>'; return; }
  $('jobs').innerHTML = jobs.map(j => {
    const tag = j.state === 'done' ? 'on' : j.state === 'error' ? 'off' : 'busy';
    return '<div class="job"><span class="pill ' + tag + '">' + j.state + '</span> ' +
      '<strong>' + esc(j.prompt.slice(0, 80)) + '</strong>' +
      '<div class="muted">' + j.width + '×' + j.height + ' · ' + j.frames +
      'f (' + j.seconds + 's) · ' + j.steps + ' steps · seed ' + j.seed +
      ' · ' + j.elapsed + 's elapsed</div>' +
      (j.error ? '<div class="err">' + esc(j.error) + '</div>' : '') + '</div>';
  }).join('');
}

const esc = s => String(s).replace(/[&<>"]/g, c =>
  ({'&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;'}[c]));

async function loadClips() {
  try {
    const r = await fetch('/videos', {headers: hdrs()});
    const b = await r.json();
    if (!b.success) return;
    $('clips').innerHTML = b.videos.map(v =>
      '<div class="card"><video src="/videos/' + encodeURIComponent(v.name) +
      '" controls preload="metadata"></video><p title="' + esc(v.prompt) + '">' +
      esc(v.prompt || v.name) + '</p></div>').join('') ||
      '<p class="muted">No clips yet.</p>';
  } catch (e) {}
}

$('toVideo').onclick = () => switchMode('video');
$('toFlux').onclick = () => switchMode('flux');
async function switchMode(mode) {
  const verb = mode === 'video' ? 'Stop the image generator and start ComfyUI + H3?'
                                : 'Stop ComfyUI and bring the image generator back?';
  if (!confirm(verb)) return;
  setErr('');
  try {
    const r = await fetch('/mode', {method: 'POST', headers: hdrs({'Content-Type': 'application/json'}),
                                    body: JSON.stringify({mode})});
    const b = await r.json();
    if (!b.success) setErr(b.error || 'switch failed');
  } catch (e) { setErr(String(e)); }
  poll();
}

$('cancel').onclick = async () => {
  await fetch('/cancel', {method: 'POST', headers: hdrs({'Content-Type': 'application/json'}),
                          body: JSON.stringify({scope: 'running'})});
  poll();
};

$('go').onclick = async () => {
  setErr('');
  const [w, h] = $('size').value.split('x').map(Number);
  const fields = {
    prompt: $('prompt').value.trim(),
    width: w, height: h, frames: Number($('frames').value),
    steps: Number($('steps').value), seed: $('seed').value.trim(),
    workflow: $('start').files[0] ? 'h3_fl2va' : 'h3_t2va',
  };
  const file = $('start').files[0];
  let opts;
  if (file) {
    const fd = new FormData();
    Object.entries(fields).forEach(([k, v]) => fd.append(k, v));
    fd.append('start_image', file);
    opts = {method: 'POST', headers: hdrs(), body: fd};
  } else {
    opts = {method: 'POST', headers: hdrs({'Content-Type': 'application/json'}),
            body: JSON.stringify(fields)};
  }
  $('go').disabled = true;
  try {
    const b = await (await fetch('/generate', opts)).json();
    if (!b.success) setErr(b.error || 'generate failed');
  } catch (e) { setErr(String(e)); }
  poll();
};

// --- boost ---------------------------------------------------------------
// The first frame is only read when "show first frame" is ticked: it makes the
// rewrite a vision call, which grounds it in what H3 will actually start from
// but costs a vision pass. has_image stays true either way, because that only
// picks which guidance the VLM is given.
const readAsDataUrl = file => new Promise((res, rej) => {
  const fr = new FileReader();
  fr.onload = () => res(fr.result);
  fr.onerror = () => rej(fr.error);
  fr.readAsDataURL(file);
});

$('start').addEventListener('change', () => {
  $('refChip').style.display = $('start').files[0] ? '' : 'none';
});

$('steerChip').onclick = () => {
  const box = $('steer');
  const showing = box.style.display !== 'none';
  box.style.display = showing ? 'none' : '';
  // Closing clears it rather than hiding a setting that still applies.
  if (showing) box.value = '';
  else box.focus();
};

let priorPrompt = null;
$('undoBoost').onclick = () => {
  if (priorPrompt === null) return;
  $('prompt').value = priorPrompt;
  priorPrompt = null;
  $('undoBoost').style.display = 'none';
};

$('boost').onclick = async () => {
  const draft = $('prompt').value.trim();
  if (!draft) { setErr('Write a draft prompt first.'); return; }
  setErr('');
  const [w, h] = $('size').value.split('x').map(Number);
  const file = $('start').files[0];
  const body = {
    prompt: draft, level: Number($('blevel').value),
    steer: $('steer').style.display === 'none' ? '' : $('steer').value.trim(),
    has_image: !!file, width: w, height: h, frames: Number($('frames').value),
  };
  if (file && $('boostRef').checked) {
    try { body.input_images = [await readAsDataUrl(file)]; }
    catch (e) { setErr('Could not read the first frame: ' + e); return; }
  }
  const btn = $('boost');
  const label = btn.textContent;
  btn.disabled = true;
  btn.textContent = 'Boosting…';
  try {
    const r = await fetch('/boost', {method: 'POST',
      headers: hdrs({'Content-Type': 'application/json'}), body: JSON.stringify(body)});
    const b = await r.json();
    if (!b.success) { setErr(b.error || 'boost failed'); }
    else {
      priorPrompt = draft;
      $('prompt').value = b.prompt;
      $('undoBoost').style.display = '';
      $('vlm').textContent = b.model + ' · ' + b.elapsed + 's' +
        (b.saw_images ? ' · saw the frame' : '');
    }
  } catch (e) { setErr(String(e)); }
  btn.disabled = false;
  btn.textContent = label;
};

(async () => {
  try {
    const b = await (await fetch('/vlm', {headers: hdrs()})).json();
    $('vlm').textContent = b.success ? b.model : ('no VLM at ' + b.url);
  } catch (e) {}
})();

poll(); loadClips();
setInterval(poll, 3000);
setInterval(loadClips, 15000);
</script></body></html>
"""


def main():
    parser = argparse.ArgumentParser(description="FLUX Video Server (MiniMax H3 via ComfyUI)")
    parser.add_argument("--port", type=int, default=PORT)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--probe", metavar="WORKFLOW", nargs="?", const="h3_t2va",
                        help="print a workflow's nodes and exit (same data as "
                             "GET /workflow/nodes, for writing the map file)")
    args = parser.parse_args()

    if args.probe:
        try:
            graph, field_map = load_workflow(args.probe)
        except WorkflowError as exc:
            print(f"ERROR: {exc}")
            return 1
        print(f"workflow: {args.probe}   map: {field_map or '(none yet)'}\n")
        for node_id, node in sorted(graph.items(), key=lambda kv: int(kv[0])
                                    if kv[0].isdigit() else 0):
            title = (node.get("_meta") or {}).get("title") or ""
            print(f"  {node_id:>4}  {node.get('class_type', '?'):<34} {title}")
            for key, value in (node.get("inputs") or {}).items():
                if isinstance(value, list):
                    continue
                shown = repr(value)
                print(f"        .{key:<22} = {shown[:70]}")
        return 0

    if not API_KEY:
        print("ERROR: FLUX_API_KEY is not set. Put it in .env or export it before starting.")
        return 1

    print(f"Video server: ComfyUI at {COMFY_URL}, clips in {VIDEO_DIR}")
    print(f"Mode is '{read_mode()}' (.mode); flux on {FLUX_PORT}, comfy on {COMFY_PORT}")
    print(f"Listening on http://{args.host}:{args.port}  (hostname: {socket.gethostname()})")
    app.run(host=args.host, port=args.port, threaded=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
