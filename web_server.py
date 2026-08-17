import os
import argparse
import hmac
import re
import json
import subprocess
import threading
import time
import base64
import io
import math
import socket
import shutil
import traceback
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional
import random
import uuid
from flask import (Flask, request, jsonify, send_from_directory, Response,
                   stream_with_context, has_request_context)
from PIL import Image, ImageDraw, ImageFont
from dotenv import load_dotenv

# The /api/v1 REST layer. It holds no logic of its own — it routes to the
# _api_* core functions below — but it is imported this early because
# require_auth needs its URL prefix to pick an error dialect.
import rest_api

# Load .env file if it exists
load_dotenv()

# Version number - update this when releasing new versions
VERSION = "1.3.0"

# Import model components from flux_core (model loading + generation).
# `--sdxl` swaps in sd_core, the uncensored Stable Diffusion XL backend —
# it mirrors the slice of flux_core's surface this server uses, so every
# flux_core.* reference below resolves against whichever core is active.
import sys
_SDXL_ACTIVE = '--sdxl' in sys.argv
if _SDXL_ACTIVE:
    import sd_core as flux_core
else:
    import flux_core
load_model = flux_core.load_model
generate_image = flux_core.generate_image
device = flux_core.device
save_prompt_file = flux_core.save_prompt_file
load_turbo_lora = flux_core.load_turbo_lora
load_uncensored_lora = flux_core.load_uncensored_lora
MAX_REFERENCE_IMAGES = flux_core.MAX_REFERENCE_IMAGES


def _output_prefix():
    """Filename prefix for generated images: sdxl_... or flux{1|2}_..."""
    return getattr(flux_core, 'OUTPUT_PREFIX', None) or f"flux{flux_core._flux_version}"

app = Flask(__name__)
# Bound request bodies (base64 input images are the largest legitimate payload).
# Without this, Flask accepts unbounded uploads — a trivial memory-DoS vector.
app.config['MAX_CONTENT_LENGTH'] = 64 * 1024 * 1024

# Security: Load API key from environment
API_KEY = os.environ.get("FLUX_API_KEY")
if API_KEY:
    API_KEY = API_KEY.strip()

if not API_KEY:
    print("\n" + "="*60)
    print("CRITICAL ERROR: FLUX_API_KEY environment variable is not set.")
    print("="*60)
    print("For security, this server now requires an API key to be set.")
    print("\nPlease set it before running the server:")
    print("  export FLUX_API_KEY=your_secret_key")
    print("\nOr create a .env file with:")
    print("  FLUX_API_KEY=your_secret_key")
    print("="*60 + "\n")
    import sys
    sys.exit(1)

def check_auth():
    # Check header or query param
    provided_key = request.headers.get("X-API-Key") or request.args.get("api_key")
    if not provided_key:
        return False
    # Constant-time compare to avoid a timing side-channel on the key
    return hmac.compare_digest(provided_key, API_KEY)

# Endpoints reachable without an API key. The main page, images, and the
# readiness probe: images are served with random filenames which provides basic
# security; /ready must be reachable before the user can enter their API key.
# rest_api.PUBLIC_ENDPOINTS adds the REST layer's equivalents (its /health and
# self-describing documents) for the same reasons.
PUBLIC_ENDPOINTS = ['index', 'alternate', 'static', 'serve_image', 'ready']


@app.before_request
def require_auth():
    if request.endpoint in PUBLIC_ENDPOINTS:
        return

    if not check_auth():
        # The REST layer speaks a different error dialect than the legacy
        # routes, so 401 has to be rendered in whichever one the caller used.
        if request.path.startswith(rest_api.URL_PREFIX):
            return jsonify({"error": {
                "code": "unauthorized",
                "message": "Provide a valid X-API-Key header or api_key parameter.",
            }}), 401
        return jsonify({"success": False, "error": "Unauthorized. Please provide a valid X-API-Key header or api_key parameter."}), 401


# Model-loading readiness state (set by the background loader in main()).
_model_ready = False
_model_load_error = None
_model_load_start_ts = 0.0
_model_load_status = "starting"

# Will be set by command-line args
_local_encoder = False
_full_model = False
_gguf_quant = None
_flux2 = False
_schnell = False
_turbo = False
_uncensored = False
_klein = False
_klein_4b = False
_kontext = False

# Configuration
OUTPUT_DIR = "web-generated"
PORT = 2222

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Hidden mode. A generation made with `hidden` set writes everything it
# produces — images, .prompt sidecars, composites, previews, saved step frames
# — into this dot-subdir of OUTPUT_DIR instead of alongside the rest. That
# keeps it out of every listing in this file (they all iterate top-level files
# only), out of the reference-image folder browser (which skips dotfiles), and
# out of image_manager.py's gallery, so the output exists on disk without
# appearing anywhere by default. Concealment, not security: anyone holding the
# API key can ask for the hidden listing, and the files are plain PNGs.
HIDDEN_DIR_NAME = ".hidden"
HIDDEN_PREFIX = HIDDEN_DIR_NAME + "/"
# The web UI sends this header on every request while its hidden toggle is on,
# so one switch in the browser puts generate/status/history/archive/delete in
# the same mode without threading a flag through each of them. An explicit
# `hidden` field in a request body or query string still wins over it.
HIDDEN_HEADER = "X-Flux-Hidden"
_TRUTHY = ('1', 'true', 'yes', 'on')


def _output_dir(hidden=False):
    """Where a job's artifacts get written: OUTPUT_DIR, or its hidden subdir."""
    if not hidden:
        return OUTPUT_DIR
    path = os.path.join(OUTPUT_DIR, HIDDEN_DIR_NAME)
    os.makedirs(path, exist_ok=True)
    return path


def _output_name(filename, hidden=False):
    """The name a hidden artifact is known by outside the server.

    Hidden output carries the `.hidden/` prefix in every filename it reports,
    so /images/, delete, save and use-as-reference keep working off a single
    identifier instead of needing a parallel flag beside every filename.
    """
    return (HIDDEN_PREFIX + filename) if hidden else filename


def _resolve_output_name(name):
    """Validate a client-supplied output filename and return its full path.

    A leading `.hidden/` is the only path component accepted; everything after
    it must be a bare filename, which is what keeps this inside OUTPUT_DIR.
    """
    if not name or not isinstance(name, str) or '\\' in name or not name.endswith('.png'):
        raise ApiError('Invalid filename', 400, 'invalid_request')
    bare = name[len(HIDDEN_PREFIX):] if name.startswith(HIDDEN_PREFIX) else name
    if '/' in bare or bare.startswith('.'):
        raise ApiError('Invalid filename', 400, 'invalid_request')
    return os.path.join(OUTPUT_DIR, name)


def _hidden_requested(explicit=None):
    """Whether this request is in hidden mode.

    An explicit `hidden` value from the caller decides; otherwise the UI's
    session header does. Outside a request (the queue worker advancing a
    multi-model run) there is no header, so the answer is no.
    """
    if explicit is not None:
        return bool(explicit) and str(explicit).strip().lower() not in ('0', 'false', 'no', '')
    if has_request_context():
        return request.headers.get(HIDDEN_HEADER, '').strip().lower() in _TRUTHY
    return False

# Queue configuration
QUEUE_MAX_SIZE = 10
RECENT_DONE_MAX = 10
# How many finished images the queue view rolls up as "just generated".
RECENT_IMAGES_MAX = 5

# Server configuration menu, mirroring run_server.sh's case statement (and
# SERVER_OPTIONS.md — keep all three in sync). The launcher exports the active
# number as FLUX_CONFIG; /switch-model writes the requested number to
# SWITCH_CONFIG_FILE and exits with SWITCH_EXIT_CODE, and the run_server.sh
# supervisor relaunches with the new config's flags.
SERVER_CONFIGS = {
    1: "FLUX.1 4-bit BNB",
    2: "FLUX.1 Full",
    3: "FLUX.1 GGUF Q8",
    4: "FLUX.1-schnell",
    5: "FLUX.1 + U-LoRA",
    6: "FLUX.2 4-bit BNB",
    7: "FLUX.2 Full + Turbo",
    8: "FLUX.2 Full (no Turbo)",
    9: "FLUX.2-klein-9B",
    10: "FLUX.1 Kontext (editor)",
    11: "FLUX.1 Kontext Full (editor, bf16)",
    12: "FLUX.1 Kontext Full + U-LoRA",
    13: "SDXL (photoreal)",
    14: "FLUX.2-klein-4B",
}
SWITCH_EXIT_CODE = 86
SWITCH_CONFIG_FILE = ".next_config"
try:
    _current_config = int(os.environ.get("FLUX_CONFIG", ""))
except ValueError:
    _current_config = None
if _current_config not in SERVER_CONFIGS:
    _current_config = None

# Edit-loop critique (vision model; see /critique). qwen3.6 is the strongest
# local VLM on this box — slower than gemma4:e2b but its judgments and revised
# prompts are markedly better.
#
# OLLAMA_URL keeps its name for compatibility but is really "the VLM endpoint":
# it may point at a local ollama daemon or at an OpenAI-compatible server
# (vLLM, llama.cpp, LM Studio) on another box, which is the better deal when
# the local card is busy holding the diffusion pipeline. edit_loop detects
# which dialect the endpoint speaks, and CRITIQUE_MODEL says which model to
# ask it for. The default, "auto", asks the endpoint what it is serving
# (edit_loop.resolve_vlm_model) instead of pinning a name here: the serving
# host gets restarted on a new checkpoint far more often than this config gets
# edited, and a stale pinned name 404s every VLM call. Name a model explicitly
# only to pin one out of several — e.g. when the endpoint serves both a vision
# and a text-only model.
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434")
CRITIQUE_MODEL = os.environ.get("CRITIQUE_MODEL", "auto").strip() or "auto"
# The reverse path (/describe) can use a different — typically Ollama Cloud —
# model, so image→prompt costs no local VRAM next to the resident FLUX
# pipeline. With OLLAMA_API_KEY set, cloud models (":cloud"/"-cloud" tags)
# are sent straight to ollama.com's hosted API (suffix stripped — hosted
# names don't carry it); without a key they go through the local daemon,
# which then needs a one-time `ollama signin`.
DESCRIBE_MODEL = os.environ.get("DESCRIBE_MODEL", "").strip() or CRITIQUE_MODEL
OLLAMA_API_KEY = os.environ.get("OLLAMA_API_KEY")
OLLAMA_CLOUD_URL = os.environ.get("OLLAMA_CLOUD_URL", "https://ollama.com")


def _ollama_call_params(model):
    """(model, url, api_key) for a VLM call: cloud models route to ollama.com
    when an API key is configured, everything else to the local daemon.

    Cloud routing keys off the ":cloud"/"-cloud" tag, so it only ever applies
    to an explicitly named model — "auto" is resolved later, against
    OLLAMA_URL, by the time anything knows what the name is."""
    if OLLAMA_API_KEY and re.search(r'[:-]cloud$', model):
        return re.sub(r'[:-]cloud$', '', model), OLLAMA_CLOUD_URL, OLLAMA_API_KEY
    return model, OLLAMA_URL, None

PREVIEW_FILENAME = "_preview_current.png"
PREVIEW_MIN_INTERVAL_S = 0.75  # throttle: skip decode if last preview was this recent

# GPU power draw for the UI's corner wattage badge. nvidia-smi takes ~100ms
# per call, so the reading is cached and refreshed at most every 2s even
# though /status is polled more often.
_POWER_CACHE_S = 2.0
_power_state = {"watts": None, "ts": 0.0}

# Whether the critique VLM is resident in ollama, for the UI's header badge.
# ollama's /api/ps is a cheap local call, but /status polls at up to 1.5s,
# so the answer is cached. _vlm_warming is set while the startup preload
# runs, so the badge can distinguish "loading" from "not loaded yet".
_VLM_CACHE_S = 10.0
_vlm_state = {"status": None, "model": None, "ts": 0.0}
_vlm_warming = False


def _probe_vlm():
    """("loaded"|"unloaded"|"unavailable", model name) for the critique model.

    ollama loads and unloads on demand, so residency is a real question and
    /api/ps answers it. An OpenAI-compatible server (vLLM et al.) instead
    serves a fixed model list for its lifetime: if /v1/models lists the model
    it is resident by definition, and there is no "unloaded" state to report.

    This is also where model auto-discovery refreshes. It runs at most once
    per _VLM_CACHE_S, so a serving host restarted on a different checkpoint is
    picked up within one poll — and the name comes back with the status, so
    the UI badge shows the model actually in use rather than the word "auto".
    """
    import edit_loop
    base = OLLAMA_URL.rstrip('/')
    try:
        models = edit_loop.vlm_endpoint_models(base, refresh=True)
        # Reads the list just fetched; no second round trip.
        model = edit_loop.resolve_vlm_model(base, CRITIQUE_MODEL)
        if not models["served"]:
            return "unavailable", model
        if edit_loop.vlm_dialect(base) == 'ollama':
            norm = lambda n: n if ":" in n else n + ":latest"
            resident = [norm(n) for n in models["resident"]]
            return ("loaded" if norm(model) in resident else "unloaded"), model
        return ("loaded" if model in models["served"] else "unloaded"), model
    except Exception:
        return "unavailable", CRITIQUE_MODEL


def _vlm_status():
    now = time.monotonic()
    if now - _vlm_state["ts"] < _VLM_CACHE_S:
        status, model = _vlm_state["status"], _vlm_state["model"]
    else:
        _vlm_state["ts"] = now
        status, model = _probe_vlm()
        _vlm_state["status"], _vlm_state["model"] = status, model
    if status != "loaded" and _vlm_warming:
        status = "loading"
    return {"model": model or CRITIQUE_MODEL, "status": status}


def _gpu_power_watts():
    now = time.monotonic()
    if now - _power_state["ts"] < _POWER_CACHE_S:
        return _power_state["watts"]
    _power_state["ts"] = now
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=power.draw", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=2)
        watts = sum(float(line) for line in out.stdout.split("\n")
                    if line.strip() and "N/A" not in line)
        _power_state["watts"] = round(watts, 1) if watts > 0 else None
    except (subprocess.SubprocessError, FileNotFoundError, ValueError):
        _power_state["watts"] = None
    return _power_state["watts"]


class JobCanceled(Exception):
    """Raised inside the generation loop when the user interrupts the running
    job; unwinds out of the diffusers pipeline back to the queue worker."""


class ApiError(Exception):
    """A request that failed for a reason the caller can act on.

    Raised by the `_api_*` core functions below, which hold the logic shared by
    the legacy flat routes and the /api/v1 REST layer (rest_api.py). Each layer
    catches this and renders it in its own envelope — the legacy
    `{"success": false, "error": ...}` shape, or REST's
    `{"error": {"code", "message"}}` — so neither dialect leaks into the other.
    `code` is the stable machine-readable identifier; keep it in sync with the
    error-code table in REST_API.md.
    """

    def __init__(self, message, status=400, code='invalid_request'):
        super().__init__(message)
        self.message = message
        self.status = status
        self.code = code


@dataclass
class Job:
    id: str
    params: dict
    state: str = 'queued'  # queued | running | done | failed | canceled
    cancel_requested: bool = False
    submitted_at: float = 0.0
    started_at: float = 0.0
    finished_at: float = 0.0
    error: Optional[str] = None
    current: int = 0
    batch: int = 0
    step: int = 0
    total_steps: int = 0
    images: list = field(default_factory=list)
    composite: Optional[str] = None
    preview: Optional[str] = None
    preview_step: int = 0
    preview_ts: int = 0
    saved_previews: int = 0  # preview frames written to steps/ (save_previews on)
    step_times: list = field(default_factory=list)  # per-step durations (s) of the current image
    generation_time: float = 0.0
    multi_run: Optional[str] = None  # id of the multi-model run this job belongs to
    # Set on every job of a `{a|b}` prompt: {'id', 'index', 'total', 'source'}.
    expansion: Optional[dict] = None
    # The group's contact sheet, filled in on all members once the last finishes.
    expansion_composite: Optional[str] = None

    @property
    def prompt(self) -> str:
        return (self.params.get('prompt') or '').strip()

    @property
    def hidden(self) -> bool:
        """True when this job's output belongs in OUTPUT_DIR/.hidden/ and the
        job itself should stay out of listings a non-hidden client asks for."""
        return bool(self.params.get('hidden', False))

    def summary(self) -> dict:
        p = self.prompt
        snippet = (p[:120] + '…') if len(p) > 120 else p
        return {
            'id': self.id,
            'state': self.state,
            'prompt': snippet,
            'submitted_at': self.submitted_at,
            'orientation': self.params.get('orientation'),
            'size': self.params.get('size'),
            'steps': self.params.get('steps'),
            'batch': self.params.get('batch', 1),
            'seed': self.params.get('seed'),
            'spectrum_grid': bool(self.params.get('spectrum_grid', False)),
            # Note: reference payloads are dropped from finished jobs, so this
            # is only meaningful while the job is queued/running.
            'refs': len(self.params.get('input_images') or []),
            'multi_run': self.multi_run,
            'expansion': self.expansion,
        }

    def full(self) -> dict:
        d = self.summary()
        d.update({
            'prompt': self.prompt,
            'started_at': self.started_at,
            'finished_at': self.finished_at,
            'current': self.current,
            'step': self.step,
            'total_steps': self.total_steps,
            # Copy to avoid the worker mutating this list while jsonify iterates it
            # after /status releases the queue lock.
            'images': list(self.images),
            'composite': self.composite,
            'expansion_composite': self.expansion_composite,
            'preview': self.preview,
            'preview_step': self.preview_step,
            'preview_ts': self.preview_ts,
            'saved_previews': self.saved_previews,
            'step_times': list(self.step_times),
            'generation_time': self.generation_time,
            'error': self.error,
            'cancel_requested': self.cancel_requested,
        })
        # summary's "batch" is the *requested* batch param; callers watching
        # progress want the actual total (may differ for spectrum grid).
        if self.batch:
            d['batch'] = self.batch
        return d


_queue_lock = threading.Lock()
_queue_cv = threading.Condition(_queue_lock)
_pending: list = []          # list[Job], front = next to run
_running_job: Optional[Job] = None
_recent_done: list = []      # list[Job], newest first, bounded by RECENT_DONE_MAX
_queue_worker_thread: Optional[threading.Thread] = None

# Lifetime counters for this process, for the status note (see _note_reporter).
# Not persisted: a model switch is a restart, so these are per-config totals,
# which is what you want when reading "how has this config been doing".
_stats = {
    'boot_ts': time.time(),
    'jobs_done': 0,
    'jobs_failed': 0,
    'jobs_canceled': 0,
    'images': 0,
    'last_error': None,
    'last_finish_ts': None,
}

# Orientation presets (width, height) at 1K base
ORIENTATIONS_1K = {
    'square': (1024, 1024),
    'portrait': (768, 1344),
    'landscape': (1360, 768),  # 16:9
    'widescreen': (1568, 672),  # ~21:9 extra-wide, ~1 MP
    'extra-tall': (672, 1568),  # ~9:21 mirror of widescreen, ~1 MP
}

SIZES = {
    '0.25mp': 0.25,
    '0.5mp': 0.5,
    '0.75mp': 0.75,
    '1mp': 1.0,
    '1.25mp': 1.25,
    '1.5mp': 1.5,
    '1.75mp': 1.75,
    '2mp': 2.0,
}

# The web UI lives in static/ (index.html + app.css + app.js); it was
# previously embedded here as one giant string. See static/README note.


def _composite_cell_size(width, height, cell_px=256):
    """Thumbnail size for one composite-grid cell, preserving aspect ratio."""
    aspect_ratio = width / height
    if width >= height:
        cell_width = cell_px
        cell_height = int(round(cell_width / aspect_ratio))
    else:
        cell_height = cell_px
        cell_width = int(round(cell_height * aspect_ratio))
    return cell_width, cell_height


_FONT_CACHE = {}
LABEL_MIN_PX = 11  # below this a caption is unreadable; truncate instead


def _composite_font(size):
    """A bold TrueType face at `size` — whichever one this box has. Falls back
    to Pillow's bitmap default (fixed size on Pillow < 10.1, hence the
    truncation path in _fit_label)."""
    if size in _FONT_CACHE:
        return _FONT_CACHE[size]
    font = None
    for font_path in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        "C:\\Windows\\Fonts\\segoeuib.ttf",
        "C:\\Windows\\Fonts\\arialbd.ttf",
    ):
        if os.path.exists(font_path):
            try:
                font = ImageFont.truetype(font_path, size)
                break
            except Exception:
                font = None
    if font is None:
        try:
            font = ImageFont.load_default(size=size)  # Pillow >= 10.1
        except TypeError:
            font = ImageFont.load_default()
    _FONT_CACHE[size] = font
    return font


def _fit_label(draw, text, max_px, max_width):
    """Return (text, font) for a label that fits `max_width`.

    Shrinks the face from `max_px` down to LABEL_MIN_PX, and only then clips the
    text. Sequence numbers always fit at full size; model names — the reason
    this exists — usually land a few points smaller."""
    size = max_px
    while size > LABEL_MIN_PX:
        font = _composite_font(size)
        if draw.textlength(text, font=font) <= max_width:
            return text, font
        size -= 1
    font = _composite_font(LABEL_MIN_PX)
    if draw.textlength(text, font=font) <= max_width:
        return text, font
    while len(text) > 1 and draw.textlength(text + '…', font=font) > max_width:
        text = text[:-1]
    return text + '…', font


def _build_composite_grid(grid_cells, cell_width, cell_height, label_height=0):
    """Paste a 2D list of (image, label)/None cells into one labeled composite.

    With `label_height` the label gets its own strip under each cell, centered
    and sized to fit — for text labels (model names) that would otherwise cover
    the image or run off the edge. Without it the label is a small badge in the
    cell's top-left corner, which is all a sequence number needs."""
    n_rows, n_cols = len(grid_cells), len(grid_cells[0])
    row_height = cell_height + label_height
    composite = Image.new('RGB', (n_cols * cell_width, n_rows * row_height), (32, 32, 32))
    for row_idx, row_images in enumerate(grid_cells):
        for col_idx, cell in enumerate(row_images):
            if cell is None:
                continue
            img, _label = cell
            img_small = img if img.size == (cell_width, cell_height) else \
                img.resize((cell_width, cell_height), Image.Resampling.LANCZOS)
            composite.paste(img_small, (col_idx * cell_width, row_idx * row_height))

    draw = ImageDraw.Draw(composite)
    badge_px = max(14, cell_height // 12)
    caption_px = max(LABEL_MIN_PX, int(label_height * 0.62))
    for row_idx, row_images in enumerate(grid_cells):
        for col_idx, cell in enumerate(row_images):
            if cell is None:
                continue
            _img, label = cell
            label = str(label)
            if label_height:
                text, font = _fit_label(draw, label, caption_px, cell_width - 12)
                bbox = draw.textbbox((0, 0), text, font=font)
                tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
                strip_y = row_idx * row_height + cell_height
                draw.rectangle(
                    [col_idx * cell_width, strip_y,
                     (col_idx + 1) * cell_width - 1, strip_y + label_height - 1],
                    fill=(0, 0, 0),
                )
                draw.text((col_idx * cell_width + (cell_width - tw) // 2 - bbox[0],
                           strip_y + (label_height - th) // 2 - bbox[1]),
                          text, fill=(255, 255, 255), font=font)
                continue
            text, font = _fit_label(draw, label, badge_px, cell_width - 20)
            pad = 4
            bbox = draw.textbbox((0, 0), text, font=font)
            tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
            x0 = col_idx * cell_width + 6
            y0 = row_idx * row_height + 6
            draw.rectangle(
                [x0 - pad, y0 - pad, x0 + tw + pad, y0 + th + pad],
                fill=(0, 0, 0),
            )
            draw.text((x0 - bbox[0], y0 - bbox[1]), text, fill=(255, 255, 255), font=font)
    return composite


# ------------------------------------------------------- expansion composites --
# A `{a|b}` prompt becomes one job per alternative, and the whole point is to
# compare them — so once the last member of a group finishes, its images are
# tiled into a contact sheet alongside the individual PNGs. The members run as
# separate jobs on the single worker, so the group is tracked here rather than
# in any one job, and the sheet is built by re-reading the saved files (holding
# every full-size image in memory until the group ends would be far worse).

_expansion_lock = threading.RLock()
_expansions = {}  # expansion id -> group state, dropped once the sheet is built
EXPANSION_GROUP_TTL = 6 * 3600  # abandoned groups (restart mid-run) expire


def _expansion_register(exp_id, total, source_prompt, hidden=False):
    with _expansion_lock:
        # A supervised restart strands whatever was in flight; don't let those
        # groups accumulate for the life of the process.
        cutoff = time.time() - EXPANSION_GROUP_TTL
        for dead in [k for k, g in _expansions.items() if g['created'] < cutoff]:
            del _expansions[dead]
        _expansions[exp_id] = {'total': total, 'done': 0, 'cells': [],
                               'prompt': source_prompt, 'created': time.time(),
                               'hidden': hidden}


def _expansion_record(job):
    """Note that an expansion member finished; build the sheet on the last one.

    Every member reaches this exactly once — done, failed, or canceled — so the
    group always completes even when part of the expansion never generated.
    Must not be called while holding _queue_cv: building the sheet does disk
    I/O.
    """
    if not job or not job.expansion:
        return
    with _expansion_lock:
        group = _expansions.get(job.expansion['id'])
        if not group:
            return
        group['done'] += 1
        if job.state == 'done' and job.images:
            # One cell per alternative: a job with batch > 1 already gets its
            # own batch grid, so take its first image as the representative.
            group['cells'].append((job.expansion['index'], job.images[0]['filename'],
                                   job.prompt))
        if group['done'] < group['total']:
            return
        del _expansions[job.expansion['id']]

    filename = _build_expansion_composite(group)
    if not filename:
        return
    # Publish the sheet on every member still in _recent_done, so a client that
    # polls any job of the group finds it.
    with _queue_cv:
        for j in _recent_done:
            if j.expansion and j.expansion['id'] == job.expansion['id']:
                j.expansion_composite = filename


def _build_expansion_composite(group):
    """Tile a finished group's images into one labeled sheet. Returns its
    filename, or None when there is nothing worth comparing."""
    cells = sorted(group['cells'])
    # A single surviving image is just that image; a sheet of one says nothing.
    if len(cells) < 2:
        return None

    thumbs, cell_width, cell_height = [], None, None
    for seq, (_index, filename, _text) in enumerate(cells, start=1):
        try:
            with Image.open(os.path.join(OUTPUT_DIR, filename)) as im:
                im = im.convert('RGB')
                if cell_width is None:
                    cell_width, cell_height = _composite_cell_size(*im.size)
                thumbs.append((im.resize((cell_width, cell_height),
                                         Image.Resampling.LANCZOS), seq))
        except OSError as e:
            # A missing or unreadable member costs one cell, not the sheet.
            print(f"[expansion] skipping {filename}: {e}", flush=True)
    if len(thumbs) < 2:
        return None

    n_cols = math.ceil(math.sqrt(len(thumbs)))
    n_rows = math.ceil(len(thumbs) / n_cols)
    grid_cells = [
        thumbs[r * n_cols:(r + 1) * n_cols] +
        [None] * (n_cols - len(thumbs[r * n_cols:(r + 1) * n_cols]))
        for r in range(n_rows)
    ]
    composite = _build_composite_grid(grid_cells, cell_width, cell_height)

    stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    hidden = bool(group.get('hidden'))
    comp_filename = f"{_output_prefix()}_{stamp}_expansion_grid.png"
    comp_path = os.path.join(_output_dir(hidden), comp_filename)
    composite.save(comp_path)

    # Sidecar: `# Prompt:` stays first-class for image_manager.py, with the
    # cell numbering spelled out so the sheet can be read on its own.
    with open(comp_path.rsplit('.', 1)[0] + '.prompt', 'w') as f:
        f.write(f"# Raw input: {group['prompt']}\n")
        f.write(f"# Prompt: {group['prompt']}\n")
        f.write(f"# Expansion: {len(thumbs)} of {group['total']} prompts\n")
        for seq, (_index, filename, text) in enumerate(cells, start=1):
            f.write(f"#   {seq}. {text} [{filename}]\n")
    print(f"[expansion] wrote {comp_filename} ({len(thumbs)} cells)", flush=True)
    return _output_name(comp_filename, hidden)


def _run_job(job: Job):
    """Execute one generation job, writing progress/results into the Job object."""
    data = job.params
    # Hidden mode: everything this job writes lands in OUTPUT_DIR/.hidden/, and
    # every name it reports back carries the prefix, so the client can still
    # fetch, reuse and delete its output by filename.
    hidden = job.hidden
    out_dir = _output_dir(hidden)
    prompt = (data.get('prompt') or '').strip()
    orientation = data.get('orientation', 'landscape')
    size = data.get('size', '1mp')
    steps = int(data.get('steps', 25))
    seed = data.get('seed')
    guidance_scale = data.get('guidance')
    batch = min(max(int(data.get('batch', 1)), 1), 128)

    # negative_prompt is SDXL-only (validated at the boundary); pass it as an
    # extra kwarg only on cores that take it, so flux_core's signature is
    # untouched.
    _neg_kwargs = {}
    if getattr(flux_core, 'SUPPORTS_NEGATIVE_PROMPT', False):
        _neg_kwargs['negative_prompt'] = (data.get('negative_prompt') or '').strip() or None

    # Reference images (validated and normalized to `input_images` by /generate).
    # The first is the primary — it drives output dimensions and inpainting;
    # generate_image handles the multi-reference semantics per model family.
    strength = float(data.get('strength', 0.5))
    input_images = []
    for b64 in data.get('input_images') or []:
        if ',' in b64:
            b64 = b64.split(',', 1)[1]
        input_images.append(Image.open(io.BytesIO(base64.b64decode(b64))).convert('RGB'))
    input_image = input_images[0] if input_images else None
    gen_input = input_images if len(input_images) > 1 else input_image

    # Handle inpaint mask (white = regenerate). FLUX.2 or SDXL; needs an input image.
    mask_image = None
    mask_b64 = data.get('mask_image')
    _inpaint_ok = flux_core._flux_version == 2 or getattr(flux_core, 'SUPPORTS_INPAINT', False)
    if mask_b64 and input_image is not None and _inpaint_ok:
        if ',' in mask_b64:
            mask_b64 = mask_b64.split(',', 1)[1]
        mask_image = Image.open(io.BytesIO(base64.b64decode(mask_b64))).convert('L')

    # Dimensions
    scale = SIZES.get(size, 1.0)
    aspect_mode = data.get('aspect_mode', 'keep')

    if input_image is not None and aspect_mode == 'keep':
        in_w, in_h = input_image.size
        target_pixels = 1_000_000 * (scale ** 2)
        current_pixels = in_w * in_h
        factor = (target_pixels / current_pixels) ** 0.5
        width = int(round(in_w * factor / 8) * 8)
        height = int(round(in_h * factor / 8) * 8)
    else:
        base_w, base_h = ORIENTATIONS_1K.get(orientation, ORIENTATIONS_1K['landscape'])
        width, height = int(base_w * scale), int(base_h * scale)

    # Inpainting regenerates only the painted region of the input image, so the
    # spectrum grid (which sweeps strength/guidance) doesn't apply. Latent tokens
    # cover 16px blocks, so dimensions must be multiples of 16 for mask alignment.
    if mask_image is not None:
        data['spectrum_grid'] = False
        width = (width // 16) * 16
        height = (height // 16) * 16

    spectrum_grid = data.get('spectrum_grid', False)
    spectrum_same_seed = data.get('spectrum_same_seed', True)
    selected_cells = data.get('selected_cells', []) # Indices 0-15

    if spectrum_grid:
        guidance_values = [0] if _schnell else [1.0, 3.0, 5.0, 7.0]
        strength_values = [0.2, 0.4, 0.6, 0.8] if input_image else [0.0, 0.0, 0.0, 0.0] # Dummy if no image

        # The UI selector is always a 4x4 grid (guidance columns x strength rows),
        # but schnell has a single guidance column (it ignores guidance). Collapse
        # 4x4 cell indices onto column 0 of their row so the generation loop's
        # cell_idx (r_idx * 4 + c_idx with c_idx == 0) can actually match them.
        if _schnell and selected_cells:
            selected_cells = sorted({(idx // 4) * 4 for idx in selected_cells})

        if selected_cells:
            total_batch = len(selected_cells)
        elif input_image and not _schnell:
            # Fallback to diagonals if nothing selected but somehow grid is on
            total_batch = 8
        else:
            total_batch = len(strength_values) * len(guidance_values)
    else:
        total_batch = batch

    job.batch = total_batch
    job.total_steps = steps

    # Encode the prompt once for the whole job: every image in a batch shares
    # the prompt, and re-encoding costs a full text-encoder forward per image
    # (the FLUX.2 encoders are LLM-sized — seconds each). None means the
    # config pre-encoding doesn't apply to (remote encoder) or it failed;
    # generate_image then encodes per-call as before.
    prompt_embeds_kwargs = None
    if total_batch > 1:
        t_enc = time.perf_counter()
        prompt_embeds_kwargs = flux_core.encode_prompt_once(prompt)
        if prompt_embeds_kwargs is not None:
            print(f"[queue] prompt encoded once for {total_batch} images "
                  f"in {time.perf_counter() - t_enc:.2f}s", flush=True)

    show_preview = bool(data.get('show_preview', False))
    # Keep every preview frame as its own file under steps/ (a subdir, so the
    # frames stay out of gallery listings, archiving, and delete-today, which
    # only iterate top-level files). Requires show_preview: the frames ARE the
    # preview decodes.
    save_previews = show_preview and bool(data.get('save_previews', False))
    steps_dir = os.path.join(out_dir, 'steps')
    preview_state = {"last_decode": 0.0}

    def _check_cancel():
        if job.cancel_requested:
            raise JobCanceled()

    def _step_callback(pipe_obj, step_index, timestep, callback_kwargs):
        # Interrupt point: raising here unwinds out of the denoising loop
        # mid-generation (the worker catches JobCanceled).
        _check_cancel()
        # The scheduler holds the *actual* timesteps for this run. For img2img the
        # pipeline only denoises ~steps*strength of them (and turbo/schnell clamp
        # the count too), so the requested `steps` overstates the work. Read the
        # real total from the scheduler the first chance we get so the progress
        # bar and "step X of Y" reflect what's actually happening.
        try:
            ts = getattr(getattr(pipe_obj, "scheduler", None), "timesteps", None)
            if ts is not None and len(ts) > 0:
                job.total_steps = len(ts)
        except Exception:
            pass
        job.step = step_index + 1
        if show_preview:
            now = time.perf_counter()
            is_final = (step_index + 1) >= job.total_steps
            # save_previews bypasses the decode throttle: "each frame" means
            # every step, not just the ones the 0.75s interval lets through.
            if is_final or save_previews or (now - preview_state["last_decode"]) >= PREVIEW_MIN_INTERVAL_S:
                latents = callback_kwargs.get("latents")
                preview_img = flux_core.decode_latents_to_preview(
                    pipe_obj, latents, height, width
                )
                if preview_img is not None:
                    try:
                        preview_img.save(os.path.join(out_dir, PREVIEW_FILENAME))
                        job.preview = _output_name(PREVIEW_FILENAME, hidden)
                        job.preview_step = step_index + 1
                        job.preview_ts = int(time.time() * 1000)
                        preview_state["last_decode"] = now
                    except Exception as e:
                        print(f"[preview] save failed: {e}", flush=True)
                    if save_previews:
                        try:
                            os.makedirs(steps_dir, exist_ok=True)
                            frame = f"{job.id}_img{max(job.current, 1):02d}_step{step_index + 1:03d}.png"
                            preview_img.save(os.path.join(steps_dir, frame))
                            job.saved_previews += 1
                        except Exception as e:
                            print(f"[preview] frame save failed: {e}", flush=True)
        return callback_kwargs

    start_time = time.perf_counter()

    if spectrum_grid:
        grid_seed = seed if seed is not None else random.randint(0, 2**32 - 1)
        grid_cells = []

        # We still want to build a full 4x4 grid for the composite, but only generate selected
        guidance_values = [0] if _schnell else [1.0, 3.0, 5.0, 7.0]
        strength_values = [0.2, 0.4, 0.6, 0.8] if input_image else [0.0, 0.2, 0.4, 0.6] # Use some defaults if no image for grid

        generated_count = 0
        for r_idx, s_val in enumerate(strength_values):
            row_images = []
            for c_idx, g_val in enumerate(guidance_values):
                cell_idx = r_idx * 4 + c_idx

                # Check if this cell should be generated
                should_gen = False
                if selected_cells:
                    should_gen = cell_idx in selected_cells
                elif input_image and not _schnell:
                    # Legacy diagonal logic
                    is_main_diag = (r_idx == c_idx)
                    is_anti_diag = (r_idx == len(guidance_values) - 1 - c_idx)
                    should_gen = is_main_diag or is_anti_diag
                else:
                    should_gen = True

                if not should_gen:
                    row_images.append(None)
                    continue

                generated_count += 1
                job.current = generated_count
                job.step = 0
                _check_cancel()
                current_seed = grid_seed if spectrum_same_seed else random.randint(0, 2**32 - 1)

                image, used_seed, timings = generate_image(
                    prompt, seed=current_seed, steps=steps, width=width, height=height,
                    local_encoder=_local_encoder, input_image=gen_input,
                    strength=(s_val if input_image else 0.5),
                    guidance_scale=g_val,
                    callback_on_step_end=_step_callback,
                    prompt_embeds_kwargs=prompt_embeds_kwargs,
                    **_neg_kwargs
                )
                t_save = time.perf_counter()
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                unique_id = uuid.uuid4().hex[:8]
                g_str = str(g_val).replace('.', '_')
                s_str = f"str_{s_val}" if s_val is not None else "txt2img"
                output_filename = f"{_output_prefix()}_{timestamp}_g{g_str}_{s_str}_{unique_id}.png"
                output_path = os.path.join(out_dir, output_filename)
                image.save(output_path)
                timings['save'] = time.perf_counter() - t_save
                save_prompt_file(output_path, prompt, prompt, width, height, used_seed, steps, timings, g_val, s_val if input_image else None)

                img_data = {
                    "filename": _output_name(output_filename, hidden),
                    "seed": used_seed,
                    "guidance": g_val,
                    "strength": s_val,
                    "timings": {
                        'encoding': round(timings['encoding'], 2),
                        'diffusion': round(timings['diffusion'], 2),
                        'save': round(timings['save'], 2),
                        'total': round(timings['encoding'] + timings['diffusion'] + timings['save'], 2)
                    }
                }
                job.images.append(img_data)
                row_images.append((image.copy(), generated_count))
            grid_cells.append(row_images)

        # Create composite (sequence numbers overlaid on each populated cell)
        cell_width, cell_height = _composite_cell_size(width, height)
        composite = _build_composite_grid(grid_cells, cell_width, cell_height)

        comp_filename = f"{_output_prefix()}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_spectrum_grid.png"
        composite.save(os.path.join(out_dir, comp_filename))
        job.composite = _output_name(comp_filename, hidden)
    else:
        # For batch > 1, collect a small thumbnail per image (not the full-res
        # frame, to avoid holding many full images in memory at once) so a
        # composite matrix can be built once the batch finishes.
        cell_width = cell_height = None
        batch_thumbs = []
        if batch > 1:
            cell_width, cell_height = _composite_cell_size(width, height)

        for i in range(batch):
            job.current = i + 1
            job.step = 0
            _check_cancel()
            current_seed = (seed + i) if seed is not None else None
            image, used_seed, timings = generate_image(
                prompt, seed=current_seed, steps=steps, width=width, height=height,
                local_encoder=_local_encoder, input_image=gen_input, strength=strength,
                guidance_scale=guidance_scale, mask_image=mask_image,
                callback_on_step_end=_step_callback,
                prompt_embeds_kwargs=prompt_embeds_kwargs,
                **_neg_kwargs
            )

            t_save = time.perf_counter()
            output_filename = f"{_output_prefix()}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}.png"
            output_path = os.path.join(out_dir, output_filename)
            image.save(output_path)
            timings['save'] = time.perf_counter() - t_save
            save_prompt_file(output_path, prompt, prompt, width, height, used_seed, steps, timings, guidance_scale, None if mask_image is not None else (strength if input_image else None))

            img_data = {
                'filename': _output_name(output_filename, hidden),
                'seed': used_seed,
                'guidance': guidance_scale,
                'strength': None if mask_image is not None else (strength if input_image else None),
                'inpaint': mask_image is not None,
                'timings': {
                    'encoding': round(timings['encoding'], 2),
                    'diffusion': round(timings['diffusion'], 2),
                    'save': round(timings['save'], 2),
                    'total': round(timings['encoding'] + timings['diffusion'] + timings['save'], 2)
                }
            }
            job.images.append(img_data)

            if batch > 1:
                thumb = image.resize((cell_width, cell_height), Image.Resampling.LANCZOS)
                batch_thumbs.append((thumb, i + 1))

        if batch > 1:
            n_cols = math.ceil(math.sqrt(batch))
            n_rows = math.ceil(batch / n_cols)
            grid_cells = [
                batch_thumbs[r * n_cols:(r + 1) * n_cols] + [None] * (n_cols - len(batch_thumbs[r * n_cols:(r + 1) * n_cols]))
                for r in range(n_rows)
            ]
            composite = _build_composite_grid(grid_cells, cell_width, cell_height)
            comp_filename = f"{_output_prefix()}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_batch_grid.png"
            composite.save(os.path.join(out_dir, comp_filename))
            job.composite = _output_name(comp_filename, hidden)

    job.generation_time = time.perf_counter() - start_time


def _queue_worker():
    """Single long-running thread that consumes queued jobs one at a time."""
    global _running_job
    while True:
        with _queue_cv:
            while not _pending:
                _queue_cv.wait()
            job = _pending.pop(0)
            _running_job = job
        job.state = 'running'
        job.started_at = time.time()
        try:
            _run_job(job)
            job.state = 'done'
        except JobCanceled:
            print(f"[queue] job {job.id} interrupted by user", flush=True)
            job.state = 'canceled'
        except Exception as e:
            print(f"[queue] job {job.id} failed: {e}", flush=True)
            job.state = 'failed'
            job.error = str(e)
        finally:
            job.finished_at = time.time()
            # Counters for the status note. Images are counted from the job's
            # own list rather than its requested batch, so a job that died
            # halfway still contributes the images it actually wrote.
            _stats['images'] += len(job.images)
            _stats['last_finish_ts'] = job.finished_at
            if job.state == 'done':
                _stats['jobs_done'] += 1
            elif job.state == 'canceled':
                _stats['jobs_canceled'] += 1
            else:
                _stats['jobs_failed'] += 1
                _stats['last_error'] = job.error
            # Finished jobs sit in _recent_done for a while; drop the (large)
            # base64 image payloads so they don't stay resident in memory.
            job.params.pop('input_image', None)
            job.params.pop('input_images', None)
            job.params.pop('mask_image', None)
            with _queue_cv:
                _running_job = None
                _recent_done.insert(0, job)
                del _recent_done[RECENT_DONE_MAX:]
            # Multi-model run bookkeeping: record this job's results, then
            # advance the run (queue its next job, restart into the next
            # config once the queue is idle, or mark it finished).
            if job.multi_run:
                _multi_run_record(job)
            # Tiles the group's images once this is the last member to finish.
            _expansion_record(job)
            _multi_run_advance()


def _start_queue_worker():
    global _queue_worker_thread
    if _queue_worker_thread is None or not _queue_worker_thread.is_alive():
        _queue_worker_thread = threading.Thread(target=_queue_worker, daemon=True, name="queue-worker")
        _queue_worker_thread.start()


# ---- Status note ----
# A small note board on this box that the operator reads instead of tailing
# server.log. Three things about it shape the code below:
#
#   * It holds ONE note. A POST replaces whatever was there, so this is a
#     status line, not an append log — always post the full current picture.
#   * Text is capped at 500 characters server-side; a longer post is rejected
#     outright (the old note survives), so we build short and truncate.
#   * The board is shared and visible, so prompt text never goes in it. Only
#     counters, queue depth, config, and error strings.
#
# Set NOTE_URL=off to disable. The default target is local because that is the
# instance being watched; the LAN copy has its own network-stats reporter
# overwriting it every few minutes.
NOTE_URL = os.environ.get("NOTE_URL", "http://127.0.0.1:9999/note")
NOTE_INTERVAL_S = int(os.environ.get("NOTE_INTERVAL", "300"))
NOTE_MAX_CHARS = 500
_note_failed = False
# The port actually bound, which is PORT unless --port overrode it.
_note_port = PORT


def _fmt_duration(seconds):
    seconds = int(seconds)
    if seconds < 3600:
        return f"{seconds // 60}m"
    if seconds < 86400:
        return f"{seconds // 3600}h{(seconds % 3600) // 60:02d}m"
    return f"{seconds // 86400}d{(seconds % 86400) // 3600}h"


def _note_text():
    """The status line to post. Must stay under NOTE_MAX_CHARS and must not
    contain prompt text."""
    config = SERVER_CONFIGS.get(_current_config, 'unknown config')
    if _model_load_error:
        model = f"MODEL LOAD FAILED: {_model_load_error}"
    elif not _model_ready:
        model = _model_load_status
    else:
        model = 'ready'
    with _queue_cv:
        queued, running = len(_pending), _running_job is not None
    lines = [
        f"flux2 :{_note_port} · {config} · up {_fmt_duration(time.time() - _stats['boot_ts'])}",
        f"model: {model}",
        f"images {_stats['images']} · jobs {_stats['jobs_done']} ok"
        f" / {_stats['jobs_failed']} fail / {_stats['jobs_canceled']} cancel",
        f"queue: {queued} waiting, {'1 running' if running else 'idle'}",
    ]
    if _stats['last_finish_ts']:
        lines.append("last finish "
                     f"{_fmt_duration(time.time() - _stats['last_finish_ts'])} ago")
    if _stats['last_error']:
        lines.append(f"last err: {_stats['last_error']}")
    text = "\n".join(lines) + f"\n@ {datetime.now().strftime('%H:%M')}"
    return text[:NOTE_MAX_CHARS]


def _post_note(text):
    """POST the note, returning True on success. Failures are non-fatal — the
    board is a convenience, and the server must not care whether it is up."""
    global _note_failed
    import urllib.request
    req = urllib.request.Request(
        NOTE_URL, data=json.dumps({'text': text}).encode(),
        headers={'Content-Type': 'application/json'})
    try:
        with urllib.request.urlopen(req, timeout=5) as r:
            r.read()
        _note_failed = False
        return True
    except Exception as e:
        # One line the first time it breaks, then quiet: this runs every few
        # minutes forever and must not fill server.log on a box where the
        # note service simply isn't running.
        if not _note_failed:
            print(f"[note] posting to {NOTE_URL} failed ({e}); "
                  "continuing quietly. Set NOTE_URL=off to disable.", flush=True)
            _note_failed = True
        return False


def _note_reporter():
    while True:
        _post_note(_note_text())
        time.sleep(NOTE_INTERVAL_S)


def _start_note_reporter(port=None):
    global _note_port
    if port:
        _note_port = port
    if NOTE_URL.lower() in ('off', 'none', '') or NOTE_INTERVAL_S <= 0:
        return
    threading.Thread(target=_note_reporter, daemon=True, name='note-reporter').start()


# ---- Multi-model runs ----
# One prompt generated on several run_server.sh configs in sequence. Each
# model switch is a full supervised restart (see /switch-model), so the run's
# state can't live in memory: it's a JSON file that survives the restarts.
# The cycle is: queue a job on the current model -> record its results in the
# file -> restart into the next config -> on ready, _multi_run_advance() picks
# the file back up and queues the next job.
MULTI_RUN_FILE = ".multi_run.json"
_multi_run_lock = threading.RLock()


def _multi_run_load():
    try:
        with open(MULTI_RUN_FILE) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def _multi_run_save(state):
    # Atomic write: the supervisor can kill the process at any point and a
    # half-written state file would strand the run.
    tmp = MULTI_RUN_FILE + ".tmp"
    with open(tmp, 'w') as f:
        json.dump(state, f)
    os.replace(tmp, MULTI_RUN_FILE)


def _multi_run_clear():
    try:
        os.remove(MULTI_RUN_FILE)
    except FileNotFoundError:
        pass


def _multi_run_next_config(state):
    done = {r['config'] for r in state['results']}
    for c in state['configs']:
        if c not in done:
            return c
    return None


MULTI_RUN_CELL_PX = 512  # bigger than the batch/expansion cells: model names have to be readable


def _build_multi_run_composite(state):
    """Tile a finished run's images into one sheet, each cell captioned with the
    model that produced it.

    The whole point of a multi-model run is the side-by-side, and the images are
    scattered across the run's separate jobs (and separate processes — every
    config change is a restart), so the sheet is built at the end from the files
    the run recorded. Returns its filename, or None when fewer than two models
    produced an image and there is nothing to compare."""
    cells = []
    for result in state.get('results', []):
        if result.get('state') != 'done' or not result.get('images'):
            continue
        # A batch job already gets its own grid; take its first image as the
        # model's representative, same as the expansion sheet does.
        cells.append((result['images'][0]['filename'],
                      result.get('label') or f"config {result.get('config')}"))
    if len(cells) < 2:
        return None

    thumbs, drawn, cell_width, cell_height = [], [], None, None
    for filename, label in cells:
        try:
            with Image.open(os.path.join(OUTPUT_DIR, filename)) as im:
                im = im.convert('RGB')
                if cell_width is None:
                    cell_width, cell_height = _composite_cell_size(
                        *im.size, cell_px=MULTI_RUN_CELL_PX)
                thumbs.append((im.resize((cell_width, cell_height),
                                         Image.Resampling.LANCZOS), label))
                drawn.append((filename, label))
        except OSError as e:
            # A missing or unreadable image costs one cell, not the sheet.
            print(f"[multi-run] skipping {filename}: {e}", flush=True)
    if len(thumbs) < 2:
        return None

    n_cols = math.ceil(math.sqrt(len(thumbs)))
    n_rows = math.ceil(len(thumbs) / n_cols)
    grid_cells = [
        thumbs[r * n_cols:(r + 1) * n_cols] +
        [None] * (n_cols - len(thumbs[r * n_cols:(r + 1) * n_cols]))
        for r in range(n_rows)
    ]
    composite = _build_composite_grid(grid_cells, cell_width, cell_height,
                                      label_height=max(26, cell_height // 10))

    stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    comp_filename = f"{_output_prefix()}_{stamp}_multi_run_grid.png"
    comp_path = os.path.join(OUTPUT_DIR, comp_filename)
    composite.save(comp_path)

    # Sidecar: `# Prompt:` stays first-class for image_manager.py, with the
    # per-cell model names so the sheet can be read on its own.
    with open(comp_path.rsplit('.', 1)[0] + '.prompt', 'w') as f:
        f.write(f"# Raw input: {state['prompt']}\n")
        f.write(f"# Prompt: {state['prompt']}\n")
        f.write(f"# Multi-model run: {len(thumbs)} of {len(state['configs'])} models\n")
        f.write(f"# Seed: {state['params'].get('seed')}\n")
        for seq, (filename, label) in enumerate(drawn, start=1):
            f.write(f"#   {seq}. {label} [{filename}]\n")
    print(f"[multi-run] wrote {comp_filename} ({len(thumbs)} cells)", flush=True)
    return comp_filename


def _multi_run_finish(state, canceled=False):
    """Mark a run finished and save it, building its comparison sheet first.

    Both endings come through here — every config done, or the run canceled
    partway — so a canceled run still gets a sheet of whatever it managed to
    generate. Caller holds _multi_run_lock."""
    state['finished'] = time.time()
    if canceled:
        state['canceled'] = True
    try:
        filename = _build_multi_run_composite(state)
    except Exception as e:
        # The sheet is a convenience; never lose the run's results over it.
        print(f"[multi-run] composite failed: {e}", flush=True)
        filename = None
    if filename:
        state['composite'] = filename
    _multi_run_save(state)


def _multi_run_enqueue(state):
    job = Job(id=uuid.uuid4().hex[:12], params=dict(state['params']), submitted_at=time.time())
    job.multi_run = state['id']
    with _queue_cv:
        _pending.append(job)
        _queue_cv.notify()
    print(f"[multi-run] queued job {job.id} on config {_current_config} "
          f"({SERVER_CONFIGS.get(_current_config)})", flush=True)


def _multi_run_switch(target):
    with open(SWITCH_CONFIG_FILE, 'w') as f:
        f.write(str(target))
    print(f"[multi-run] restarting into config {target} ({SERVER_CONFIGS[target]})", flush=True)
    # Small delay so any in-flight HTTP responses flush before the exit.
    threading.Timer(0.5, lambda: os._exit(SWITCH_EXIT_CODE)).start()


def _multi_run_record(job):
    """Append a finished multi-run job's results to the run's state file."""
    with _multi_run_lock:
        state = _multi_run_load()
        if not state or state.get('id') != job.multi_run or state.get('finished'):
            return
        state['results'].append({
            'config': _current_config,
            'label': SERVER_CONFIGS.get(_current_config, str(_current_config)),
            'model': _model_type_string(),
            'state': job.state,
            'error': job.error,
            'images': [{'filename': i['filename'], 'seed': i['seed']} for i in job.images],
            'generation_time': round(job.generation_time, 2),
        })
        if job.state == 'canceled':
            # Interrupting the run's job cancels the whole run — the sheet still
            # gets built from the models that did finish.
            _multi_run_finish(state, canceled=True)
        else:
            _multi_run_save(state)


def _multi_run_advance():
    """Drive the active multi-model run one step forward.

    Called when the model becomes ready (startup / after a switch) and after
    every finished job. Queues the run's job if the current config is next;
    otherwise restarts into the next config — but only once the queue is
    idle, so user-submitted jobs are never killed mid-generation.
    """
    if _current_config is None or not _model_ready:
        return
    with _multi_run_lock:
        state = _multi_run_load()
        if not state or state.get('finished'):
            return
        nxt = _multi_run_next_config(state)
        if nxt is None:
            _multi_run_finish(state)
            print(f"[multi-run] run {state['id']} complete", flush=True)
            return
        if nxt == _current_config:
            with _queue_cv:
                queued = any(j.multi_run == state['id'] for j in _pending) or \
                    (_running_job is not None and _running_job.multi_run == state['id'])
            if not queued:
                _multi_run_enqueue(state)
        else:
            with _queue_cv:
                idle = _running_job is None and not _pending
            if idle:
                _multi_run_switch(nxt)


@app.route('/')
def index():
    return app.send_static_file('index.html')


@app.route('/alternate')
def alternate():
    # Same functionality as '/', a denser sidebar+main dashboard layout
    # instead of one long scrolling form. Shares app.js/app.css unchanged;
    # see static/alternate.css for the layout-only overrides.
    return app.send_static_file('alternate.html')


@app.route('/ready')
def ready():
    return jsonify(_api_readiness())


# ---------------------------------------------------------- prompt expansion --
# `a {red|blue} car` queues one job per alternative — the cartesian product
# across every group, so `{red|blue} car in {rain|snow}` is four jobs. Groups
# nest, and a backslash escapes a brace or bar that is meant literally
# (`\{not a group\}`). A braced run with no top-level `|` is ordinary text, so
# prompts that merely contain braces are untouched.


def _split_alternatives(body):
    """Split a brace body on its top-level `|`, honoring nesting and escapes."""
    parts, depth, buf, i = [], 0, [], 0
    while i < len(body):
        c = body[i]
        if c == '\\' and i + 1 < len(body):
            buf.append(c)
            buf.append(body[i + 1])
            i += 2
            continue
        if c == '{':
            depth += 1
        elif c == '}':
            depth -= 1
        elif c == '|' and depth == 0:
            parts.append(''.join(buf))
            buf = []
            i += 1
            continue
        buf.append(c)
        i += 1
    parts.append(''.join(buf))
    return parts


def _find_alternation(prompt):
    """(start, end, body) of the first `{...}` holding a top-level `|`.

    `end` is exclusive. Returns None when the prompt has no alternation left.
    Braced runs without a top-level `|` are descended into rather than skipped,
    so the inner group of `{keep {a|b}}` still expands.
    """
    i = 0
    while i < len(prompt):
        if prompt[i] == '\\':
            i += 2
            continue
        if prompt[i] == '{':
            depth, j = 1, i + 1
            while j < len(prompt) and depth:
                if prompt[j] == '\\':
                    j += 2
                    continue
                if prompt[j] == '{':
                    depth += 1
                elif prompt[j] == '}':
                    depth -= 1
                j += 1
            # An unbalanced '{' is literal text; nothing after it can be a
            # group either, so stop looking.
            if depth:
                return None
            body = prompt[i + 1:j - 1]
            if len(_split_alternatives(body)) > 1:
                return i, j, body
        i += 1
    return None


def _finalize_prompt(text):
    """Drop the escaping backslashes and tidy the seams left by expansion."""
    out, i = [], 0
    while i < len(text):
        if text[i] == '\\' and i + 1 < len(text) and text[i + 1] in '{}|\\':
            out.append(text[i + 1])
            i += 2
            continue
        out.append(text[i])
        i += 1
    # An empty alternative (`a {big |}cat`) leaves a double space behind.
    # Collapse runs of spaces only — newlines in the prompt are deliberate.
    return re.sub(r'[ \t]{2,}', ' ', ''.join(out)).strip()


def expand_prompt(prompt, limit):
    """Every prompt `prompt` expands to, in `{a|b}` left-to-right order.

    Returns None if the product exceeds `limit`, so a pathological prompt is
    refused before its expansion is ever materialized. Duplicate results are
    dropped: with the queue only QUEUE_MAX_SIZE deep, a repeated alternative
    is a typo far more often than a request for the same image twice.
    """
    out, frontier = [], [prompt]
    while frontier:
        # Every unexpanded prompt still yields at least one result, so this
        # sum only grows — once it passes the limit the product cannot fit.
        if len(out) + len(frontier) > limit:
            return None
        current = frontier.pop(0)
        found = _find_alternation(current)
        if not found:
            out.append(_finalize_prompt(current))
            continue
        start, end, body = found
        head, tail = current[:start], current[end:]
        # Depth-first, so the leftmost group varies slowest and the queue
        # order reads the way the prompt does. Alternatives keep their own
        # spacing — `{big |}cat` needs that trailing space, and the padding in
        # `{red | blue}` is collapsed with the other seams at the end.
        frontier[:0] = [head + alt + tail
                        for alt in _split_alternatives(body)]
    return list(dict.fromkeys(out))


def _validate_generate_params(data):
    """Validate and normalize a /generate request body in place.

    Returns an error string for a 400 response, or None if the request is
    valid. Doing this at the API boundary means bad input fails fast with a
    clear message instead of surfacing later as an opaque failed job.
    """
    # Hidden mode is a property of the request, not of the model call: it only
    # decides where the output lands and who gets told about it. Normalizing it
    # here means both API dialects and the queue see one boolean.
    data['hidden'] = _hidden_requested(data.get('hidden'))

    try:
        data['steps'] = int(data.get('steps', 25))
    except (TypeError, ValueError):
        return "steps must be an integer"
    if not 1 <= data['steps'] <= 200:
        return "steps must be between 1 and 200"

    try:
        data['batch'] = int(data.get('batch', 1))
    except (TypeError, ValueError):
        return "batch must be an integer"
    if not 1 <= data['batch'] <= 128:
        return "batch must be between 1 and 128"

    try:
        data['strength'] = float(data.get('strength', 0.5))
    except (TypeError, ValueError):
        return "strength must be a number"
    if not 0.0 <= data['strength'] <= 1.0:
        return "strength must be between 0.0 and 1.0"

    if data.get('seed') is not None:
        try:
            data['seed'] = int(data['seed'])
        except (TypeError, ValueError):
            return "seed must be an integer"

    if data.get('guidance') is not None:
        try:
            data['guidance'] = float(data['guidance'])
        except (TypeError, ValueError):
            return "guidance must be a number"
        if data['guidance'] < 0:
            return "guidance must be non-negative"

    if data.get('negative_prompt'):
        if not isinstance(data['negative_prompt'], str):
            return "negative_prompt must be a string"
        if not getattr(flux_core, 'SUPPORTS_NEGATIVE_PROMPT', False):
            return "negative_prompt requires the SDXL backend (start the server with --sdxl)"

    if data.get('orientation') is not None and data['orientation'] not in ORIENTATIONS_1K:
        return f"orientation must be one of {sorted(ORIENTATIONS_1K)}"
    if data.get('size') is not None and data['size'] not in SIZES:
        return f"size must be one of {sorted(SIZES)}"

    def _decodable(b64):
        try:
            payload = b64.split(',', 1)[1] if ',' in b64 else b64
            Image.open(io.BytesIO(base64.b64decode(payload))).verify()
            return True
        except Exception:
            return False

    # Reference images: `input_images` (a list, up to MAX_REFERENCE_IMAGES) is
    # canonical; the legacy single `input_image` field is folded into it here so
    # the rest of the server only ever sees the list form. Decode images at the
    # boundary so a corrupt upload is a 400, not a failed job.
    imgs = data.get('input_images')
    if imgs is not None and not isinstance(imgs, list):
        return "input_images must be a list of base64 images"
    imgs = list(imgs or [])
    if not imgs and data.get('input_image'):
        imgs = [data['input_image']]
    for i, b64 in enumerate(imgs):
        if not isinstance(b64, str) or not _decodable(b64):
            return f"input_images[{i}] is not a decodable base64 image"
    # Server-side reference images: `input_paths` lists files already on this
    # machine (absolute, ~-prefixed, or relative to web-generated/). They are
    # loaded here and appended to input_images so the rest of the server only
    # ever sees the base64 list.
    paths = data.pop('input_paths', None)
    if paths is not None and not isinstance(paths, list):
        return "input_paths must be a list of server file paths"
    for i, p in enumerate(paths or []):
        if not isinstance(p, str):
            return f"input_paths[{i}] must be a string"
        image, perr = _load_server_image(p)
        if perr:
            return f"input_paths[{i}]: {perr}"
        imgs.append(_image_to_data_url(image))
    if len(imgs) > MAX_REFERENCE_IMAGES:
        return f"at most {MAX_REFERENCE_IMAGES} reference images are supported"
    if len(imgs) > 1 and not (flux_core._kontext_enabled or flux_core._flux_version == 2):
        return ("multiple reference images require the Kontext editor or a "
                "FLUX.2 server; this server's FLUX.1 img2img takes one image")
    data['input_images'] = imgs
    data.pop('input_image', None)

    if data.get('mask_image'):
        if not isinstance(data['mask_image'], str) or not _decodable(data['mask_image']):
            return "mask_image is not a decodable base64 image"
        if len(imgs) != 1:
            return "inpainting (mask_image) requires exactly one input image"
        if flux_core._flux_version != 2 and not getattr(flux_core, 'SUPPORTS_INPAINT', False):
            return "inpainting (mask_image) requires a FLUX.2 or SDXL server"

    return None


def _api_enqueue_generation(data):
    """Validate a generation request and put it on the queue.

    `data` is normalized in place by _validate_generate_params. Returns a list
    of (job, 1-based position in the pending list) pairs — one per prompt the
    request expands to, all sharing the validated parameters; raises ApiError
    otherwise. An unexpanded prompt yields a one-element list.
    """
    if not _model_ready:
        raise ApiError('Model still loading. Please wait.', 503, 'model_loading')

    prompt = (data.get('prompt') or '').strip()
    if not prompt:
        raise ApiError('prompt is required', 400, 'invalid_request')

    error = _validate_generate_params(data)
    if error:
        raise ApiError(error, 400, 'invalid_request')

    prompts = expand_prompt(prompt, QUEUE_MAX_SIZE)
    # More prompts than the queue can ever hold is a bad request, not a busy
    # server — no amount of waiting makes it fit.
    if prompts is None:
        raise ApiError(
            f'prompt expands to more than {QUEUE_MAX_SIZE} prompts, which is '
            f'the whole queue; use fewer alternatives in {{...}} groups',
            400, 'invalid_request')

    with _queue_cv:
        free = QUEUE_MAX_SIZE - len(_pending)
        if free < len(prompts):
            detail = (f'Queue is full ({QUEUE_MAX_SIZE} max). Cancel a queued '
                      f'job or wait.') if len(prompts) == 1 else (
                      f'This prompt expands to {len(prompts)} jobs but only '
                      f'{free} of the {QUEUE_MAX_SIZE} queue slots are free. '
                      f'Cancel a queued job or wait.')
            raise ApiError(detail, 429, 'queue_full')
        # An expansion is a group: its members are tracked together so the
        # last one to finish can tile them all into one contact sheet.
        exp_id = uuid.uuid4().hex[:12] if len(prompts) > 1 else None
        if exp_id:
            # The group exists to compare alternatives, so the prompt has to be
            # the only variable: draw one seed here and share it. Without this
            # each member would draw its own at generation time and the sheet
            # would compare two different images, not two prompts. Same
            # reasoning (and same default) as a multi-model run.
            if data.get('seed') is None and data.get('expansion_same_seed', True):
                data['seed'] = random.randint(0, 2**32 - 1)
            _expansion_register(exp_id, len(prompts), prompt,
                                hidden=bool(data.get('hidden')))
        queued = []
        for index, text in enumerate(prompts, start=1):
            # One params dict per job so each carries its own prompt; the
            # reference images inside are read-only and safely shared.
            params = dict(data)
            params['prompt'] = text
            job = Job(id=uuid.uuid4().hex[:12], params=params,
                      submitted_at=time.time())
            if exp_id:
                job.expansion = {'id': exp_id, 'index': index,
                                 'total': len(prompts), 'source': prompt}
            _pending.append(job)
            # 1-based position of this job in the pending list
            queued.append((job, len(_pending)))
        _queue_cv.notify()
    return queued


def _api_find_job(job_id):
    """The running, queued, or recently-finished job with this id, or None."""
    with _queue_cv:
        if _running_job and _running_job.id == job_id:
            return _running_job.full()
        for j in _pending:
            if j.id == job_id:
                return j.full()
        for j in _recent_done:
            if j.id == job_id:
                return j.full()
    return None


def _api_cancel_job(job_id):
    """Cancel a queued job outright, or ask the running one to stop. Returns a
    human-readable message; raises ApiError(404) if the id is unknown."""
    canceled = None
    with _queue_cv:
        for i, j in enumerate(_pending):
            if j.id == job_id:
                j.state = 'canceled'
                j.finished_at = time.time()
                del _pending[i]
                _recent_done.insert(0, j)
                del _recent_done[RECENT_DONE_MAX:]
                canceled = j
                break
        else:
            if _running_job and _running_job.id == job_id:
                # Interrupt: the generation loop checks this flag at every step
                # and raises JobCanceled; the worker then marks the job
                # canceled. Any batch images already finished are kept.
                _running_job.cancel_requested = True
                return f'Job {job_id} is stopping'
            raise ApiError('Job not found', 404, 'not_found')

    # A job canceled before it ran never reaches the worker, so close it out
    # with the group here — otherwise the rest of the expansion waits forever
    # for a sheet that never gets built. Outside the lock: this does disk I/O.
    _expansion_record(canceled)
    return f'Job {job_id} canceled'


def _api_list_step_frames(job_id):
    """A job's saved preview frames, in image/step order, as /images/ paths.

    Both steps/ dirs are scanned: the job that wrote the frames may have been
    a hidden one, and the caller only has its id to go on."""
    if not job_id.isalnum():
        raise ApiError('invalid job id', 400, 'invalid_request')
    paths = []
    for hidden in (False, True):
        prefix = _output_name('steps/', hidden)
        try:
            names = sorted(f for f in os.listdir(os.path.join(OUTPUT_DIR, prefix))
                           if f.startswith(job_id + '_') and f.endswith('.png'))
        except FileNotFoundError:
            continue
        paths.extend(prefix + n for n in names)
    return paths


def _api_queue_snapshot(include_hidden=False):
    """Running job, pending queue, and recently-finished jobs in one lock hold.

    Hidden jobs are omitted entirely unless the caller asks for them: their
    prompts and images are the whole point of the mode, and a second browser
    tab polling /status is exactly where they would otherwise surface."""
    def _visible(jobs):
        return jobs if include_hidden else [j for j in jobs if not j.hidden]

    with _queue_cv:
        running = _running_job if (include_hidden or not _running_job or
                                   not _running_job.hidden) else None
        return {
            'running': running.full() if running else None,
            'queued': [j.summary() for j in _visible(_pending)],
            'recent_done': [j.full() for j in _visible(_recent_done)],
        }


def _api_job_previews(job_id):
    """Intermediate imagery for one job.

    Two different things qualify, and a caller usually wants both:

    * `live` — the latent preview of the image being denoised right now,
      decoded and overwritten in place as generation proceeds. Present only
      while the job runs with `show_preview`. `ts` changes on every new frame,
      so use it as a cache-buster; the path itself is stable.
    * `frames` — every per-step frame written to disk, which only happens with
      `save_previews`. These persist after the job ends, so a finished job can
      still be replayed step by step.
    """
    if not job_id.isalnum():
        raise ApiError('invalid job id', 400, 'invalid_request')

    job = _api_find_job(job_id)
    frames = []
    for path in _api_list_step_frames(job_id):
        # steps/<jobid>_img<NN>_step<NNN>.png — the index and step are worth
        # parsing out so a client can group by image without re-deriving the
        # naming convention.
        match = re.search(r'_img(\d+)_step(\d+)\.png$', path)
        frames.append({
            'path': path,
            'image': int(match.group(1)) if match else None,
            'step': int(match.group(2)) if match else None,
        })

    live = None
    if job and job.get('preview'):
        live = {
            'path': job['preview'],
            'step': job.get('preview_step') or 0,
            'total_steps': job.get('total_steps') or 0,
            'image': job.get('current') or 0,
            # Milliseconds; changes with each decoded frame.
            'ts': job.get('preview_ts') or 0,
        }

    # The finished outputs, so a caller can show the whole build in one place:
    # noise -> saved steps -> the image it actually landed on.
    images = [{'filename': i.get('filename'), 'seed': i.get('seed')}
              for i in ((job.get('images') if job else None) or [])]

    return {
        'id': job_id,
        'state': job['state'] if job else None,
        'prompt': (job.get('prompt') if job else None) or '',
        'live': live,
        'frames': frames,
        'count': len(frames),
        'images': images,
        'generation_time': (job.get('generation_time') if job else 0) or 0,
        # False means no frames were ever written for this job, not that they
        # were lost — save_previews has to be requested at generation time.
        'saving': bool(job.get('saved_previews')) if job else bool(frames),
    }


def _recent_image_entry(job, img):
    """One entry of the queue view's "just generated" roll: the final PNG plus
    enough context to say what produced it."""
    filename = img.get('filename') or ''
    # flux{1|2}_YYYYMMDD_HHMMSS_{8hex}.png — the wall clock is in the name.
    parts = filename.split('_')
    stamp = parts[2] if len(parts) >= 3 else ''
    p = job.prompt
    return {
        'filename': filename,
        'job_id': job.id,
        'seed': img.get('seed'),
        'prompt': (p[:120] + '…') if len(p) > 120 else p,
        'time': f"{stamp[:2]}:{stamp[2:4]}:{stamp[4:6]}" if len(stamp) == 6 else None,
        'size': job.params.get('size'),
        'orientation': job.params.get('orientation'),
        'seconds': (img.get('timings') or {}).get('total'),
    }


def _recent_images_locked(limit=RECENT_IMAGES_MAX, include_hidden=False):
    """The last `limit` finished images, newest first. Caller holds _queue_cv.

    The running job leads: the batch members it has already written are on
    disk and final, even though the job itself isn't done. _recent_done is
    newest-job-first, but a job appends its images in generation order, so
    each job's own list is walked backwards. Hidden jobs are skipped unless
    asked for — /queue.html is a page anyone with the key can leave open.
    """
    out = []
    for job in ([_running_job] if _running_job else []) + _recent_done:
        if job.hidden and not include_hidden:
            continue
        for img in reversed(job.images):
            if img.get('filename'):
                out.append(_recent_image_entry(job, img))
                if len(out) >= limit:
                    return out
    return out


def _api_queue_view(include_hidden=False):
    """Queue-centric view: what is generating now, what is waiting and in what
    order, how much room is left, how long the backlog is likely to take, and
    the last few images that came out.

    Distinct from _api_queue_snapshot, which is the raw three-list dump the
    /status route has always returned. This one answers "where is my job in
    line and when will it run", so each waiting entry carries its 1-based
    position and the queue carries a wait estimate.
    """
    with _queue_cv:
        # Hidden jobs are dropped from `running`/`waiting` — their prompts are
        # the thing being concealed — but the counts below still include them.
        # A queue that claims to be accepting and then answers queue_full would
        # be a bug in the client's face; depth is a number, not content.
        shown_running = (_running_job if (_running_job and
                         (include_hidden or not _running_job.hidden)) else None)
        running = shown_running.full() if shown_running else None
        waiting = []
        for position, job in enumerate(_pending, start=1):
            if job.hidden and not include_hidden:
                continue
            entry = job.summary()
            entry['position'] = position
            waiting.append(entry)
        depth = len(_pending)
        busy = _running_job is not None or bool(_pending)
        # Images still to produce: everything queued, plus whatever is left of
        # the running job's batch.
        images_ahead = sum(int(j.params.get('batch') or 1) for j in _pending)
        if _running_job:
            images_ahead += max(0, int(_running_job.params.get('batch') or 1)
                                - int(_running_job.current or 0))
        recent_images = _recent_images_locked(include_hidden=include_hidden)
        # Only completed jobs carry a trustworthy duration; canceled ones
        # stopped early and would bias the estimate downward.
        samples = [(j.generation_time, len(j.images)) for j in _recent_done
                   if j.state == 'done' and j.generation_time > 0 and j.images]

    per_image = (sum(t / n for t, n in samples) / len(samples)) if samples else None

    return {
        'running': running,
        'waiting': waiting,
        'depth': depth,
        'capacity': QUEUE_MAX_SIZE,
        # The running job does not occupy a pending slot, so a full queue can
        # still have one job generating.
        'accepting': depth < QUEUE_MAX_SIZE,
        'busy': busy,
        'images_pending': images_ahead,
        'seconds_per_image': round(per_image, 2) if per_image else None,
        'estimated_wait_s': round(per_image * images_ahead, 1) if per_image else None,
        # Newest first, at most RECENT_IMAGES_MAX; empty until something
        # finishes, and only as deep as _recent_done still remembers.
        'recent_images': recent_images,
    }


@app.route('/generate', methods=['POST'])
def generate():
    try:
        queued = _api_enqueue_generation(request.json or {})
    except ApiError as e:
        return jsonify({'success': False, 'error': e.message}), e.status
    job, position = queued[0]
    # job_id/position stay the first job's so existing clients keep working;
    # a `{a|b}` prompt reports the rest in jobs/expanded.
    payload = {'success': True, 'job_id': job.id, 'position': position}
    if len(queued) > 1:
        payload['expanded'] = len(queued)
        payload['jobs'] = [{'job_id': j.id, 'position': p, 'prompt': j.params['prompt']}
                           for j, p in queued]
    return jsonify(payload)


@app.route('/steps/<job_id>')
def list_step_frames(job_id):
    """List a job's saved preview frames (the save_previews toggle), in
    image/step order, as paths servable via /images/."""
    try:
        frames = _api_list_step_frames(job_id)
    except ApiError as e:
        return jsonify({'success': False, 'error': e.message}), e.status
    return jsonify({'success': True, 'frames': frames})


# <path:> so saved preview frames under steps/ are reachable too;
# send_from_directory rejects anything escaping OUTPUT_DIR.
@app.route('/images/<path:filename>')
def serve_image(filename):
    return send_from_directory(OUTPUT_DIR, filename)


@app.route('/status')
def status():
    snapshot = _api_queue_snapshot(_hidden_requested(request.args.get('hidden')))
    snapshot.update({
        'queue_max_size': QUEUE_MAX_SIZE,
        'power_w': _gpu_power_watts(),
        'vlm': _vlm_status(),
    })
    return jsonify(snapshot)


@app.route('/reset', methods=['POST'])
def reset_recent():
    """Clear the in-memory list of recently-completed jobs so the client's
    results view stays empty across page reloads."""
    with _queue_cv:
        _recent_done.clear()
    return jsonify({'success': True})


@app.route('/jobs/<job_id>/cancel', methods=['POST'])
def cancel_job(job_id):
    try:
        message = _api_cancel_job(job_id)
    except ApiError as e:
        return jsonify({'success': False, 'error': e.message}), e.status
    return jsonify({'success': True, 'message': message})


@app.route('/configs')
def configs():
    """The launcher's model-config menu, for the UI's model switcher.

    `switchable` is false when the server was started directly (no
    run_server.sh supervisor), in which case /switch-model is unavailable
    and the UI hides the control.
    """
    return jsonify(_api_configs())


@app.route('/switch-model', methods=['POST'])
def switch_model():
    """Switch to another run_server.sh configuration on the fly.

    The models are far too large to hot-swap in-process (and the SDXL backend
    is chosen at import time), so switching is a supervised restart: write the
    requested config number to SWITCH_CONFIG_FILE and exit with
    SWITCH_EXIT_CODE. The run_server.sh restart loop picks the file up and
    relaunches with the new config's flags; clients poll /ready until the new
    model is up.
    """
    try:
        label = _api_switch_model((request.json or {}).get('config'))
    except ApiError as e:
        return jsonify({'success': False, 'error': e.message}), e.status
    return jsonify({'success': True, 'switching_to': label})


def _api_switch_model(raw_config):
    """Schedule the supervised restart into another config. Returns the target
    config's label; the process exits 0.5s later. Raises ApiError when there is
    no supervisor, the id is bad, or work is in flight."""
    if _current_config is None:
        raise ApiError(
            'Model switching requires launching via run_server.sh (no supervisor detected).',
            400, 'no_supervisor')
    try:
        target = int(raw_config)
    except (TypeError, ValueError):
        raise ApiError('config must be an integer', 400, 'invalid_request')
    if target not in SERVER_CONFIGS:
        raise ApiError(f'config must be one of {sorted(SERVER_CONFIGS)}', 400, 'invalid_request')
    if target == _current_config:
        raise ApiError('Already running this configuration', 400, 'invalid_request')

    with _queue_cv:
        if _running_job is not None or _pending:
            raise ApiError(
                'Jobs are running or queued. Interrupt/cancel them before switching models.',
                409, 'busy')

    state = _multi_run_load()
    if state and not state.get('finished'):
        raise ApiError(
            'A multi-model run is active. Cancel it before switching models manually.',
            409, 'busy')

    with open(SWITCH_CONFIG_FILE, 'w') as f:
        f.write(str(target))
    print(f"[switch] restarting into config {target} ({SERVER_CONFIGS[target]})", flush=True)
    # Give Flask a moment to flush this response before the process exits.
    threading.Timer(0.5, lambda: os._exit(SWITCH_EXIT_CODE)).start()
    return SERVER_CONFIGS[target]


@app.route('/multi-run', methods=['POST'])
def multi_run_start():
    """Start a multi-model run: the same prompt generated on each selected
    run_server.sh config in turn (see the multi-run engine above).

    Body: {configs: [ids], prompt, and the usual /generate params}. Reference
    images, masks, and negative prompts are per-model capabilities, so runs
    are text-to-image only. Unless a seed is given, one is drawn here and
    shared by every model so the outputs are comparable.
    """
    try:
        state = _api_start_multi_run(request.json or {})
    except ApiError as e:
        return jsonify({'success': False, 'error': e.message}), e.status
    return jsonify({'success': True, 'id': state['id'], 'configs': state['configs'],
                    'seed': state['params']['seed']})


def _api_start_multi_run(data):
    """Validate and start a multi-model run. Returns the persisted run state
    document; raises ApiError on any rejection."""
    if _current_config is None:
        raise ApiError(
            'Multi-model runs require launching via run_server.sh (no supervisor detected).',
            400, 'no_supervisor')
    if not _model_ready:
        raise ApiError('Model still loading. Please wait.', 503, 'model_loading')

    prompt = (data.get('prompt') or '').strip()
    if not prompt:
        raise ApiError('prompt is required', 400, 'invalid_request')

    configs = data.get('configs')
    if not isinstance(configs, list) or not configs:
        raise ApiError('configs must be a non-empty list', 400, 'invalid_request')
    try:
        configs = list(dict.fromkeys(int(c) for c in configs))
    except (TypeError, ValueError):
        raise ApiError('configs must be a list of integers', 400, 'invalid_request')
    bad = [c for c in configs if c not in SERVER_CONFIGS]
    if bad:
        raise ApiError(f'unknown configs: {bad}', 400, 'invalid_request')

    if data.get('input_images') or data.get('input_image') or data.get('input_paths') or data.get('mask_image'):
        raise ApiError('multi-model runs are text-to-image only (remove reference images)',
                       400, 'invalid_request')
    if data.get('negative_prompt'):
        raise ApiError('negative_prompt is SDXL-only and not supported in multi-model runs',
                       400, 'invalid_request')

    params = {'prompt': prompt}
    for k in ('orientation', 'size', 'steps', 'seed', 'guidance', 'batch', 'show_preview'):
        if data.get(k) is not None:
            params[k] = data[k]
    error = _validate_generate_params(params)
    if error:
        raise ApiError(error, 400, 'invalid_request')
    # One shared seed (unless given) so the models' outputs are comparable.
    if params.get('seed') is None:
        params['seed'] = random.randint(0, 2**32 - 1)

    with _multi_run_lock:
        state = _multi_run_load()
        if state and not state.get('finished'):
            raise ApiError('A multi-model run is already active. Cancel it first.', 409, 'busy')
        with _queue_cv:
            if _running_job is not None or _pending:
                raise ApiError(
                    'Jobs are running or queued. Wait for or cancel them before a multi-model run.',
                    409, 'busy')
        # Run the current model first when it's selected — saves one restart.
        if _current_config in configs:
            configs.remove(_current_config)
            configs.insert(0, _current_config)
        state = {
            'id': uuid.uuid4().hex[:12],
            'created': time.time(),
            'prompt': prompt,
            'params': params,
            'configs': configs,
            'results': [],
        }
        _multi_run_save(state)
        _multi_run_advance()
    return state


def _api_multi_run_state():
    """State of the active (or last finished, un-dismissed) multi-model run."""
    state = _multi_run_load()
    if not state:
        return {'active': False, 'run': None}
    return {
        'active': not state.get('finished'),
        'run': state,
        'labels': {str(k): v for k, v in SERVER_CONFIGS.items()},
        'current_config': _current_config,
        'next_config': _multi_run_next_config(state),
    }


def _api_cancel_multi_run():
    """Cancel the active run (interrupting its in-flight job) or dismiss a
    finished one. Returns True if a run was actually cleared."""
    with _multi_run_lock:
        state = _multi_run_load()
        _multi_run_clear()
    if not state:
        return False
    rid = state['id']
    with _queue_cv:
        for i, j in enumerate(_pending):
            if j.multi_run == rid:
                j.state = 'canceled'
                j.finished_at = time.time()
                del _pending[i]
                _recent_done.insert(0, j)
                del _recent_done[RECENT_DONE_MAX:]
                break
        if _running_job is not None and _running_job.multi_run == rid:
            _running_job.cancel_requested = True
    return True


@app.route('/multi-run', methods=['GET'])
def multi_run_status():
    """State of the active (or last finished, un-dismissed) multi-model run."""
    return jsonify(_api_multi_run_state())


@app.route('/multi-run/cancel', methods=['POST'])
def multi_run_cancel():
    """Cancel the active multi-model run (interrupting its job if one is
    queued or generating), or dismiss a finished run's results."""
    _api_cancel_multi_run()
    return jsonify({'success': True})


def _model_type_string():
    flux_name = f"FLUX.{flux_core._flux_version}"
    variant = "-klein-4b" if _klein_4b else "-klein" if _klein else "-dev"
    if _SDXL_ACTIVE:
        # Deliberately not the checkpoint basename — this string shows in the
        # UI's model-name hover, and checkpoint repo ids can be lurid.
        return "SDXL (photoreal)"
    if _kontext:
        kontext_prec = "full bf16" if _full_model else "4-bit"
        return f"FLUX.1-Kontext (editor, {kontext_prec})"
    if _schnell:
        return f"{flux_name}-schnell (4-step)"
    if _gguf_quant:
        return f"{flux_name}-dev GGUF {_gguf_quant.upper()}"
    if _full_model:
        return f"{flux_name}{variant} (full)"
    return f"{flux_name}-dev-bnb-4bit"


@app.route('/model-info')
def model_info():
    return jsonify(_api_model_info())


def _api_model_info():
    """Capabilities of the loaded backend: which optional features (negative
    prompts, inpainting, Kontext editing) this process can actually serve."""
    model_type = _model_type_string()
    encoder_type = ("local CLIP encoders" if _SDXL_ACTIVE
                    else "local encoder" if _local_encoder else "remote encoder")
    turbo_str = " + Turbo" if flux_core._turbo_enabled else ""
    uncensored_str = " + U-LoRA" if flux_core._uncensored_enabled and not _SDXL_ACTIVE else ""
    return {
        'model': model_type,
        'encoder': encoder_type,
        'turbo': flux_core._turbo_enabled,
        'schnell': _schnell,
        'uncensored': flux_core._uncensored_enabled,
        'kontext': flux_core._kontext_enabled,
        'flux_version': flux_core._flux_version,
        'sd': _SDXL_ACTIVE,
        'negative_prompt': getattr(flux_core, 'SUPPORTS_NEGATIVE_PROMPT', False),
        'inpaint': flux_core._flux_version == 2 or getattr(flux_core, 'SUPPORTS_INPAINT', False),
        'vae_tiling': getattr(flux_core, '_vae_tiling_mode', 'auto'),
        'vae_tiling_threshold_mp': getattr(flux_core, '_vae_tiling_threshold_mp', None),
        'hostname': socket.gethostname(),
        'version': VERSION,
        'description': f"{model_type}{turbo_str}{uncensored_str} with {encoder_type}"
    }


def _api_readiness():
    """Model-load progress, for readiness probes."""
    elapsed = time.perf_counter() - _model_load_start_ts if _model_load_start_ts else 0.0
    return {
        'ready': _model_ready,
        'error': _model_load_error,
        'status': _model_load_status,
        'elapsed_s': round(elapsed, 1),
    }


def _api_configs():
    """The launcher's model-config menu plus which one is live."""
    return {
        'configs': [{'id': k, 'label': v} for k, v in SERVER_CONFIGS.items()],
        'current': _current_config,
        'switchable': _current_config is not None,
    }


# Async VLM job bookkeeping (critique, describe, boost). A qwen3.6 vision call
# can run for several minutes — far past the ~60s connection cap browsers
# (Safari especially) put on a single fetch — so the POST endpoints only
# validate and start the work, and the client polls GET /<route>/<id> for
# the result.
_vlm_jobs_lock = threading.Lock()
_vlm_jobs: dict = {}  # id -> {'done': bool, 'created': float, 'result': dict|None}
VLM_JOB_TTL_S = 3600


def _vlm_job_start(runner, *args):
    """Register a job, run `runner(cid, *args)` on a daemon thread, and
    return the id the client polls with."""
    cid = uuid.uuid4().hex[:12]
    now = time.time()
    with _vlm_jobs_lock:
        for old_id in [k for k, v in _vlm_jobs.items()
                       if v['done'] and now - v['created'] > VLM_JOB_TTL_S]:
            del _vlm_jobs[old_id]
        _vlm_jobs[cid] = {'done': False, 'created': now, 'result': None}
    threading.Thread(target=runner, daemon=True, name=f'vlm-job-{cid}',
                     args=(cid,) + args).start()
    return cid


def _vlm_job_finish(cid, payload):
    with _vlm_jobs_lock:
        entry = _vlm_jobs.get(cid)
        if entry is not None:
            entry['result'] = payload
            entry['done'] = True


def _api_vlm_result(cid):
    """(payload, done) for an async VLM job. `payload` is None while the job is
    still running. Raises ApiError(404) if the id is unknown or expired."""
    with _vlm_jobs_lock:
        entry = _vlm_jobs.get(cid)
        if entry is None:
            raise ApiError(f'unknown job id {cid}', 404, 'not_found')
        if not entry['done']:
            return None, False
        return dict(entry['result']), True


def _vlm_job_status(cid):
    """Shared poll response for GET /critique|describe|boost/<id>."""
    try:
        result, done = _api_vlm_result(cid)
    except ApiError as e:
        return jsonify({'success': False, 'error': e.message}), e.status
    if not done:
        return jsonify({'success': True, 'done': False})
    result['done'] = True
    status = 200 if result.get('success') else 500
    return jsonify(result), status


def _run_critique(cid, model, direction, prompt, reference, output, history, style):
    from edit_loop import (vlm_critique, edit_metrics, heuristic_revision,
                           describe_metrics)
    try:
        metrics = edit_metrics(reference, output)
        result = vlm_critique(model, direction, prompt, reference, output, metrics,
                              ollama_url=OLLAMA_URL, style=style, history=history)
        if result:
            payload = {'success': True, 'vlm': True, 'metrics': metrics,
                       'metrics_text': describe_metrics(metrics),
                       'applied': bool(result.get('applied')),
                       'score': result.get('score'),
                       'critique': result.get('critique', ''),
                       'revised_prompt': result.get('revised_prompt')}
        else:
            payload = {'success': True, 'vlm': False, 'metrics': metrics,
                       'metrics_text': describe_metrics(metrics),
                       'applied': None,
                       'score': None,
                       'critique': 'Vision model unavailable — revision based on pixel metrics only.',
                       'revised_prompt': heuristic_revision(direction, prompt, metrics)}
    except Exception as e:
        payload = {'success': False, 'error': f'critique failed: {e}'}
    _vlm_job_finish(cid, payload)


@app.route('/critique', methods=['POST'])
def critique():
    """Compare an edit output against its reference and propose a revised
    instruction — the "look at the output" step of the edit loop (UI panel
    and edit_loop.py). Vision critique runs on the VLM endpoint (OLLAMA_URL,
    with CRITIQUE_MODEL naming the model or "auto" to use whatever that
    endpoint is currently serving); when it's unavailable the
    result falls back to pixel-metric heuristics. Returns a critique_id
    immediately; poll GET /critique/<id> for the result."""
    try:
        cid = _api_start_critique(request.json or {})
    except ApiError as e:
        return jsonify({'success': False, 'error': e.message}), e.status
    return jsonify({'success': True, 'critique_id': cid})


def _api_start_critique(data):
    """Validate a critique request and start it on a worker thread. Returns the
    job id to poll; raises ApiError on bad input."""
    direction = (data.get('direction') or '').strip()
    prompt = (data.get('prompt') or direction).strip()
    raw_out = data.get('output_filename') or ''
    # A hidden-mode image is named `.hidden/<file>`; every other name is
    # reduced to its basename, as before.
    out_filename = raw_out if raw_out.startswith(HIDDEN_PREFIX) else os.path.basename(raw_out)
    ref_b64 = data.get('ref_image') or ''
    if not direction or not out_filename or not ref_b64:
        raise ApiError('direction, ref_image, and output_filename are required',
                       400, 'invalid_request')

    out_path = _resolve_output_name(out_filename)
    if not os.path.exists(out_path):
        raise ApiError(f'unknown output image {out_filename}', 404, 'not_found')
    try:
        if ref_b64.startswith('data:'):
            ref_b64 = ref_b64.split(',', 1)[1]
        reference = Image.open(io.BytesIO(base64.b64decode(ref_b64))).convert('RGB')
    except Exception:
        raise ApiError('ref_image is not a decodable base64 image', 400, 'invalid_request')

    output = Image.open(out_path).convert('RGB')
    model = data.get('model') or CRITIQUE_MODEL
    style = ("description" if getattr(flux_core, 'OUTPUT_PREFIX', '') == 'sdxl'
             else "instruction")
    # Prompt trajectory from earlier iterations ({prompt, applied, score,
    # critique} dicts) so the critic doesn't re-propose failed phrasings.
    history = data.get('history') if isinstance(data.get('history'), list) else []

    return _vlm_job_start(_run_critique, model, direction, prompt, reference,
                          output, history, style)


@app.route('/critique/<cid>')
def critique_result(cid):
    """Poll for an async critique started by POST /critique."""
    return _vlm_job_status(cid)


def _run_describe(cid, model, images, think):
    from edit_loop import vlm_describe
    try:
        model, url, api_key = _ollama_call_params(model)
        prompt = vlm_describe(model, images, think=think, ollama_url=url,
                              api_key=api_key)
        if prompt:
            payload = {'success': True, 'prompt': prompt}
        else:
            payload = {'success': False,
                       'error': 'vision model unavailable or returned no description'}
    except Exception as e:
        payload = {'success': False, 'error': f'describe failed: {e}'}
    _vlm_job_finish(cid, payload)


@app.route('/describe', methods=['POST'])
def describe():
    """The reverse path: have the local vision model write a detailed
    text-to-image prompt that would recreate the posted photo, for
    generating a fresh image from that prompt alone. Accepts `images` (a
    list, up to MAX_REFERENCE_IMAGES) or legacy single `image`; with several
    images the prompt is a composite describing one scene that combines
    them. `think` (default false) enables the VLM's deliberation phase —
    deeper but much slower. Returns a describe_id immediately; poll
    GET /describe/<id> for the prompt."""
    try:
        cid = _api_start_describe(request.json or {})
    except ApiError as e:
        return jsonify({'success': False, 'error': e.message}), e.status
    return jsonify({'success': True, 'describe_id': cid})


def _api_start_describe(data):
    """Validate a describe request and start it. Returns the job id to poll."""
    imgs_b64 = data.get('images') if isinstance(data.get('images'), list) else []
    if not imgs_b64 and data.get('image'):
        imgs_b64 = [data['image']]
    if not imgs_b64:
        raise ApiError('image is required', 400, 'invalid_request')
    if len(imgs_b64) > MAX_REFERENCE_IMAGES:
        raise ApiError(f'at most {MAX_REFERENCE_IMAGES} images are supported',
                       400, 'invalid_request')
    images = []
    try:
        for img_b64 in imgs_b64:
            if img_b64.startswith('data:'):
                img_b64 = img_b64.split(',', 1)[1]
            images.append(Image.open(io.BytesIO(base64.b64decode(img_b64))).convert('RGB'))
    except Exception:
        raise ApiError('image is not a decodable base64 image', 400, 'invalid_request')
    think = bool(data.get('think', False))
    model = data.get('model') or DESCRIBE_MODEL
    return _vlm_job_start(_run_describe, model, images, think)


@app.route('/describe/<cid>')
def describe_result(cid):
    """Poll for an async describe started by POST /describe."""
    return _vlm_job_status(cid)


# Camera RAW (NEF etc.) reference support: browsers can't decode RAW files, so
# the UI posts them here as-is and gets back a browser-usable JPEG data URL
# that then flows through the normal reference path. Decoded with LibRaw via
# rawpy (ffmpeg has no NEF decoder). References are conditioned at ~2MP, so
# half_size skips the full-resolution demosaic and MAX_RAW_EDGE bounds the
# returned JPEG the same way the client bounds ordinary uploads.
MAX_RAW_EDGE = 2048


def _largest_embedded_jpeg(data):
    """Largest JPEG embedded in a RAW container, or None. Cameras store
    full-size JPEG previews inside RAW files; when LibRaw can't decode the
    raw data itself (body newer than the LibRaw release, Nikon High
    Efficiency NEFs), the preview is still a faithful full-res rendering.
    Scans for JPEG SOI markers and lets PIL parse from each — PIL stops at
    the matching EOI, so trailing container bytes are harmless."""
    best = None
    pos = 0
    for _ in range(16):
        pos = data.find(b'\xff\xd8\xff', pos)
        if pos < 0:
            break
        try:
            img = Image.open(io.BytesIO(data[pos:]))
            img.load()
            if best is None or img.width * img.height > best.width * best.height:
                best = img.convert('RGB')
        except Exception:
            pass
        pos += 3
    return best


def _raw_to_pil(data):
    """Decode camera RAW bytes to a PIL image: full LibRaw demosaic first,
    then LibRaw's thumbnail extractor, then the embedded-JPEG scan."""
    import rawpy
    try:
        with rawpy.imread(io.BytesIO(data)) as raw:
            return Image.fromarray(raw.postprocess(use_camera_wb=True, half_size=True))
    except Exception as demosaic_err:
        try:
            with rawpy.imread(io.BytesIO(data)) as raw:
                thumb = raw.extract_thumb()
            if thumb.format == rawpy.ThumbFormat.JPEG:
                return Image.open(io.BytesIO(thumb.data)).convert('RGB')
            return Image.fromarray(thumb.data)
        except Exception:
            pass
        image = _largest_embedded_jpeg(data)
        if image is not None:
            return image
        raise demosaic_err


@app.route('/convert-raw', methods=['POST'])
def convert_raw():
    """Convert an uploaded camera RAW file (NEF/DNG/CR3/...) to a JPEG data
    URL for use as a reference image. Takes a multipart form `file` field
    (raw bytes, not base64 — a 50MB NEF must fit the 64MB body cap)."""
    f = request.files.get('file')
    if f is None:
        return jsonify({'success': False, 'error': 'multipart form field "file" is required'}), 400
    try:
        image = _api_convert_raw(f.read(), f.filename)
    except ApiError as e:
        return jsonify({'success': False, 'error': e.message}), e.status
    return jsonify(dict(_api_reference_payload(image), success=True))


def _api_reference_payload(image):
    """The {image, width, height} body every reference-import path returns:
    a JPEG data URL bounded to MAX_RAW_EDGE, plus its final dimensions.
    _image_to_data_url resizes in place, so read the size after encoding."""
    data_url = _image_to_data_url(image)
    return {'image': data_url, 'width': image.width, 'height': image.height}


def _api_convert_raw(data, filename=''):
    """Decode camera RAW bytes to a PIL image. Raises ApiError with a message
    that identifies the file when the decode fails."""
    try:
        import rawpy  # noqa: F401 — fail fast with a clear error if absent
    except ImportError:
        raise ApiError('RAW conversion requires the rawpy package on the server '
                       '(uv pip install rawpy)', 501, 'not_implemented')
    if not data:
        raise ApiError('uploaded file is empty', 400, 'invalid_request')
    try:
        return _raw_to_pil(data)
    except Exception as e:
        # Size + header bytes make "what was this file actually?" answerable
        # from the client-side error alone.
        print(f"convert-raw failed: {filename!r}, {len(data)} bytes, header {data[:12].hex()}: {e}")
        raise ApiError(f'could not decode RAW file ({len(data)} bytes, '
                       f'header {data[:12].hex()}): {e}', 400, 'undecodable_image')


# Cap on a fetched remote image, mirroring the request body cap so a URL
# can't pull in more than an upload could.
MAX_URL_FETCH_BYTES = 64 * 1024 * 1024


@app.route('/fetch-image-url', methods=['POST'])
def fetch_image_url():
    """Fetch an image from a remote http(s) URL server-side (no browser CORS
    restrictions) and return it as a JPEG data URL for use as a reference
    image. JSON body: {"url": "https://..."}. Response shape matches
    /convert-raw."""
    try:
        image = _api_fetch_url_image((request.get_json(silent=True) or {}).get('url'))
    except ApiError as e:
        return jsonify({'success': False, 'error': e.message}), e.status
    return jsonify(dict(_api_reference_payload(image), success=True))


def _api_fetch_url_image(raw_url):
    """Fetch a remote image server-side and return it as a PIL image, bounded
    by MAX_URL_FETCH_BYTES. Raises ApiError on a bad URL, transport failure, or
    a body that is neither a normal image nor camera RAW."""
    import requests
    url = (raw_url or '').strip()
    if not url.lower().startswith(('http://', 'https://')):
        raise ApiError('url must start with http:// or https://', 400, 'invalid_request')
    # Browser-like header set: Wikimedia (and similar CDNs) 429 requests that
    # carry a browser User-Agent without the matching Accept/Accept-Language
    # headers, so the UA alone is not enough.
    browser_headers = {
        'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64; rv:140.0) Gecko/20100101 Firefox/140.0',
        'Accept': 'image/avif,image/webp,image/png,image/svg+xml,image/*;q=0.8,*/*;q=0.5',
        'Accept-Language': 'en-US,en;q=0.5',
        'Sec-Fetch-Dest': 'image',
        'Sec-Fetch-Mode': 'no-cors',
        'Sec-Fetch-Site': 'cross-site',
    }
    try:
        resp = requests.get(url, timeout=30, stream=True, allow_redirects=True,
                            headers=browser_headers)
        resp.raise_for_status()
        chunks, total = [], 0
        for chunk in resp.iter_content(1024 * 1024):
            total += len(chunk)
            if total > MAX_URL_FETCH_BYTES:
                raise ApiError(f'image exceeds the {MAX_URL_FETCH_BYTES // (1024 * 1024)}MB '
                               'fetch limit', 400, 'too_large')
            chunks.append(chunk)
        data = b''.join(chunks)
    except requests.RequestException as e:
        raise ApiError(f'could not fetch URL: {e}', 400, 'fetch_failed')
    if not data:
        raise ApiError('URL returned an empty response', 400, 'fetch_failed')
    try:
        return Image.open(io.BytesIO(data)).convert('RGB')
    except Exception:
        try:
            # A URL can point at a camera RAW file too; reuse the RAW pipeline.
            return _raw_to_pil(data).convert('RGB')
        except Exception:
            print(f"fetch-image-url failed: {url!r}, {len(data)} bytes, header {data[:12].hex()}")
            raise ApiError(f'URL did not return a decodable image ({len(data)} bytes, '
                           f'header {data[:12].hex()})', 400, 'undecodable_image')


def _load_server_image(raw_path):
    """Open an image that already lives on this machine's filesystem for use
    as a reference. Bare/relative paths resolve against OUTPUT_DIR; absolute
    and ~-prefixed paths are used as-is (the global API-key check is what
    gates this — the server owner reading their own disk). Returns
    (PIL image, None) on success or (None, error string) on failure."""
    path = os.path.expanduser(raw_path)
    if not os.path.isabs(path):
        path = os.path.join(OUTPUT_DIR, path)
    if not os.path.isfile(path):
        return None, f'no such file on the server: {path}'
    if os.path.getsize(path) > MAX_URL_FETCH_BYTES:
        return None, f'{path} exceeds the {MAX_URL_FETCH_BYTES // (1024 * 1024)}MB limit'
    with open(path, 'rb') as f:
        data = f.read()
    try:
        return Image.open(io.BytesIO(data)).convert('RGB'), None
    except Exception:
        try:
            # A server path can point at a camera RAW file too.
            return _raw_to_pil(data).convert('RGB'), None
        except Exception:
            print(f"load server image failed: {path!r}, {len(data)} bytes, header {data[:12].hex()}")
            return None, f'{path} is not a decodable image ({len(data)} bytes, header {data[:12].hex()})'


def _image_to_data_url(image):
    """Bound a reference image to MAX_RAW_EDGE and encode it as a JPEG data
    URL, matching what /convert-raw and /fetch-image-url return."""
    image.thumbnail((MAX_RAW_EDGE, MAX_RAW_EDGE), Image.LANCZOS)
    buf = io.BytesIO()
    image.save(buf, format='JPEG', quality=92)
    return 'data:image/jpeg;base64,' + base64.b64encode(buf.getvalue()).decode('ascii')


@app.route('/fetch-image-path', methods=['POST'])
def fetch_image_path():
    """Load an image from the server's own filesystem and return it as a JPEG
    data URL for use as a reference image. JSON body: {"path": "..."} —
    absolute, ~-prefixed, or relative to web-generated/. Response shape
    matches /convert-raw and /fetch-image-url."""
    try:
        image = _api_load_path_image((request.get_json(silent=True) or {}).get('path'))
    except ApiError as e:
        return jsonify({'success': False, 'error': e.message}), e.status
    return jsonify(dict(_api_reference_payload(image), success=True))


def _api_load_path_image(raw_path):
    """Load a reference image from this machine's filesystem as a PIL image."""
    raw_path = (raw_path or '').strip()
    if not raw_path:
        raise ApiError('path is required', 400, 'invalid_request')
    image, err = _load_server_image(raw_path)
    if err:
        raise ApiError(err, 400, 'undecodable_image')
    return image


@app.route('/browse-files')
def browse_files():
    """List subfolders and displayable images in a directory on this
    machine's filesystem, for the reference-image picker's folder browser.
    `dir` query param resolves the same way /fetch-image-path resolves a
    reference path: relative to OUTPUT_DIR, or used as-is if absolute/~
    (the API-key check is what gates this — the server owner browsing their
    own disk). Defaults to the archive folder. Returns the resolved absolute
    `dir` plus its `parent` (null at filesystem root) so the client can
    navigate up/down without doing its own path arithmetic."""
    try:
        listing = _api_browse_dir(request.args.get('dir'))
    except ApiError as e:
        return jsonify({'success': False, 'error': e.message}), e.status
    return jsonify(dict(listing, success=True))


def _api_browse_dir(raw_dir):
    """{dir, parent, dirs, files} for a directory on this machine, with
    subfolders name-sorted and images newest-first."""
    raw_dir = (raw_dir or 'archive').strip() or 'archive'
    path = os.path.expanduser(raw_dir)
    if not os.path.isabs(path):
        path = os.path.join(OUTPUT_DIR, path)
    # Always return an absolute dir: the client round-trips it verbatim for
    # "up" navigation and thumbnail paths, and a relative value here would
    # get OUTPUT_DIR joined onto it a second time on the next request.
    path = os.path.abspath(path)
    if not os.path.isdir(path):
        raise ApiError(f'no such directory on the server: {path}', 400, 'not_found')
    exts = ('.png', '.jpg', '.jpeg', '.webp')
    dirs, files = [], []
    try:
        for entry in os.scandir(path):
            if entry.name.startswith('.'):
                continue
            if entry.is_dir():
                dirs.append(entry.name)
            elif entry.is_file() and entry.name.lower().endswith(exts):
                files.append({'filename': entry.name, 'mtime': entry.stat().st_mtime})
    except PermissionError:
        raise ApiError(f'permission denied: {path}', 400, 'forbidden')
    dirs.sort(key=str.lower)
    files.sort(key=lambda f: f['mtime'], reverse=True)
    for f in files:
        del f['mtime']
    parent = os.path.dirname(path) if os.path.dirname(path) != path else None
    return {'dir': path, 'parent': parent, 'dirs': dirs, 'files': files}


def _api_thumbnail_bytes(raw_path, edge=240):
    """JPEG bytes of a small thumbnail for any image _load_server_image can
    reach. Raises ApiError(404) when the path can't be read."""
    raw_path = (raw_path or '').strip()
    if not raw_path:
        raise ApiError('path is required', 400, 'invalid_request')
    image, err = _load_server_image(raw_path)
    if err:
        raise ApiError(err, 404, 'not_found')
    image.thumbnail((edge, edge), Image.LANCZOS)
    buf = io.BytesIO()
    image.save(buf, format='JPEG', quality=80)
    return buf.getvalue()


@app.route('/browse-thumb')
def browse_thumb():
    """Small JPEG thumbnail for an image anywhere on this machine's
    filesystem, for the /browse-files picker grid. Unlike /images/<path>
    (exempt from auth, restricted to OUTPUT_DIR by send_from_directory),
    this can read any path _load_server_image can reach, so — unlike
    /images/ — it stays behind the normal API-key check."""
    try:
        data = _api_thumbnail_bytes(request.args.get('path'))
    except ApiError as e:
        # Legacy contract: an empty body, since the caller renders this
        # straight into an <img> and never reads an error payload.
        return '', 400 if e.status == 400 else 404
    return Response(data, mimetype='image/jpeg')


def _boost_family(has_image=False):
    """Prompting idiom of the loaded backend, keying edit_loop.BOOST_GUIDANCE.
    With reference image(s) attached the idiom shifts: FLUX.2 and Kontext
    want the prompt as active edit instructions against the reference;
    SDXL and FLUX.1 img2img want the desired final image described in full,
    never the delta."""
    if _SDXL_ACTIVE:
        return 'sdxl-img2img' if has_image else 'sdxl'
    if _kontext:
        return 'kontext'
    if flux_core._flux_version == 2:
        return 'flux2-edit' if has_image else 'flux2'
    return 'flux1-img2img' if has_image else 'flux1'


def _run_boost(cid, model, prompt, family, model_desc, level, think, variant=None,
               negative_prompt=None):
    from edit_loop import vlm_boost
    try:
        boosted = vlm_boost(model, prompt, family=family, model_desc=model_desc,
                            level=level, think=think, variant=variant,
                            negative_prompt=negative_prompt, ollama_url=OLLAMA_URL)
        if boosted:
            payload = {'success': True, 'prompt': boosted['prompt'],
                       'negative_prompt': boosted.get('negative_prompt')}
        else:
            # Boost is a text-only call, so "vision model" misdirected: every
            # failure here is the VLM endpoint being unreachable, speaking a
            # dialect we got wrong, or replying without a usable prompt. The
            # specific reason is printed by vlm_boost.
            payload = {'success': False,
                       'error': f'no rewrite from VLM at {OLLAMA_URL} '
                                f'(model {model}) — see server.log'}
    except Exception as e:
        payload = {'success': False, 'error': f'boost failed: {e}'}
    _vlm_job_finish(cid, payload)


@app.route('/boost', methods=['POST'])
def boost():
    """Rewrite the user's draft prompt into a stronger one tuned to the
    prompting idiom of the currently loaded model (descriptive prose for
    FLUX, an imperative instruction for Kontext, tag phrases for SDXL —
    shifted to edit-instruction / final-image idiom when `has_image` says
    references are attached), via the local ollama model. `level` 1-5 sets
    how far the rewrite may depart from the draft (1 = polish wording only,
    5 = reimagine boldly); `think` (default false) enables the VLM's
    thinking phase — deeper but much slower. Optional `negative_prompt`
    (SDXL) is rewritten alongside the positive prompt in the same call, so
    the pair stays consistent — the poll response then carries both `prompt`
    and `negative_prompt`. Optional `variant_index` + `variant_count` mark
    this as one of N independent rewrites of the same draft (the evolve
    feature): the VLM is pushed toward a direction the other rewrites are
    unlikely to take. Returns a boost_id immediately;
    poll GET /boost/<id> for the improved prompt (+ negative_prompt)."""
    try:
        cid = _api_start_boost(request.json or {})
    except ApiError as e:
        return jsonify({'success': False, 'error': e.message}), e.status
    return jsonify({'success': True, 'boost_id': cid})


def _api_start_boost(data):
    """Validate a boost request and start it. Returns the job id to poll."""
    prompt = (data.get('prompt') or '').strip()
    if not prompt:
        raise ApiError('prompt is required', 400, 'invalid_request')
    try:
        level = int(data.get('level', 3))
    except (TypeError, ValueError):
        raise ApiError('level must be an integer 1-5', 400, 'invalid_request')
    if not 1 <= level <= 5:
        raise ApiError('level must be an integer 1-5', 400, 'invalid_request')
    variant = None
    if data.get('variant_count') is not None:
        try:
            v_idx = int(data.get('variant_index', 0))
            v_cnt = int(data['variant_count'])
        except (TypeError, ValueError):
            raise ApiError('variant_index/variant_count must be integers', 400, 'invalid_request')
        if not 1 <= v_idx <= v_cnt:
            raise ApiError('variant_index must be between 1 and variant_count',
                           400, 'invalid_request')
        variant = (v_idx, v_cnt)
    has_image = bool(data.get('has_image'))
    think = bool(data.get('think', False))
    model = data.get('model') or CRITIQUE_MODEL
    negative_prompt = (data.get('negative_prompt') or '').strip() or None
    return _vlm_job_start(_run_boost, model, prompt, _boost_family(has_image),
                          _model_type_string(), level, think, variant, negative_prompt)


@app.route('/boost/<cid>')
def boost_result(cid):
    """Poll for an async boost started by POST /boost."""
    return _vlm_job_status(cid)


def _warm_critique_model():
    """Pull the critique VLM into ollama's memory at server startup so the
    edit loop's first critique doesn't pay the multi-minute cold load.

    Skipped entirely under the default VLM_KEEP_ALIVE="0": preloading a model
    that unloads the moment it is used would hold ~5GB of VRAM away from the
    diffusion pipeline for no benefit. Warming only makes sense when the model
    is configured to stay resident.
    """
    global _vlm_warming
    # Imported here, matching the other edit_loop uses in this file — it pulls
    # in PIL/requests and is not needed unless a VLM path actually runs.
    import edit_loop
    if edit_loop.vlm_dialect(OLLAMA_URL) != 'ollama':
        # An OpenAI-compatible server holds its model resident for its own
        # lifetime; there is nothing for a client to preload, and /api/generate
        # doesn't exist there.
        print(f"Critique model warm-up skipped ({OLLAMA_URL} is not ollama; "
              "the model is resident on the serving host).")
        return
    if edit_loop.KEEP_ALIVE in ('0', 0, '', None):
        print("Critique model warm-up skipped (VLM_KEEP_ALIVE=0: the vision "
              "model loads on demand and releases its VRAM after each call).")
        return
    import urllib.request
    model = edit_loop.resolve_vlm_model(OLLAMA_URL, CRITIQUE_MODEL)
    payload = json.dumps({"model": model, "stream": False,
                          "keep_alive": edit_loop.KEEP_ALIVE}).encode()
    req = urllib.request.Request(f"{OLLAMA_URL}/api/generate", data=payload,
                                 headers={"Content-Type": "application/json"})
    _vlm_warming = True
    try:
        with urllib.request.urlopen(req, timeout=600) as r:
            r.read()
        print(f"Critique model {model} loaded in ollama.")
    except Exception as e:
        print(f"Critique model warm-up skipped: {e}")
    finally:
        _vlm_warming = False
        _vlm_state["ts"] = 0.0  # re-check residency on the next /status


@app.route('/loop-strip', methods=['POST'])
def loop_strip():
    """Finish an edit-loop run: preserve every iteration image in `.saved`
    (so Archive/Delete Today don't remove them) and compose a film strip of
    the reference plus each edit in sequence, saved as a regular output so it
    appears in history."""
    try:
        result = _api_build_loop_strip(request.get_json(silent=True) or {})
    except ApiError as e:
        return jsonify({'success': False, 'error': e.message}), e.status
    return jsonify(dict(result, success=True))


def _api_build_loop_strip(data):
    """Preserve an edit loop's iterations in .saved and compose the film strip.
    Returns {filename, kept}. A loop run in hidden mode keeps both the
    preserved copies and the strip inside .hidden/."""
    from edit_loop import build_film_strip

    hidden = _hidden_requested(data.get('hidden'))
    # Only the `.hidden/` prefix survives normalization; anything else is
    # reduced to a bare name, as it always was.
    filenames = [f if (f or '').startswith(HIDDEN_PREFIX) else os.path.basename(f or '')
                 for f in (data.get('filenames') or [])]
    filenames = [f for f in filenames if f.endswith('.png')]
    if not filenames:
        raise ApiError('filenames is required', 400, 'invalid_request')
    direction = (data.get('direction') or '').strip()
    prompts = data.get('prompts') or []

    frames = []
    ref_b64 = data.get('ref_image') or ''
    if ref_b64:
        try:
            if ref_b64.startswith('data:'):
                ref_b64 = ref_b64.split(',', 1)[1]
            frames.append(('input', Image.open(io.BytesIO(base64.b64decode(ref_b64))).convert('RGB')))
        except Exception:
            raise ApiError('ref_image is not a decodable base64 image', 400, 'invalid_request')

    out_dir = _output_dir(hidden)
    saved_dir = os.path.join(out_dir, '.saved')
    os.makedirs(saved_dir, exist_ok=True)
    for i, fn in enumerate(filenames, start=1):
        path = _resolve_output_name(fn)
        if not os.path.isfile(path):
            raise ApiError(f'unknown image {fn}', 404, 'not_found')
        name = os.path.basename(fn)
        frames.append((str(i), Image.open(path).convert('RGB')))
        shutil.copy2(path, os.path.join(saved_dir, name))
        sidecar = name.rsplit('.', 1)[0] + '.prompt'
        src_sidecar = os.path.join(os.path.dirname(path), sidecar)
        if os.path.isfile(src_sidecar):
            shutil.copy2(src_sidecar, os.path.join(saved_dir, sidecar))

    strip = build_film_strip(frames)
    strip_name = f"{_output_prefix()}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_editloop_strip.png"
    strip_path = os.path.join(out_dir, strip_name)
    strip.save(strip_path)
    sidecar_path = os.path.join(out_dir, strip_name.rsplit('.', 1)[0] + '.prompt')
    with open(sidecar_path, 'w') as f:
        f.write(f"# Prompt: Edit loop film strip: {direction}\n")
        for i, fn in enumerate(filenames):
            p = prompts[i] if i < len(prompts) else ''
            f.write(f"# Iteration {i + 1}: {fn} — {p}\n")
    # The strip itself survives housekeeping too.
    shutil.copy2(strip_path, os.path.join(saved_dir, strip_name))
    shutil.copy2(sidecar_path, os.path.join(saved_dir, os.path.basename(sidecar_path)))

    return {'filename': _output_name(strip_name, hidden), 'kept': filenames}


@app.route('/history')
def history():
    """Return today's generated images, newest first."""
    return jsonify({'images': _api_history(_hidden_requested(request.args.get('hidden')))})


def _api_history(hidden=False):
    """Today's top-level generated PNGs, newest first, each with the prompt
    read from its .prompt sidecar. Read errors are logged, not raised — a
    listing is best-effort. `hidden` lists the .hidden subdir instead, which is
    the only way its contents ever reach a client."""
    today = datetime.now().strftime("%Y%m%d")
    source_dir = _output_dir(hidden)
    images = []
    try:
        for filename in os.listdir(source_dir):
            if not filename.endswith('.png'): continue
            parts = filename.split('_')
            if len(parts) >= 3 and parts[1] == today:
                time_str = parts[2]
                display_time = f"{time_str[:2]}:{time_str[2:4]}:{time_str[4:6]}" if len(time_str) == 6 else time_str
                prompt = None
                prompt_file = os.path.join(source_dir, filename.rsplit('.', 1)[0] + '.prompt')
                if os.path.exists(prompt_file):
                    try:
                        with open(prompt_file, 'r') as f:
                            for line in f:
                                if line.startswith('# Prompt: '):
                                    prompt = line[10:].strip()
                                    break
                    except Exception: pass
                images.append({'filename': _output_name(filename, hidden), 'time': display_time, 'prompt': prompt, 'sort_key': parts[2] if len(parts) >= 3 else '000000'})
        images.sort(key=lambda x: x['sort_key'], reverse=True)
        for img in images: del img['sort_key']
    except Exception as e:
        print(f"Error reading history: {e}")
    return images


@app.route('/archive', methods=['POST'])
def archive_today():
    try:
        moved = _api_archive_today(
            _hidden_requested((request.get_json(silent=True) or {}).get('hidden')))
    except ApiError as e:
        # Legacy quirk preserved: this route has always answered 200 with
        # success:false on an I/O failure. The REST layer returns a real 500.
        return jsonify({'success': False, 'error': e.message})
    return jsonify({'success': True, 'moved': moved})


def _api_archive_today(hidden=False):
    """Move today's top-level files into web-generated/archive/. Returns the
    number moved. In hidden mode both sides stay inside .hidden/ — archiving
    must not lift concealed output into the visible tree."""
    today = datetime.now().strftime("%Y%m%d")
    source_dir = _output_dir(hidden)
    archive_dir = os.path.join(source_dir, "archive")
    os.makedirs(archive_dir, exist_ok=True)
    moved = 0
    try:
        for filename in os.listdir(source_dir):
            filepath = os.path.join(source_dir, filename)
            if not os.path.isfile(filepath): continue
            parts = filename.split('_')
            if len(parts) >= 3 and parts[1] == today:
                shutil.move(filepath, os.path.join(archive_dir, filename))
                moved += 1
    except Exception as e:
        raise ApiError(str(e), 500, 'io_error')
    return moved


@app.route('/delete', methods=['POST'])
def delete_today():
    """Permanently delete today's generated files (PNG + .prompt).

    If a single filename is provided in the JSON body, only that image (and its
    sidecar .prompt) is deleted. Otherwise all of today's files are removed.
    """
    body = request.get_json(silent=True) or {}
    try:
        deleted = _api_delete_today(body.get('filename'),
                                    _hidden_requested(body.get('hidden')))
    except ApiError as e:
        if e.status == 500:
            # Legacy quirk preserved: I/O failures answered 200 here.
            return jsonify({'success': False, 'error': e.message})
        return jsonify({'success': False, 'error': e.message}), e.status
    return jsonify({'success': True, 'deleted': deleted})


def _api_delete_today(target=None, hidden=False):
    """Permanently delete one of today's images (plus its sidecar), or all of
    today's files when `target` is None. Returns the count of files removed.
    A `.hidden/`-prefixed target deletes from the hidden subdir; `hidden`
    scopes the delete-everything case to it."""
    today = datetime.now().strftime("%Y%m%d")
    deleted = 0

    def _remove_pair(png_path):
        nonlocal deleted
        if os.path.isfile(png_path):
            os.remove(png_path)
            deleted += 1
        prompt_path = png_path.rsplit('.', 1)[0] + '.prompt'
        if os.path.isfile(prompt_path):
            os.remove(prompt_path)

    if target:
        # Guard against path traversal and ensure it's a today image
        target_path = _resolve_output_name(target)
        parts = os.path.basename(target).split('_')
        if len(parts) < 3 or parts[1] != today:
            raise ApiError('Not a today image', 400, 'invalid_request')
    try:
        if target:
            _remove_pair(target_path)
        else:
            source_dir = _output_dir(hidden)
            for filename in os.listdir(source_dir):
                filepath = os.path.join(source_dir, filename)
                if not os.path.isfile(filepath): continue
                parts = filename.split('_')
                if len(parts) >= 3 and parts[1] == today:
                    os.remove(filepath)
                    deleted += 1
    except Exception as e:
        raise ApiError(str(e), 500, 'io_error')
    return deleted


@app.route('/save-hidden', methods=['POST'])
def save_hidden():
    """Copy an image (and its .prompt sidecar) into the .saved subdir.

    The `.saved` dir lives inside OUTPUT_DIR but is excluded from listings,
    archiving, and delete-today (those only iterate top-level files), so saving
    an image here preserves it independently of the day's housekeeping.
    """
    try:
        saved = _api_save_hidden((request.get_json(silent=True) or {}).get('filename'))
    except ApiError as e:
        return jsonify({'success': False, 'error': e.message}), e.status
    return jsonify({'success': True, 'saved': saved})


def _api_save_hidden(target):
    """Copy an output image and its sidecar into .saved/, out of reach of
    archive and delete-today. Returns the filename. A hidden image is saved
    into .hidden/.saved/ — surviving housekeeping must not mean surfacing."""
    src = _resolve_output_name(target)
    if not os.path.isfile(src):
        raise ApiError('File not found', 404, 'not_found')

    name = os.path.basename(target)
    saved_dir = os.path.join(os.path.dirname(src), '.saved')
    os.makedirs(saved_dir, exist_ok=True)
    try:
        shutil.copy2(src, os.path.join(saved_dir, name))
        sidecar = name.rsplit('.', 1)[0] + '.prompt'
        src_sidecar = os.path.join(os.path.dirname(src), sidecar)
        if os.path.isfile(src_sidecar):
            shutil.copy2(src_sidecar, os.path.join(saved_dir, sidecar))
    except Exception as e:
        raise ApiError(str(e), 500, 'io_error')
    return target


# Mount /api/v1 once every _api_* core function above exists. The live module
# object is handed over rather than imported on the far side: this file runs as
# __main__, so `import web_server` there would execute it a second time and give
# the REST layer its own queue, its own model, and a second API-key check.
rest_api.init_app(app, sys.modules[__name__])
PUBLIC_ENDPOINTS.extend(rest_api.PUBLIC_ENDPOINTS)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="FLUX Web Server")
    parser.add_argument("--local-encoder", action="store_true", help="Use local text encoder instead of remote API")
    parser.add_argument("--full-model", action="store_true", help="Use full FLUX model instead of 4-bit quantized")
    parser.add_argument("--gguf", type=str, choices=["bf16", "q8", "q4"], default=None, help="Use GGUF model")
    parser.add_argument("--flux2", action="store_true", help="Use FLUX.2 model")
    parser.add_argument("--schnell", action="store_true", help="Use FLUX.1-schnell")
    parser.add_argument("--klein", action="store_true", help="Use FLUX.2-klein (9B) instead of FLUX.2-dev (32B). Implies --flux2 --full-model")
    parser.add_argument("--klein-4b", action="store_true", help="Use FLUX.2-klein-4B instead of the 9B. Implies --klein --flux2 --full-model")
    parser.add_argument("--quantize-encoder", action="store_true", help="FLUX.2 full/klein: load the text encoder 4-bit NF4 (transformer stays bf16). For discrete-VRAM cards where both won't fit in bf16")
    parser.add_argument("--vae-tiling", nargs='?', const='always', default='auto',
                        choices=['auto', 'always', 'off'],
                        help="Tiled VAE decoding, which bounds the peak memory of the full-resolution final decode. 'auto' (default) tiles only above VAE_TILING_THRESHOLD_MP (1.9MP), since tiling costs ~50%% below that and rescues generations that would otherwise stall on the last step. Bare --vae-tiling means 'always'")
    parser.add_argument("--turbo", action="store_true", default=None, help="Enable turbo LoRA")
    parser.add_argument("--no-turbo", action="store_true", help="Disable turbo LoRA")
    parser.add_argument("--uncensored", action="store_true", help="Load the uncensored LoRA (FLUX.1 only)")
    parser.add_argument("--kontext", action="store_true", help="Use FLUX.1 Kontext, an instruction-based image editor (4-bit; add --full-model for full bf16)")
    parser.add_argument("--sdxl", nargs='?', const='', default=None, metavar='MODEL',
                        help="Serve a Stable Diffusion XL checkpoint instead of FLUX. "
                             "Optional MODEL is an HF repo id, local diffusers dir, or single-file "
                             ".safetensors (e.g. a Civitai download); default is "
                             "a photoreal merge (or the SD_MODEL env var). "
                             "Enables negative prompts; ignores the FLUX model flags.")
    parser.add_argument("--compile", action="store_true", help="torch.compile the transformer after load: the first generation per resolution is much slower (compilation), later ones ~10-25%% faster. Best when generating at consistent resolutions")
    parser.add_argument("--port", type=int, default=PORT, help=f"Port (default: {PORT})")
    args = parser.parse_args()

    if args.klein_4b:
        args.klein = True
    if args.klein:
        args.flux2 = True
        args.full_model = True
    _full_model, _gguf_quant, _flux2, _schnell, _uncensored, _klein, _klein_4b = args.full_model, args.gguf, args.flux2, args.schnell, args.uncensored, args.klein, args.klein_4b
    _kontext = args.kontext
    _local_encoder = args.local_encoder or args.full_model or args.schnell or args.uncensored or args.kontext
    if args.uncensored and not args.full_model: _full_model = True
    # Turbo LoRA is a FLUX.2-dev LoRA — don't auto-enable for klein (different
    # architecture) or the SDXL backend.
    _turbo = (args.turbo or (args.flux2 and not _klein)) and not args.no_turbo and not _SDXL_ACTIVE

    def _load_in_background():
        global _model_ready, _model_load_error, _model_load_status
        try:
            if _SDXL_ACTIVE:
                _model_load_status = "loading SDXL model"
                print("Loading SDXL...")
                load_model(model_id=args.sdxl or None, vae_tiling=args.vae_tiling)
            else:
                _model_name = "FLUX.1-Kontext" if _kontext else ("FLUX.2" if _flux2 else "FLUX.1")
                _model_load_status = f"loading {_model_name} model"
                print(f"Loading {_model_name}...")
                load_model(local_encoder=_local_encoder, full_model=_full_model, gguf_quant=_gguf_quant, flux2=_flux2, schnell=_schnell, for_lora=_uncensored, klein=_klein, klein_4b=_klein_4b, kontext=_kontext, quantize_encoder=args.quantize_encoder, vae_tiling=args.vae_tiling)
            if _turbo:
                _model_load_status = "loading turbo LoRA"
                load_turbo_lora()
            if _uncensored and not _SDXL_ACTIVE:
                _model_load_status = "loading U-LoRA"
                load_uncensored_lora()
            if args.compile:
                # Wraps the transformer; actual compilation happens lazily on
                # the first forward pass (so the first generation is slow).
                _model_load_status = "wrapping transformer with torch.compile"
                flux_core.compile_pipeline()
            _model_load_status = "ready"
            _model_ready = True
            print("Model ready.")
            # Resume an in-flight multi-model run (this launch may BE the
            # switch it requested): queue its job for this config, or keep
            # switching if this config's turn is already done.
            _multi_run_advance()
        except Exception as e:
            _model_load_error = str(e)
            _model_load_status = "error"
            print(f"FATAL: model load failed: {e}")
            traceback.print_exc()

    _model_load_start_ts = time.perf_counter()
    threading.Thread(target=_load_in_background, daemon=True).start()
    if _flux2 and _full_model:
        # Full FLUX.2 (32B bf16 + Mistral3 encoder) fills nearly all unified
        # memory during load; preloading the critique VLM alongside it has
        # OOM-killed the box. Let ollama cold-load on the first real VLM call.
        print("Skipping critique model preload (full FLUX.2 config; VLM loads on first use).")
    else:
        threading.Thread(target=_warm_critique_model, daemon=True,
                         name='critique-warmup').start()
    _start_queue_worker()
    _start_note_reporter(args.port)
    print(f"\nStarting web server on http://0.0.0.0:{args.port} (model loading in background)")
    app.run(host='0.0.0.0', port=args.port, threaded=True)
