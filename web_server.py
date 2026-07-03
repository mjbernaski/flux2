import os
import argparse
import hmac
import json
import subprocess
import threading
import time
import base64
import io
import socket
import shutil
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional
import random
import uuid
from flask import Flask, request, jsonify, send_from_directory, Response, stream_with_context
from PIL import Image, ImageDraw, ImageFont
from dotenv import load_dotenv

# Load .env file if it exists
load_dotenv()

# Version number - update this when releasing new versions
VERSION = "1.2.0"

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

@app.before_request
def require_auth():
    # Allow the main page, images, and the readiness probe to load without auth.
    # Images are served with random filenames which provides basic security;
    # /ready must be reachable before the user can enter their API key.
    if request.endpoint in ['index', 'static', 'serve_image', 'ready']:
        return

    if not check_auth():
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
_kontext = False

# Configuration
OUTPUT_DIR = "web-generated"
PORT = 2222

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Queue configuration
QUEUE_MAX_SIZE = 10
RECENT_DONE_MAX = 10

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
    5: "FLUX.1 + Uncensored LoRA",
    6: "FLUX.2 4-bit BNB",
    7: "FLUX.2 Full + Turbo",
    8: "FLUX.2 Full (no Turbo)",
    9: "FLUX.2-klein-9B",
    10: "FLUX.1 Kontext (editor)",
    11: "FLUX.1 Kontext Full (editor, bf16)",
    12: "FLUX.1 Kontext Full + Uncensored LoRA",
    13: "SDXL Uncensored (LUSTIFY!)",
}
SWITCH_EXIT_CODE = 86
SWITCH_CONFIG_FILE = ".next_config"
try:
    _current_config = int(os.environ.get("FLUX_CONFIG", ""))
except ValueError:
    _current_config = None
if _current_config not in SERVER_CONFIGS:
    _current_config = None

# Edit-loop critique (local ollama vision model; see /critique). qwen3.6 is
# the strongest local VLM on this box — slower than gemma4:e2b but its
# judgments and revised prompts are markedly better.
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434")
CRITIQUE_MODEL = os.environ.get("CRITIQUE_MODEL", "qwen3.6:latest")

PREVIEW_FILENAME = "_preview_current.png"
PREVIEW_MIN_INTERVAL_S = 0.75  # throttle: skip decode if last preview was this recent

# GPU power draw for the UI's corner wattage badge. nvidia-smi takes ~100ms
# per call, so the reading is cached and refreshed at most every 2s even
# though /status is polled more often.
_POWER_CACHE_S = 2.0
_power_state = {"watts": None, "ts": 0.0}


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
    step_times: list = field(default_factory=list)  # per-step durations (s) of the current image
    generation_time: float = 0.0

    @property
    def prompt(self) -> str:
        return (self.params.get('prompt') or '').strip()

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
            'preview': self.preview,
            'preview_step': self.preview_step,
            'preview_ts': self.preview_ts,
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

# Orientation presets (width, height) at 1K base
ORIENTATIONS_1K = {
    'square': (1024, 1024),
    'portrait': (768, 1344),
    'landscape': (1360, 768),  # 16:9
    'widescreen': (1568, 672),  # ~21:9 extra-wide, ~1 MP
    'extra-tall': (672, 1568),  # ~9:21 mirror of widescreen, ~1 MP
}

SIZES = {
    '0.75mp': 0.75,
    '1mp': 1.0,
    '2mp': 2.0,
}

# The web UI lives in static/ (index.html + app.css + app.js); it was
# previously embedded here as one giant string. See static/README note.


def _run_job(job: Job):
    """Execute one generation job, writing progress/results into the Job object."""
    data = job.params
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
            if is_final or (now - preview_state["last_decode"]) >= PREVIEW_MIN_INTERVAL_S:
                latents = callback_kwargs.get("latents")
                preview_img = flux_core.decode_latents_to_preview(
                    pipe_obj, latents, height, width
                )
                if preview_img is not None:
                    try:
                        preview_img.save(os.path.join(OUTPUT_DIR, PREVIEW_FILENAME))
                        job.preview = PREVIEW_FILENAME
                        job.preview_step = step_index + 1
                        job.preview_ts = int(time.time() * 1000)
                        preview_state["last_decode"] = now
                    except Exception as e:
                        print(f"[preview] save failed: {e}", flush=True)
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
                output_path = os.path.join(OUTPUT_DIR, output_filename)
                image.save(output_path)
                timings['save'] = time.perf_counter() - t_save
                save_prompt_file(output_path, prompt, prompt, width, height, used_seed, steps, timings, g_val, s_val if input_image else None)
                
                img_data = {
                    "filename": output_filename,
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

        # Create composite
        # Calculate cell size based on aspect ratio
        aspect_ratio = width / height
        if width >= height:
            cell_width = 256
            cell_height = int(round(cell_width / aspect_ratio))
        else:
            cell_height = 256
            cell_width = int(round(cell_height * aspect_ratio))

        n_rows, n_cols = len(grid_cells), len(grid_cells[0])
        composite = Image.new('RGB', (n_cols * cell_width, n_rows * cell_height), (32, 32, 32))
        for row_idx, row_images in enumerate(grid_cells):
            for col_idx, cell in enumerate(row_images):
                if cell is None: continue # Skip empty diagonal cells
                img, _seq = cell
                img_small = img.resize((cell_width, cell_height), Image.Resampling.LANCZOS)
                composite.paste(img_small, (col_idx * cell_width, row_idx * cell_height))

        # Overlay the generation-sequence number on each populated cell
        draw = ImageDraw.Draw(composite)
        font_size = max(14, cell_height // 12)
        font = None
        for font_path in (
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        ):
            if os.path.exists(font_path):
                try:
                    font = ImageFont.truetype(font_path, font_size)
                    break
                except Exception:
                    font = None
        if font is None:
            font = ImageFont.load_default()
        for row_idx, row_images in enumerate(grid_cells):
            for col_idx, cell in enumerate(row_images):
                if cell is None: continue
                _img, seq = cell
                label = str(seq)
                pad = 4
                bbox = draw.textbbox((0, 0), label, font=font)
                tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
                x0 = col_idx * cell_width + 6
                y0 = row_idx * cell_height + 6
                draw.rectangle(
                    [x0 - pad, y0 - pad, x0 + tw + pad, y0 + th + pad],
                    fill=(0, 0, 0),
                )
                draw.text((x0 - bbox[0], y0 - bbox[1]), label, fill=(255, 255, 255), font=font)

        comp_filename = f"{_output_prefix()}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_spectrum_grid.png"
        composite.save(os.path.join(OUTPUT_DIR, comp_filename))
        job.composite = comp_filename
    else:
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
            output_path = os.path.join(OUTPUT_DIR, output_filename)
            image.save(output_path)
            timings['save'] = time.perf_counter() - t_save
            save_prompt_file(output_path, prompt, prompt, width, height, used_seed, steps, timings, guidance_scale, None if mask_image is not None else (strength if input_image else None))

            img_data = {
                'filename': output_filename,
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
            # Finished jobs sit in _recent_done for a while; drop the (large)
            # base64 image payloads so they don't stay resident in memory.
            job.params.pop('input_image', None)
            job.params.pop('input_images', None)
            job.params.pop('mask_image', None)
            with _queue_cv:
                _running_job = None
                _recent_done.insert(0, job)
                del _recent_done[RECENT_DONE_MAX:]


def _start_queue_worker():
    global _queue_worker_thread
    if _queue_worker_thread is None or not _queue_worker_thread.is_alive():
        _queue_worker_thread = threading.Thread(target=_queue_worker, daemon=True, name="queue-worker")
        _queue_worker_thread.start()


@app.route('/')
def index():
    return app.send_static_file('index.html')


@app.route('/ready')
def ready():
    elapsed = time.perf_counter() - _model_load_start_ts if _model_load_start_ts else 0.0
    return jsonify({
        'ready': _model_ready,
        'error': _model_load_error,
        'status': _model_load_status,
        'elapsed_s': round(elapsed, 1),
    })


def _validate_generate_params(data):
    """Validate and normalize a /generate request body in place.

    Returns an error string for a 400 response, or None if the request is
    valid. Doing this at the API boundary means bad input fails fast with a
    clear message instead of surfacing later as an opaque failed job.
    """
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
    if len(imgs) > MAX_REFERENCE_IMAGES:
        return f"at most {MAX_REFERENCE_IMAGES} reference images are supported"
    for i, b64 in enumerate(imgs):
        if not isinstance(b64, str) or not _decodable(b64):
            return f"input_images[{i}] is not a decodable base64 image"
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


@app.route('/generate', methods=['POST'])
def generate():
    if not _model_ready:
        return jsonify({'success': False, 'error': 'Model still loading. Please wait.'}), 503

    data = request.json or {}
    prompt = (data.get('prompt') or '').strip()
    if not prompt:
        return jsonify({'success': False, 'error': 'prompt is required'}), 400

    error = _validate_generate_params(data)
    if error:
        return jsonify({'success': False, 'error': error}), 400

    with _queue_cv:
        if len(_pending) >= QUEUE_MAX_SIZE:
            return jsonify({
                'success': False,
                'error': f'Queue is full ({QUEUE_MAX_SIZE} max). Cancel a queued job or wait.',
            }), 429
        job = Job(id=uuid.uuid4().hex[:12], params=data, submitted_at=time.time())
        _pending.append(job)
        position = len(_pending)  # 1-based position of this job in the pending list
        _queue_cv.notify()

    return jsonify({'success': True, 'job_id': job.id, 'position': position})


@app.route('/images/<filename>')
def serve_image(filename):
    return send_from_directory(OUTPUT_DIR, filename)


@app.route('/status')
def status():
    with _queue_cv:
        running = _running_job.full() if _running_job else None
        queued = [j.summary() for j in _pending]
        recent = [j.full() for j in _recent_done]
    return jsonify({
        'running': running,
        'queued': queued,
        'recent_done': recent,
        'queue_max_size': QUEUE_MAX_SIZE,
        'power_w': _gpu_power_watts(),
    })


@app.route('/reset', methods=['POST'])
def reset_recent():
    """Clear the in-memory list of recently-completed jobs so the client's
    results view stays empty across page reloads."""
    with _queue_cv:
        _recent_done.clear()
    return jsonify({'success': True})


@app.route('/jobs/<job_id>/cancel', methods=['POST'])
def cancel_job(job_id):
    with _queue_cv:
        for i, j in enumerate(_pending):
            if j.id == job_id:
                j.state = 'canceled'
                j.finished_at = time.time()
                del _pending[i]
                _recent_done.insert(0, j)
                del _recent_done[RECENT_DONE_MAX:]
                return jsonify({'success': True, 'message': f'Job {job_id} canceled'})
        if _running_job and _running_job.id == job_id:
            # Interrupt: the generation loop checks this flag at every step and
            # raises JobCanceled; the worker then marks the job canceled. Any
            # batch images already finished are kept.
            _running_job.cancel_requested = True
            return jsonify({'success': True, 'message': f'Job {job_id} is stopping'})
    return jsonify({'success': False, 'error': 'Job not found'}), 404


@app.route('/configs')
def configs():
    """The launcher's model-config menu, for the UI's model switcher.

    `switchable` is false when the server was started directly (no
    run_server.sh supervisor), in which case /switch-model is unavailable
    and the UI hides the control.
    """
    return jsonify({
        'configs': [{'id': k, 'label': v} for k, v in SERVER_CONFIGS.items()],
        'current': _current_config,
        'switchable': _current_config is not None,
    })


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
    if _current_config is None:
        return jsonify({
            'success': False,
            'error': 'Model switching requires launching via run_server.sh (no supervisor detected).',
        }), 400

    data = request.json or {}
    try:
        target = int(data.get('config'))
    except (TypeError, ValueError):
        return jsonify({'success': False, 'error': 'config must be an integer'}), 400
    if target not in SERVER_CONFIGS:
        return jsonify({'success': False, 'error': f'config must be one of {sorted(SERVER_CONFIGS)}'}), 400
    if target == _current_config:
        return jsonify({'success': False, 'error': 'Already running this configuration'}), 400

    with _queue_cv:
        if _running_job is not None or _pending:
            return jsonify({
                'success': False,
                'error': 'Jobs are running or queued. Interrupt/cancel them before switching models.',
            }), 409

    with open(SWITCH_CONFIG_FILE, 'w') as f:
        f.write(str(target))
    print(f"[switch] restarting into config {target} ({SERVER_CONFIGS[target]})", flush=True)
    # Give Flask a moment to flush this response before the process exits.
    threading.Timer(0.5, lambda: os._exit(SWITCH_EXIT_CODE)).start()
    return jsonify({'success': True, 'switching_to': SERVER_CONFIGS[target]})


@app.route('/model-info')
def model_info():
    flux_name = f"FLUX.{flux_core._flux_version}"
    variant = "-klein" if _klein else "-dev"
    if _SDXL_ACTIVE:
        model_type = f"SDXL ({flux_core.model_name()})"
    elif _kontext:
        kontext_prec = "full bf16" if _full_model else "4-bit"
        model_type = f"FLUX.1-Kontext (editor, {kontext_prec})"
    elif _schnell:
        model_type = f"{flux_name}-schnell (4-step)"
    elif _gguf_quant:
        model_type = f"{flux_name}-dev GGUF {_gguf_quant.upper()}"
    elif _full_model:
        model_type = f"{flux_name}{variant} (full)"
    else:
        model_type = f"{flux_name}-dev-bnb-4bit"
    encoder_type = ("local CLIP encoders" if _SDXL_ACTIVE
                    else "local encoder" if _local_encoder else "remote encoder")
    turbo_str = " + Turbo" if flux_core._turbo_enabled else ""
    uncensored_str = " + Uncensored" if flux_core._uncensored_enabled else ""
    return jsonify({
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
        'hostname': socket.gethostname(),
        'version': VERSION,
        'description': f"{model_type}{turbo_str}{uncensored_str} with {encoder_type}"
    })


@app.route('/critique', methods=['POST'])
def critique():
    """Compare an edit output against its reference and propose a revised
    instruction — the "look at the output" step of the edit loop (UI panel
    and edit_loop.py). Vision critique runs on the local ollama daemon
    (OLLAMA_URL / CRITIQUE_MODEL env vars); when it's unavailable the
    response falls back to pixel-metric heuristics."""
    from edit_loop import (vlm_critique, edit_metrics, heuristic_revision,
                           describe_metrics)

    data = request.json or {}
    direction = (data.get('direction') or '').strip()
    prompt = (data.get('prompt') or direction).strip()
    out_filename = os.path.basename(data.get('output_filename') or '')
    ref_b64 = data.get('ref_image') or ''
    if not direction or not out_filename or not ref_b64:
        return jsonify({'success': False,
                        'error': 'direction, ref_image, and output_filename are required'}), 400

    out_path = os.path.join(OUTPUT_DIR, out_filename)
    if not os.path.exists(out_path):
        return jsonify({'success': False, 'error': f'unknown output image {out_filename}'}), 404
    try:
        if ref_b64.startswith('data:'):
            ref_b64 = ref_b64.split(',', 1)[1]
        reference = Image.open(io.BytesIO(base64.b64decode(ref_b64))).convert('RGB')
    except Exception:
        return jsonify({'success': False, 'error': 'ref_image is not a decodable base64 image'}), 400

    output = Image.open(out_path).convert('RGB')
    metrics = edit_metrics(reference, output)
    model = data.get('model') or CRITIQUE_MODEL
    style = ("description" if getattr(flux_core, 'OUTPUT_PREFIX', '') == 'sdxl'
             else "instruction")
    # Prompt trajectory from earlier iterations ({prompt, applied, score,
    # critique} dicts) so the critic doesn't re-propose failed phrasings.
    history = data.get('history') if isinstance(data.get('history'), list) else []
    result = vlm_critique(model, direction, prompt, reference, output, metrics,
                          ollama_url=OLLAMA_URL, style=style, history=history)
    if result:
        return jsonify({'success': True, 'vlm': True, 'metrics': metrics,
                        'metrics_text': describe_metrics(metrics),
                        'applied': bool(result.get('applied')),
                        'score': result.get('score'),
                        'critique': result.get('critique', ''),
                        'revised_prompt': result.get('revised_prompt')})
    return jsonify({'success': True, 'vlm': False, 'metrics': metrics,
                    'metrics_text': describe_metrics(metrics),
                    'applied': None,
                    'score': None,
                    'critique': 'Vision model unavailable — revision based on pixel metrics only.',
                    'revised_prompt': heuristic_revision(direction, prompt, metrics)})


@app.route('/loop-strip', methods=['POST'])
def loop_strip():
    """Finish an edit-loop run: preserve every iteration image in `.hidden`
    (so Archive/Delete Today don't remove them) and compose a film strip of
    the reference plus each edit in sequence, saved as a regular output so it
    appears in history."""
    from edit_loop import build_film_strip

    data = request.get_json(silent=True) or {}
    filenames = [os.path.basename(f or '') for f in (data.get('filenames') or [])]
    filenames = [f for f in filenames if f.endswith('.png')]
    if not filenames:
        return jsonify({'success': False, 'error': 'filenames is required'}), 400
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
            return jsonify({'success': False, 'error': 'ref_image is not a decodable base64 image'}), 400

    hidden_dir = os.path.join(OUTPUT_DIR, '.hidden')
    os.makedirs(hidden_dir, exist_ok=True)
    for i, fn in enumerate(filenames, start=1):
        path = os.path.join(OUTPUT_DIR, fn)
        if not os.path.isfile(path):
            return jsonify({'success': False, 'error': f'unknown image {fn}'}), 404
        frames.append((str(i), Image.open(path).convert('RGB')))
        shutil.copy2(path, os.path.join(hidden_dir, fn))
        sidecar = fn.rsplit('.', 1)[0] + '.prompt'
        if os.path.isfile(os.path.join(OUTPUT_DIR, sidecar)):
            shutil.copy2(os.path.join(OUTPUT_DIR, sidecar), os.path.join(hidden_dir, sidecar))

    strip = build_film_strip(frames)
    strip_name = f"{_output_prefix()}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_editloop_strip.png"
    strip.save(os.path.join(OUTPUT_DIR, strip_name))
    sidecar_path = os.path.join(OUTPUT_DIR, strip_name.rsplit('.', 1)[0] + '.prompt')
    with open(sidecar_path, 'w') as f:
        f.write(f"# Prompt: Edit loop film strip: {direction}\n")
        for i, fn in enumerate(filenames):
            p = prompts[i] if i < len(prompts) else ''
            f.write(f"# Iteration {i + 1}: {fn} — {p}\n")
    # The strip itself survives housekeeping too.
    shutil.copy2(os.path.join(OUTPUT_DIR, strip_name), os.path.join(hidden_dir, strip_name))
    shutil.copy2(sidecar_path, os.path.join(hidden_dir, os.path.basename(sidecar_path)))

    return jsonify({'success': True, 'filename': strip_name, 'kept': filenames})


@app.route('/history')
def history():
    """Return today's generated images, newest first."""
    today = datetime.now().strftime("%Y%m%d")
    images = []
    try:
        for filename in os.listdir(OUTPUT_DIR):
            if not filename.endswith('.png'): continue
            parts = filename.split('_')
            if len(parts) >= 3 and parts[1] == today:
                time_str = parts[2]
                display_time = f"{time_str[:2]}:{time_str[2:4]}:{time_str[4:6]}" if len(time_str) == 6 else time_str
                prompt = None
                prompt_file = os.path.join(OUTPUT_DIR, filename.rsplit('.', 1)[0] + '.prompt')
                if os.path.exists(prompt_file):
                    try:
                        with open(prompt_file, 'r') as f:
                            for line in f:
                                if line.startswith('# Prompt: '):
                                    prompt = line[10:].strip()
                                    break
                    except Exception: pass
                images.append({'filename': filename, 'time': display_time, 'prompt': prompt, 'sort_key': parts[2] if len(parts) >= 3 else '000000'})
        images.sort(key=lambda x: x['sort_key'], reverse=True)
        for img in images: del img['sort_key']
    except Exception as e:
        print(f"Error reading history: {e}")
    return jsonify({'images': images})


@app.route('/archive', methods=['POST'])
def archive_today():
    today = datetime.now().strftime("%Y%m%d")
    archive_dir = os.path.join(OUTPUT_DIR, "archive")
    os.makedirs(archive_dir, exist_ok=True)
    moved = 0
    try:
        for filename in os.listdir(OUTPUT_DIR):
            filepath = os.path.join(OUTPUT_DIR, filename)
            if not os.path.isfile(filepath): continue
            parts = filename.split('_')
            if len(parts) >= 3 and parts[1] == today:
                shutil.move(filepath, os.path.join(archive_dir, filename))
                moved += 1
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})
    return jsonify({'success': True, 'moved': moved})


@app.route('/delete', methods=['POST'])
def delete_today():
    """Permanently delete today's generated files (PNG + .prompt).

    If a single filename is provided in the JSON body, only that image (and its
    sidecar .prompt) is deleted. Otherwise all of today's files are removed.
    """
    today = datetime.now().strftime("%Y%m%d")
    body = request.get_json(silent=True) or {}
    target = body.get('filename')
    deleted = 0

    def _remove_pair(png_name):
        nonlocal deleted
        png_path = os.path.join(OUTPUT_DIR, png_name)
        if os.path.isfile(png_path):
            os.remove(png_path)
            deleted += 1
        prompt_path = os.path.join(OUTPUT_DIR, png_name.rsplit('.', 1)[0] + '.prompt')
        if os.path.isfile(prompt_path):
            os.remove(prompt_path)

    try:
        if target:
            # Guard against path traversal and ensure it's a today image
            if '/' in target or '\\' in target or not target.endswith('.png'):
                return jsonify({'success': False, 'error': 'Invalid filename'}), 400
            parts = target.split('_')
            if len(parts) < 3 or parts[1] != today:
                return jsonify({'success': False, 'error': 'Not a today image'}), 400
            _remove_pair(target)
        else:
            for filename in os.listdir(OUTPUT_DIR):
                filepath = os.path.join(OUTPUT_DIR, filename)
                if not os.path.isfile(filepath): continue
                parts = filename.split('_')
                if len(parts) >= 3 and parts[1] == today:
                    os.remove(filepath)
                    deleted += 1
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})
    return jsonify({'success': True, 'deleted': deleted})


@app.route('/save-hidden', methods=['POST'])
def save_hidden():
    """Copy an image (and its .prompt sidecar) into a hidden subdir.

    The `.hidden` dir lives inside OUTPUT_DIR but is excluded from listings,
    archiving, and delete-today (those only iterate top-level files), so saving
    an image here preserves it independently of the day's housekeeping.
    """
    body = request.get_json(silent=True) or {}
    target = body.get('filename')
    if not target or '/' in target or '\\' in target or not target.endswith('.png'):
        return jsonify({'success': False, 'error': 'Invalid filename'}), 400

    src = os.path.join(OUTPUT_DIR, target)
    if not os.path.isfile(src):
        return jsonify({'success': False, 'error': 'File not found'}), 404

    hidden_dir = os.path.join(OUTPUT_DIR, '.hidden')
    os.makedirs(hidden_dir, exist_ok=True)
    try:
        shutil.copy2(src, os.path.join(hidden_dir, target))
        sidecar = target.rsplit('.', 1)[0] + '.prompt'
        src_sidecar = os.path.join(OUTPUT_DIR, sidecar)
        if os.path.isfile(src_sidecar):
            shutil.copy2(src_sidecar, os.path.join(hidden_dir, sidecar))
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500
    return jsonify({'success': True, 'saved': target})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="FLUX Web Server")
    parser.add_argument("--local-encoder", action="store_true", help="Use local text encoder instead of remote API")
    parser.add_argument("--full-model", action="store_true", help="Use full FLUX model instead of 4-bit quantized")
    parser.add_argument("--gguf", type=str, choices=["bf16", "q8", "q4"], default=None, help="Use GGUF model")
    parser.add_argument("--flux2", action="store_true", help="Use FLUX.2 model")
    parser.add_argument("--schnell", action="store_true", help="Use FLUX.1-schnell")
    parser.add_argument("--klein", action="store_true", help="Use FLUX.2-klein (9B) instead of FLUX.2-dev (32B). Implies --flux2 --full-model")
    parser.add_argument("--turbo", action="store_true", default=None, help="Enable turbo LoRA")
    parser.add_argument("--no-turbo", action="store_true", help="Disable turbo LoRA")
    parser.add_argument("--uncensored", action="store_true", help="Load Lustly.ai uncensored NSFW LoRA")
    parser.add_argument("--kontext", action="store_true", help="Use FLUX.1 Kontext, an instruction-based image editor (4-bit; add --full-model for full bf16)")
    parser.add_argument("--sdxl", nargs='?', const='', default=None, metavar='MODEL',
                        help="Serve an uncensored Stable Diffusion XL checkpoint instead of FLUX. "
                             "Optional MODEL is an HF repo id, local diffusers dir, or single-file "
                             ".safetensors (e.g. a Civitai download); default is "
                             "the LUSTIFY! NSFW/SFW merge (or the SD_MODEL env var). "
                             "Enables negative prompts; ignores the FLUX model flags.")
    parser.add_argument("--compile", action="store_true", help="torch.compile the transformer after load: the first generation per resolution is much slower (compilation), later ones ~10-25%% faster. Best when generating at consistent resolutions")
    parser.add_argument("--port", type=int, default=PORT, help=f"Port (default: {PORT})")
    args = parser.parse_args()

    if args.klein:
        args.flux2 = True
        args.full_model = True
    _full_model, _gguf_quant, _flux2, _schnell, _uncensored, _klein = args.full_model, args.gguf, args.flux2, args.schnell, args.uncensored, args.klein
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
                load_model(model_id=args.sdxl or None)
            else:
                _model_name = "FLUX.1-Kontext" if _kontext else ("FLUX.2" if _flux2 else "FLUX.1")
                _model_load_status = f"loading {_model_name} model"
                print(f"Loading {_model_name}...")
                load_model(local_encoder=_local_encoder, full_model=_full_model, gguf_quant=_gguf_quant, flux2=_flux2, schnell=_schnell, for_lora=_uncensored, klein=_klein, kontext=_kontext)
            if _turbo:
                _model_load_status = "loading turbo LoRA"
                load_turbo_lora()
            if _uncensored and not _SDXL_ACTIVE:
                _model_load_status = "loading uncensored LoRA"
                load_uncensored_lora()
            if args.compile:
                # Wraps the transformer; actual compilation happens lazily on
                # the first forward pass (so the first generation is slow).
                _model_load_status = "wrapping transformer with torch.compile"
                flux_core.compile_pipeline()
            _model_load_status = "ready"
            _model_ready = True
            print("Model ready.")
        except Exception as e:
            _model_load_error = str(e)
            _model_load_status = "error"
            print(f"FATAL: model load failed: {e}")

    _model_load_start_ts = time.perf_counter()
    threading.Thread(target=_load_in_background, daemon=True).start()
    _start_queue_worker()
    print(f"\nStarting web server on http://0.0.0.0:{args.port} (model loading in background)")
    app.run(host='0.0.0.0', port=args.port, threaded=True)
