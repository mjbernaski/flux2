"""Sample client for the FLUX image generator REST API (/api/v1).

A single dependency-light module showing how to drive every major function of
the server, plus a CLI so each one is runnable as-is:

    export FLUX_API_KEY=...            # same key the server was started with
    python examples/flux_client.py health
    python examples/flux_client.py generate "a red fox in snow" --steps 30
    python examples/flux_client.py batch "a lighthouse at dusk" --count 4
    python examples/flux_client.py img2img photo.jpg "make it winter"
    python examples/flux_client.py inpaint photo.jpg mask.png "a brass telescope"
    python examples/flux_client.py describe photo.jpg
    python examples/flux_client.py boost "a castle" --level 4
    python examples/flux_client.py progress "a mountain range"
    python examples/flux_client.py history
    python examples/flux_client.py models
    python examples/flux_client.py switch 9
    python examples/flux_client.py multirun "a red fox" 9 6 1
    python examples/flux_client.py demo                  # everything read-only

Only `requests` is needed (already a server dependency).

The API is asynchronous everywhere it matters: POST /jobs returns immediately
with a job id even when the queue is empty, because generation runs on a single
worker thread. Every helper here that returns images does the polling for you;
`FluxClient.submit` is the non-blocking version if you would rather drive the
loop yourself.
"""

import argparse
import base64
import io
import mimetypes
import os
import sys
import time

import requests

DEFAULT_BASE_URL = os.environ.get('FLUX_URL', 'http://localhost:2222')
API_PREFIX = '/api/v1'

# Terminal states a job can settle into (see Job.state on the server).
TERMINAL_STATES = ('done', 'failed', 'canceled')


class FluxError(RuntimeError):
    """An error the server reported, carrying its machine-readable code.

    Every /api/v1 failure is {"error": {"code", "message"}}; `code` is what to
    branch on ("queue_full", "model_loading", "not_found", ...) since messages
    are free to change.
    """

    def __init__(self, code, message, status=None):
        super().__init__(f"{code}: {message}" if code else message)
        self.code = code
        self.message = message
        self.status = status


# --------------------------------------------------------------- helpers ----

def encode_image(path):
    """Read an image file into the data URL the API accepts for references.

    Reference images can also be sent as bare base64, or skipped entirely by
    passing `input_paths` (paths the server itself can read) — see
    FluxClient.generate.
    """
    mime = mimetypes.guess_type(path)[0] or 'image/png'
    with open(path, 'rb') as f:
        return f"data:{mime};base64," + base64.b64encode(f.read()).decode('ascii')


def _fmt_bytes(n):
    return f"{n / 1024 / 1024:.1f}MB" if n > 1024 * 1024 else f"{n / 1024:.0f}KB"


class FluxClient:
    """Thin wrapper over the REST API. One method per server capability."""

    def __init__(self, base_url=DEFAULT_BASE_URL, api_key=None, timeout=60):
        self.base_url = base_url.rstrip('/')
        self.api_key = api_key or os.environ.get('FLUX_API_KEY')
        if not self.api_key:
            raise SystemExit(
                "No API key. Set FLUX_API_KEY, or pass api_key=... — it must match "
                "the key the server was started with.")
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers['X-API-Key'] = self.api_key

    # -- plumbing ----------------------------------------------------------

    def _url(self, path):
        return f"{self.base_url}{API_PREFIX}{path}"

    def _request(self, method, path, **kwargs):
        kwargs.setdefault('timeout', self.timeout)
        response = self.session.request(method, self._url(path), **kwargs)
        if response.status_code == 204:
            return None
        # Binary endpoints (image bytes, thumbnails) never carry a JSON body.
        if response.ok and not response.headers.get('content-type', '').startswith(
                'application/json'):
            return response.content
        try:
            body = response.json()
        except ValueError:
            raise FluxError('unparseable_response',
                            f"HTTP {response.status_code}: {response.text[:200]}",
                            response.status_code)
        if not response.ok:
            err = body.get('error') or {}
            raise FluxError(err.get('code', 'unknown'),
                            err.get('message', str(body)), response.status_code)
        return body

    def get(self, path, **kw):
        return self._request('GET', path, **kw)

    def post(self, path, **kw):
        return self._request('POST', path, **kw)

    def put(self, path, **kw):
        return self._request('PUT', path, **kw)

    def delete(self, path, **kw):
        return self._request('DELETE', path, **kw)

    # -- readiness and model ----------------------------------------------

    def health(self):
        """{ready, status, error, elapsed_s}. The one endpoint that needs no key.

        Returns the body even when the server answers 503 (still loading) —
        "not ready" is information, not a failure.
        """
        r = self.session.get(self._url('/health'), timeout=self.timeout)
        return r.json()

    def wait_until_ready(self, timeout=1800, poll=3.0, verbose=True):
        """Block until the model finishes loading. Loading a 32B FLUX.2 from
        cold can take minutes, and a config switch restarts the process."""
        deadline = time.time() + timeout
        last = None
        while time.time() < deadline:
            try:
                state = self.health()
            except requests.RequestException:
                state = {'ready': False, 'status': 'server unreachable (restarting?)'}
            if state.get('ready'):
                if verbose:
                    print(f"Model ready in {state.get('elapsed_s', 0)}s")
                return state
            if state.get('error'):
                raise FluxError('model_load_failed', state['error'])
            if verbose and state.get('status') != last:
                last = state.get('status')
                print(f"  {last}...")
            time.sleep(poll)
        raise TimeoutError(f"Model was not ready within {timeout}s")

    def model(self):
        """Capabilities of the loaded backend: which of negative_prompt,
        inpaint, kontext and multi-reference this process can actually serve."""
        return self.get('/model')

    def models(self):
        """The launcher's config menu, plus which one is live."""
        return self.get('/models')

    def switch_model(self, config, wait=True):
        """Switch to another config. This restarts the server process, so the
        202 means "restart scheduled", not "new model ready"."""
        result = self.put('/models/current', json={'config': int(config)})
        if wait:
            # Give the old process time to exit before polling, or the first
            # /health would answer from the server that is about to die.
            time.sleep(3)
            self.wait_until_ready()
        return result

    def telemetry(self):
        """GPU power draw and vision-model residency."""
        return self.get('/telemetry')

    # -- generation --------------------------------------------------------

    def submit(self, prompt, **params):
        """Queue a job and return it immediately, without waiting.

        Accepts every generation parameter: steps, batch, seed, guidance,
        strength, orientation, size, negative_prompt, input_images,
        input_paths, mask_image, aspect_mode, show_preview, save_previews,
        spectrum_grid. See GET /api/v1/openapi.json for the full schema.
        """
        return self.post('/jobs', json=dict(params, prompt=prompt))

    def job(self, job_id):
        """Current state of one job."""
        return self.get(f'/jobs/{job_id}')

    def jobs(self):
        """The whole queue: running, queued, and recently finished."""
        return self.get('/jobs')

    def queue(self):
        """The current queue, ordered.

        `waiting` carries each pending job's 1-based `position`, `accepting`
        says whether a new job would be taken or rejected with queue_full, and
        `estimated_wait_s` extrapolates from recently completed jobs (None
        until at least one has finished).
        """
        return self.get('/queue')

    def cancel(self, job_id):
        """Cancel a queued job, or interrupt the running one. Images already
        finished within a batch are kept."""
        return self.delete(f'/jobs/{job_id}')

    def wait_for_job(self, job_id, poll=1.5, timeout=3600, on_progress=None):
        """Poll until the job settles. Returns the finished job dict.

        `on_progress` is called with the job dict on every change of
        (state, current image, step) — enough to render a progress bar.
        """
        deadline = time.time() + timeout
        last_key = None
        while time.time() < deadline:
            job = self.job(job_id)
            key = (job['state'], job.get('current'), job.get('step'))
            if on_progress and key != last_key:
                on_progress(job)
                last_key = key
            if job['state'] in TERMINAL_STATES:
                return job
            time.sleep(poll)
        raise TimeoutError(f"Job {job_id} did not finish within {timeout}s")

    def generate(self, prompt, wait=True, on_progress=None, **params):
        """Submit a job and (by default) wait for it.

        Returns the finished job dict; `job["images"]` holds one entry per
        output with its filename and the seed that produced it. Raises
        FluxError if the job failed.
        """
        job = self.submit(prompt, **params)
        if not wait:
            return job
        finished = self.wait_for_job(job['id'], on_progress=on_progress)
        if finished['state'] == 'failed':
            raise FluxError('generation_failed', finished.get('error') or 'unknown error')
        return finished

    def previews(self, job_id):
        """The intermediate images from a generation.

        Returns {live, frames, count, saving}. `live` is the frame being
        denoised right now — only while the job runs with show_preview=True,
        and it is overwritten in place, so use its `ts` as a cache-buster.
        `frames` are per-step images written to disk, which requires
        save_previews=True but outlives the job.
        """
        return self.get(f'/jobs/{job_id}/previews')

    # -- images ------------------------------------------------------------

    def history(self):
        """Today's generated images, newest first."""
        return self.get('/images')['images']

    def image_bytes(self, filename):
        """Raw bytes of a generated image."""
        return self.get(f'/images/{filename}')

    def download(self, filename, dest_dir='.'):
        """Save a generated image next to this script. Returns the local path."""
        os.makedirs(dest_dir, exist_ok=True)
        path = os.path.join(dest_dir, os.path.basename(filename))
        data = self.image_bytes(filename)
        with open(path, 'wb') as f:
            f.write(data)
        return path

    def save_image(self, filename):
        """Copy an image into .saved/, out of reach of archive and delete."""
        return self.post(f'/images/{filename}/save')

    def delete_image(self, filename):
        """Permanently delete one of today's images and its sidecar."""
        return self.delete(f'/images/{filename}')

    def delete_today(self):
        """Permanently delete all of today's output. Irreversible."""
        return self.delete('/images')

    def archive(self):
        """Move today's output into web-generated/archive/."""
        return self.post('/archive')

    def filmstrip(self, filenames, direction='', prompts=None, ref_image=None):
        """Compose an edit-loop film strip and preserve its iterations."""
        body = {'filenames': filenames, 'direction': direction,
                'prompts': prompts or []}
        if ref_image:
            body['ref_image'] = ref_image
        return self.post('/filmstrips', json=body)

    # -- reference-image ingestion -----------------------------------------

    def import_raw(self, path):
        """Convert a camera RAW file (NEF/DNG/CR3/...) into a reference data
        URL. Browsers cannot decode RAW, so the server does it with LibRaw."""
        with open(path, 'rb') as f:
            return self.post('/imports/raw', files={'file': (os.path.basename(path), f)})

    def import_url(self, url):
        """Fetch a remote image server-side (no CORS limits) as a reference."""
        return self.post('/imports/url', json={'url': url})

    def import_path(self, path):
        """Load a reference from a path the server itself can read."""
        return self.post('/imports/path', json={'path': path})

    def browse(self, directory=None):
        """List subfolders and images in a server-side directory."""
        return self.get('/files', params={'dir': directory} if directory else None)

    def thumbnail(self, path):
        """JPEG bytes of a small thumbnail for any server-side image."""
        return self.get('/files/thumbnail', params={'path': path})

    # -- vision-model jobs -------------------------------------------------

    def vlm_job(self, task, **fields):
        """Start a describe/boost/critique job; returns the id to poll."""
        return self.post('/vlm/jobs', json=dict(fields, task=task))['id']

    def await_vlm(self, job_id, poll=2.0, timeout=900):
        """Poll a vision-model job until it finishes. Returns its result dict.

        A failure here is a 502 with code "vlm_failed" — the request was fine,
        the vision model was not (usually ollama not running, or the model not
        pulled).
        """
        deadline = time.time() + timeout
        while time.time() < deadline:
            state = self.get(f'/vlm/jobs/{job_id}')
            if state['done']:
                return state['result']
            time.sleep(poll)
        raise TimeoutError(f"VLM job {job_id} did not finish within {timeout}s")

    def describe(self, image_paths, think=False, **kw):
        """Have the vision model write a prompt that would recreate the photo.

        Accepts up to MAX_REFERENCE_IMAGES paths; with several, the prompt
        describes one scene combining them.
        """
        if isinstance(image_paths, str):
            image_paths = [image_paths]
        job = self.vlm_job('describe',
                           images=[encode_image(p) for p in image_paths],
                           think=think, **kw)
        return self.await_vlm(job)['prompt']

    def boost(self, prompt, level=3, has_image=False, negative_prompt=None,
              think=False, **kw):
        """Rewrite a draft prompt into the idiom of the loaded model.

        `level` 1-5 sets how far the rewrite may depart from the draft: 1
        polishes wording, 5 reimagines. Pass has_image=True when references
        will be attached — the idiom shifts to edit instructions.
        """
        fields = dict(prompt=prompt, level=level, has_image=has_image, think=think, **kw)
        if negative_prompt:
            fields['negative_prompt'] = negative_prompt
        return self.await_vlm(self.vlm_job('boost', **fields))

    def critique(self, direction, ref_image_path, output_filename, prompt=None,
                 history=None, **kw):
        """Compare an edit's output against its reference and propose a revised
        instruction — the "look at the result" step of the edit loop."""
        job = self.vlm_job('critique',
                           direction=direction,
                           prompt=prompt or direction,
                           ref_image=encode_image(ref_image_path),
                           output_filename=output_filename,
                           history=history or [], **kw)
        return self.await_vlm(job)

    # -- multi-model runs --------------------------------------------------

    def multi_run(self, prompt, configs, **params):
        """Generate one prompt on several configs in turn, sharing one seed.

        Each config change restarts the server, so this returns as soon as the
        run is recorded; follow it with multi_run_status().
        """
        return self.post('/multi-runs', json=dict(params, prompt=prompt, configs=configs))

    def multi_run_status(self):
        """The active run, or None when there isn't one."""
        try:
            return self.get('/multi-runs/current')
        except FluxError as e:
            if e.code == 'not_found':
                return None
            raise

    def cancel_multi_run(self):
        return self.delete('/multi-runs/current')

    def wait_for_multi_run(self, poll=5.0, timeout=7200, verbose=True):
        """Follow a multi-model run to completion across its restarts."""
        deadline = time.time() + timeout
        seen = 0
        while time.time() < deadline:
            try:
                state = self.multi_run_status()
            except requests.RequestException:
                # Expected: the server is mid-restart between configs.
                time.sleep(poll)
                continue
            if state is None:
                return None
            run = state['run']
            if verbose and len(run['results']) > seen:
                for result in run['results'][seen:]:
                    images = len(result.get('images') or [])
                    print(f"  {result['label']}: {result['state']} ({images} image(s))")
                seen = len(run['results'])
            if not state['active']:
                return run
            time.sleep(poll)
        raise TimeoutError("Multi-model run did not finish in time")


# ------------------------------------------------------------------ demos ----

def _progress_printer():
    """A one-line progress renderer for wait_for_job's on_progress hook."""
    def render(job):
        if job['state'] == 'running' and job.get('total_steps'):
            total_imgs = job.get('batch') or 1
            bar_width = 24
            filled = int(bar_width * job['step'] / job['total_steps'])
            bar = '#' * filled + '.' * (bar_width - filled)
            sys.stdout.write(
                f"\r  image {job.get('current', 1)}/{total_imgs} "
                f"[{bar}] step {job['step']}/{job['total_steps']}")
            sys.stdout.flush()
        elif job['state'] in TERMINAL_STATES:
            sys.stdout.write(f"\r  {job['state']}{' ' * 44}\n")
            sys.stdout.flush()
    return render


def demo_health(client, args):
    """Readiness, capabilities, and what the box is doing right now."""
    state = client.health()
    print(f"ready:  {state['ready']}  ({state['status']}, {state.get('elapsed_s', 0)}s)")
    if not state['ready']:
        print("The model is still loading; generation calls will 503 until it is up.")
        return
    info = client.model()
    print(f"model:  {info['description']}")
    caps = [name for name, on in (('negative prompts', info['negative_prompt']),
                                  ('inpainting', info['inpaint']),
                                  ('kontext editing', info['kontext']),
                                  ('turbo LoRA', info['turbo'])) if on]
    print(f"can:    {', '.join(caps) or 'text-to-image only'}")
    tele = client.telemetry()
    if tele.get('power_w'):
        print(f"gpu:    {tele['power_w']}W")
    print(f"vlm:    {tele['vlm']['model']} ({tele['vlm']['status']})")
    queue = client.jobs()
    print(f"queue:  {len(queue['queued'])} waiting, "
          f"{'1 running' if queue['running'] else 'idle'}")


def demo_generate(client, args):
    """The core path: text to image, waiting for the result."""
    client.wait_until_ready()
    print(f"Generating: {args.prompt!r}")
    job = client.generate(
        args.prompt, steps=args.steps, seed=args.seed, size=args.size,
        orientation=args.orientation, guidance=args.guidance,
        on_progress=_progress_printer())
    for image in job['images']:
        path = client.download(image['filename'], args.out)
        print(f"  saved {path}  (seed {image['seed']}, "
              f"{_fmt_bytes(os.path.getsize(path))})")
    print(f"  {job['generation_time']:.1f}s total")
    return job


def demo_batch(client, args):
    """Several images from one prompt. The server pre-encodes the prompt once
    for the whole batch, so this is much cheaper than N separate jobs."""
    client.wait_until_ready()
    print(f"Generating {args.count} variations of {args.prompt!r}")
    job = client.generate(args.prompt, batch=args.count, steps=args.steps,
                          size=args.size, on_progress=_progress_printer())
    for image in job['images']:
        print(f"  {client.download(image['filename'], args.out)}  (seed {image['seed']})")


def demo_img2img(client, args):
    """Image-to-image / instruction editing from a local reference file.

    Multiple references need a Kontext or FLUX.2 backend; FLUX.1 img2img takes
    exactly one. `strength` controls how far the result may drift from the
    reference.
    """
    client.wait_until_ready()
    info = client.model()
    print(f"Editing {args.image} with {info['model']}")
    job = client.generate(
        args.prompt,
        input_images=[encode_image(args.image)],
        strength=args.strength, steps=args.steps,
        aspect_mode='keep',          # derive output dims from the reference
        on_progress=_progress_printer())
    for image in job['images']:
        print(f"  {client.download(image['filename'], args.out)}")


def demo_inpaint(client, args):
    """Masked editing. Requires a FLUX.2 or SDXL backend and exactly one
    reference; white in the mask marks the region to regenerate."""
    client.wait_until_ready()
    if not client.model()['inpaint']:
        print("This backend cannot inpaint — start the server with a FLUX.2 config.")
        return
    job = client.generate(
        args.prompt,
        input_images=[encode_image(args.image)],
        mask_image=encode_image(args.mask),
        steps=args.steps, on_progress=_progress_printer())
    for image in job['images']:
        print(f"  {client.download(image['filename'], args.out)}")


def demo_progress(client, args):
    """Watch the intermediate images while a generation runs.

    show_preview decodes the latent being denoised; save_previews additionally
    writes every frame to disk so the run can be replayed afterwards.
    """
    client.wait_until_ready()
    job = client.submit(args.prompt, steps=args.steps,
                        show_preview=True, save_previews=args.save)
    print(f"Job {job['id']} queued at position {job['position']}")

    seen = None
    while True:
        previews = client.previews(job['id'])
        live = previews['live']
        if live and live['step'] != seen:
            seen = live['step']
            print(f"  step {live['step']}/{live['total_steps']}  {live['url']}")
        if previews['state'] in TERMINAL_STATES:
            break
        time.sleep(1.0)

    final = client.job(job['id'])
    if args.save:
        frames = client.previews(job['id'])['frames']
        print(f"\n{len(frames)} frame(s) kept on disk:")
        for frame in frames[:10]:
            print(f"  image {frame['image']} step {frame['step']:>3}  {frame['path']}")
        if len(frames) > 10:
            print(f"  ... and {len(frames) - 10} more")

    for image in final['images']:
        print(f"  {client.download(image['filename'], args.out)}")


def demo_describe(client, args):
    """Reverse path: photo in, prompt out — then optionally regenerate from it."""
    print(f"Describing {args.image}...")
    prompt = client.describe(args.image, think=args.think)
    print(f"\n{prompt}\n")
    if args.regenerate:
        client.wait_until_ready()
        job = client.generate(prompt, on_progress=_progress_printer())
        for image in job['images']:
            print(f"  {client.download(image['filename'], args.out)}")


def demo_boost(client, args):
    """Rewrite a draft prompt into the loaded model's idiom before generating."""
    print(f"Boosting (level {args.level}): {args.prompt!r}")
    result = client.boost(args.prompt, level=args.level, think=args.think)
    print(f"\n{result['prompt']}\n")
    if result.get('negative_prompt'):
        print(f"negative: {result['negative_prompt']}\n")
    if args.generate:
        client.wait_until_ready()
        job = client.generate(result['prompt'], on_progress=_progress_printer())
        for image in job['images']:
            print(f"  {client.download(image['filename'], args.out)}")


def demo_queue(client, args):
    """What is generating now and what is waiting behind it."""
    q = client.queue()
    running = q['running']
    if running:
        total = running.get('batch') or 1
        print(f"running   {running['id']}  image {running.get('current', 1)}/{total} "
              f"step {running.get('step', 0)}/{running.get('total_steps', 0)}")
        print(f"          {running['prompt'][:70]}")
    else:
        print("running   (idle)")

    print(f"\nwaiting   {q['depth']}/{q['capacity']}"
          f"{'' if q['accepting'] else '  — FULL, new jobs are rejected'}")
    for job in q['waiting']:
        print(f"  {job['position']:>2}.  {job['id']}  x{job.get('batch', 1)}  "
              f"{job['prompt'][:56]}")

    if q['estimated_wait_s'] is not None:
        print(f"\n{q['images_pending']} image(s) pending, about "
              f"{q['estimated_wait_s']:.0f}s at {q['seconds_per_image']:.1f}s each")
    elif q['images_pending']:
        print(f"\n{q['images_pending']} image(s) pending "
              "(no completed job yet to estimate from)")


def demo_history(client, args):
    """Today's output, and where the housekeeping actions live."""
    images = client.history()
    print(f"{len(images)} image(s) generated today")
    for image in images[:args.limit]:
        prompt = (image.get('prompt') or '')[:64]
        print(f"  {image['time']}  {image['filename']}  {prompt}")
    if len(images) > args.limit:
        print(f"  ... and {len(images) - args.limit} more")
    print("\nHousekeeping: client.archive(), client.save_image(name), "
          "client.delete_image(name), client.delete_today()")


def demo_models(client, args):
    """The config menu, marking the live one."""
    info = client.models()
    for config in info['configs']:
        marker = ' <- running' if config['id'] == info['current'] else ''
        print(f"  {config['id']:>2}  {config['label']}{marker}")
    if not info['switchable']:
        print("\nNot switchable: the server was started without the run_server "
              "supervisor, so it cannot restart itself into another config.")


def demo_switch(client, args):
    """Switch configs and wait through the restart."""
    print(f"Switching to config {args.config}...")
    result = client.switch_model(args.config)
    print(f"Now running: {result['switching_to']}")
    print(client.model()['description'])


def demo_multirun(client, args):
    """One prompt across several models, for a like-for-like comparison."""
    run = client.multi_run(args.prompt, args.configs, steps=args.steps)
    print(f"Run {run['id']} over configs {run['configs']} (shared seed {run['seed']})")
    print("The server restarts between models; this follows it through.\n")
    finished = client.wait_for_multi_run()
    if finished:
        print(f"\nDone — {len(finished['results'])} model(s)")


def demo_browse(client, args):
    """Server-side file browsing, for picking references without uploading."""
    listing = client.browse(args.dir)
    print(f"{listing['dir']}")
    for name in listing['dirs'][:20]:
        print(f"  [dir]  {name}")
    for entry in listing['files'][:20]:
        print(f"         {entry['filename']}")
    print(f"\n{len(listing['dirs'])} folder(s), {len(listing['files'])} image(s). "
          "Use these paths with generate(input_paths=[...]) to skip uploading.")


def demo_errors(client, args):
    """How failures surface: every error carries a stable `code` to branch on."""
    print("Errors are {'error': {'code', 'message'}} — branch on code:\n")
    for label, call in (
            ('unknown job', lambda: client.job('nosuchjob1234')),
            ('empty prompt', lambda: client.submit('')),
            ('bad step count', lambda: client.submit('x', steps=9999)),
            ('bad import url', lambda: client.import_url('ftp://example.com/x.png')),
    ):
        try:
            call()
            print(f"  {label:<16} unexpectedly succeeded")
        except FluxError as e:
            print(f"  {label:<16} HTTP {e.status}  code={e.code}")


def demo_all(client, args):
    """Every read-only call, so you can verify a deployment end to end."""
    for name, fn in (('health', demo_health), ('queue', demo_queue),
                     ('models', demo_models), ('history', demo_history),
                     ('errors', demo_errors)):
        print(f"\n--- {name} ---")
        fn(client, args)


# -------------------------------------------------------------------- cli ----

def build_parser():
    parser = argparse.ArgumentParser(
        description=__doc__.split('\n')[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Set FLUX_API_KEY (and FLUX_URL for a remote host) first.")
    parser.add_argument('--url', default=DEFAULT_BASE_URL, help='server base URL')
    parser.add_argument('--out', default='.', help='directory for downloaded images')
    sub = parser.add_subparsers(dest='command', required=True)

    def add(name, fn, help_text):
        p = sub.add_parser(name, help=help_text,
                           description=(fn.__doc__ or '').strip())
        p.set_defaults(fn=fn)
        return p

    p = add('health', demo_health, 'server readiness and capabilities')

    p = add('generate', demo_generate, 'text to image')
    p.add_argument('prompt')
    p.add_argument('--steps', type=int, default=25)
    p.add_argument('--seed', type=int)
    p.add_argument('--guidance', type=float)
    p.add_argument('--size', default='1mp')
    p.add_argument('--orientation', default='landscape')

    p = add('batch', demo_batch, 'several images from one prompt')
    p.add_argument('prompt')
    p.add_argument('--count', type=int, default=4)
    p.add_argument('--steps', type=int, default=25)
    p.add_argument('--size', default='1mp')

    p = add('img2img', demo_img2img, 'edit an existing image')
    p.add_argument('image')
    p.add_argument('prompt')
    p.add_argument('--strength', type=float, default=0.5)
    p.add_argument('--steps', type=int, default=25)

    p = add('inpaint', demo_inpaint, 'masked editing (FLUX.2 or SDXL)')
    p.add_argument('image')
    p.add_argument('mask')
    p.add_argument('prompt')
    p.add_argument('--steps', type=int, default=25)

    p = add('progress', demo_progress, 'watch the intermediate images')
    p.add_argument('prompt')
    p.add_argument('--steps', type=int, default=25)
    p.add_argument('--save', action='store_true',
                   help='also keep every frame on disk (save_previews)')

    p = add('describe', demo_describe, 'photo to prompt, via the vision model')
    p.add_argument('image')
    p.add_argument('--think', action='store_true', help='deeper but much slower')
    p.add_argument('--regenerate', action='store_true',
                   help='generate a fresh image from the description')

    p = add('boost', demo_boost, 'rewrite a draft prompt')
    p.add_argument('prompt')
    p.add_argument('--level', type=int, default=3, choices=range(1, 6))
    p.add_argument('--think', action='store_true')
    p.add_argument('--generate', action='store_true', help='then generate it')

    p = add('queue', demo_queue, 'what is running and what is waiting')

    p = add('history', demo_history, "today's images")
    p.add_argument('--limit', type=int, default=20)

    p = add('models', demo_models, 'list configs')

    p = add('switch', demo_switch, 'switch config (restarts the server)')
    p.add_argument('config', type=int)

    p = add('multirun', demo_multirun, 'one prompt across several configs')
    p.add_argument('prompt')
    p.add_argument('configs', type=int, nargs='+')
    p.add_argument('--steps', type=int, default=25)

    p = add('browse', demo_browse, 'browse server-side folders')
    p.add_argument('--dir', default=None)

    p = add('errors', demo_errors, 'how failures surface')

    p = add('demo', demo_all, 'every read-only call')
    p.add_argument('--limit', type=int, default=10)

    return parser


def main():
    args = build_parser().parse_args()
    client = FluxClient(base_url=args.url)
    try:
        args.fn(client, args)
    except FluxError as e:
        print(f"\nServer rejected the request — code={e.code} (HTTP {e.status})\n"
              f"  {e.message}", file=sys.stderr)
        return 1
    except requests.ConnectionError:
        print(f"\nCould not reach {args.url}. Is the server running?", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\ninterrupted", file=sys.stderr)
        return 130
    return 0


if __name__ == '__main__':
    sys.exit(main())
