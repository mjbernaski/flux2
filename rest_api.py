"""REST API (/api/v1) for the FLUX image generation server.

This is a second, resource-oriented dialect over the same machinery the flat
legacy routes use. Both call the `_api_*` core functions in web_server.py, so
there is exactly one implementation of each behavior — this module only owns
routing, HTTP semantics, and the response envelope.

Why it exists alongside the legacy routes: static/app.js calls the flat routes
(/generate, /status, /delete, ...) throughout, and they answer 200 with
`{"success": false}` in places, use three different id fields for one async job
type, and mix resources with actions. Rather than break the UI, /api/v1 gives
external clients a consistent surface:

  * resources and HTTP verbs        POST /jobs, DELETE /jobs/{id}
  * status codes carry the outcome  201 created, 202 accepted, 204 no content,
                                    404 not found, 409 conflict, 429 too many
  * one error envelope              {"error": {"code", "message"}}
  * one async job resource          /vlm/jobs replaces critique/describe/boost

Auth matches the rest of the server: an `X-API-Key` header or `api_key` query
param, enforced by web_server's global before_request hook. Only /health,
/openapi.json, /docs and the index are exempt (see PUBLIC_ENDPOINTS).

Wire it up from web_server.py with:

    import rest_api
    rest_api.init_app(app, sys.modules[__name__])

The module object is passed in rather than imported: web_server runs as
`__main__`, so `import web_server` here would execute the file a second time
and give this module a different queue, a different model, and a second API-key
check.
"""

import functools
import io
import traceback

from flask import Blueprint, Response, jsonify, request, send_from_directory, url_for

API_VERSION = 'v1'
URL_PREFIX = '/api/' + API_VERSION

# Endpoint names (as Flask sees them: "<blueprint>.<function>") reachable
# without an API key. Everything else on the blueprint is gated. /health is
# public for the same reason /ready is — a client has to be able to see whether
# the server is up before it can be told a key is wrong.
PUBLIC_ENDPOINTS = ['rest.index', 'rest.health', 'rest.openapi', 'rest.docs']

rest = Blueprint('rest', __name__, url_prefix=URL_PREFIX)

# Set by init_app to the live web_server module object.
ws = None


# ---------------------------------------------------------------- plumbing --

def _error(message, status=400, code='invalid_request', **extra):
    """The single error shape this API returns."""
    body = {'code': code, 'message': message}
    body.update(extra)
    return jsonify({'error': body}), status


def endpoint(fn):
    """Translate the core layer's ApiError into the REST error envelope, and
    stop an unexpected exception from leaking a Flask HTML traceback into a
    JSON client."""
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except ws.ApiError as e:
            return _error(e.message, e.status, e.code)
        except Exception as e:  # noqa: BLE001 — last line of defense
            traceback.print_exc()
            return _error(f'internal error: {e}', 500, 'internal_error')
    return wrapper


def _body():
    """The JSON request body as a dict, tolerating an absent or empty body."""
    return request.get_json(silent=True) or {}


def _job_url(job_id):
    return url_for('rest.get_job', job_id=job_id, _external=False)


# ------------------------------------------------------- service discovery --

@rest.route('', methods=['GET'])
@rest.route('/', methods=['GET'])
@endpoint
def index():
    """Entry point: what this API is and where the machine-readable spec is."""
    return jsonify({
        'service': 'flux-image-generator',
        'api_version': API_VERSION,
        'server_version': ws.VERSION,
        'openapi': URL_PREFIX + '/openapi.json',
        'docs': URL_PREFIX + '/docs',
        'authentication': {
            'header': 'X-API-Key',
            'query_param': 'api_key',
            'public_paths': [URL_PREFIX, URL_PREFIX + '/health',
                             URL_PREFIX + '/openapi.json', URL_PREFIX + '/docs'],
        },
        'resources': {
            'health': URL_PREFIX + '/health',
            'model': URL_PREFIX + '/model',
            'models': URL_PREFIX + '/models',
            'jobs': URL_PREFIX + '/jobs',
            'queue': URL_PREFIX + '/queue',
            'images': URL_PREFIX + '/images',
            'imports': URL_PREFIX + '/imports',
            'files': URL_PREFIX + '/files',
            'vlm_jobs': URL_PREFIX + '/vlm/jobs',
            'multi_runs': URL_PREFIX + '/multi-runs',
            'filmstrips': URL_PREFIX + '/filmstrips',
            'telemetry': URL_PREFIX + '/telemetry',
        },
    })


# --------------------------------------------------------- health and model --

@rest.route('/health', methods=['GET'])
@endpoint
def health():
    """Model-load readiness. Public, and the only endpoint safe to poll before
    the caller knows whether its key is good."""
    payload = ws._api_readiness()
    payload['server_version'] = ws.VERSION
    return jsonify(payload), (200 if payload['ready'] else 503)


@rest.route('/model', methods=['GET'])
@endpoint
def get_model():
    """Capabilities of the currently loaded backend."""
    return jsonify(ws._api_model_info())


@rest.route('/models', methods=['GET'])
@endpoint
def list_models():
    """The launcher's config menu. `switchable` is false when the server was
    started without the run_server supervisor, in which case PUT
    /models/current cannot work."""
    return jsonify(ws._api_configs())


@rest.route('/models/current', methods=['GET'])
@endpoint
def get_current_model():
    cfg = ws._api_configs()
    return jsonify({
        'config': cfg['current'],
        'label': ws.SERVER_CONFIGS.get(cfg['current']),
        'switchable': cfg['switchable'],
        'model': ws._api_model_info(),
    })


@rest.route('/models/current', methods=['PUT'])
@endpoint
def switch_current_model():
    """Switch configs. The models are too large to hot-swap, so this is a
    supervised process restart: 202 means the restart was scheduled, not that
    the new model is up. Poll /health until `ready` before generating."""
    label = ws._api_switch_model(_body().get('config'))
    return jsonify({
        'switching_to': label,
        'config': int(_body().get('config')),
        'poll': URL_PREFIX + '/health',
        'message': 'Server is restarting into the requested config.',
    }), 202


@rest.route('/telemetry', methods=['GET'])
@endpoint
def telemetry():
    """GPU draw and vision-model residency — the numbers behind the UI badges."""
    return jsonify({
        'power_w': ws._gpu_power_watts(),
        'vlm': ws._vlm_status(),
        'queue_max_size': ws.QUEUE_MAX_SIZE,
    })


# ------------------------------------------------------ generation and jobs --

@rest.route('/jobs', methods=['POST'])
@endpoint
def create_job():
    """Queue a generation job.

    Returns 201 with the queued job and a Location header. Generation is
    asynchronous in every case — even an empty queue runs the job on the
    single worker thread — so poll GET /jobs/{id} for progress and results.
    """
    job, position = ws._api_enqueue_generation(_body())
    payload = job.full()
    payload['position'] = position
    payload['url'] = _job_url(job.id)
    return jsonify(payload), 201, {'Location': _job_url(job.id)}


@rest.route('/jobs', methods=['GET'])
@endpoint
def list_jobs():
    """The whole queue in one call: what's running, what's waiting, what
    recently finished (bounded by RECENT_DONE_MAX)."""
    snapshot = ws._api_queue_snapshot()
    return jsonify({
        'running': snapshot['running'],
        'queued': snapshot['queued'],
        'recent': snapshot['recent_done'],
        'queue_max_size': ws.QUEUE_MAX_SIZE,
    })


@rest.route('/queue', methods=['GET'])
@endpoint
def get_queue():
    """The current queue, ordered.

    `waiting` lists pending jobs with their 1-based `position`; `running` is
    the one generating now. `accepting` says whether POST /jobs would be taken
    or rejected with queue_full — the running job holds no pending slot, so a
    full queue can still have one in flight. `estimated_wait_s` extrapolates
    from recently completed jobs and is null until at least one has finished.
    """
    return jsonify(ws._api_queue_view())


@rest.route('/queue.html', methods=['GET'])
@endpoint
def queue_page():
    """Human-readable view of GET /queue, refreshing itself every 2s.

    A browser cannot send an X-API-Key header when you just navigate to a URL,
    so open this with ?api_key=... — the page then reuses that key for its own
    polling. Image thumbnails come from the legacy /images/ route, which is
    deliberately unauthenticated so <img> tags work.
    """
    return Response(_QUEUE_PAGE.replace('__PREFIX__', URL_PREFIX), mimetype='text/html')


@rest.route('/jobs/<job_id>/previews.html', methods=['GET'])
@endpoint
def job_previews_page(job_id):
    """Human-readable view of the intermediate images from one generation.

    Shows the frame currently being denoised at full width, and every saved
    per-step frame below it grouped by batch image. Same ?api_key=... rule as
    /queue.html.
    """
    if not job_id.isalnum():
        return _error('invalid job id', 400, 'invalid_request')
    page = (_PREVIEWS_PAGE
            .replace('__PREFIX__', URL_PREFIX)
            .replace('__JOB_ID__', job_id))
    return Response(page, mimetype='text/html')


@rest.route('/jobs/recent', methods=['DELETE'])
@endpoint
def clear_recent_jobs():
    """Forget finished jobs so a fresh client doesn't replay an old session.
    Registered before the /jobs/<id> rule; job ids are 12 hex chars and can
    never collide with the literal "recent"."""
    with ws._queue_cv:
        ws._recent_done.clear()
    return '', 204


@rest.route('/jobs/<job_id>', methods=['GET'])
@endpoint
def get_job(job_id):
    """A single job by id, whether running, queued, or recently finished."""
    job = ws._api_find_job(job_id)
    if job is None:
        return _error(f'no job with id {job_id}', 404, 'not_found')
    return jsonify(job)


@rest.route('/jobs/<job_id>', methods=['DELETE'])
@endpoint
def cancel_job(job_id):
    """Cancel a queued job, or interrupt the running one. Images already
    finished within a batch are kept."""
    message = ws._api_cancel_job(job_id)
    return jsonify({'id': job_id, 'message': message})


@rest.route('/jobs/<job_id>/previews', methods=['GET'])
@endpoint
def job_previews(job_id):
    """The intermediate images produced while generating.

    `live` is the frame being denoised right now (only while the job runs with
    `show_preview`); `frames` are the per-step images written to disk (only
    with `save_previews`, but they outlive the job). Each entry carries a ready
    URL — the underlying paths are served by the image resource.
    """
    payload = ws._api_job_previews(job_id)
    for frame in payload['frames']:
        frame['url'] = f"{URL_PREFIX}/images/{frame['path']}"
    if payload['live']:
        payload['live']['url'] = (
            f"{URL_PREFIX}/images/{payload['live']['path']}?t={payload['live']['ts']}")
    return jsonify(payload)


# ------------------------------------------------------------------ images --

@rest.route('/images', methods=['GET'])
@endpoint
def list_images():
    """Today's generated images, newest first."""
    images = ws._api_history()
    return jsonify({'images': images, 'count': len(images)})


@rest.route('/images', methods=['DELETE'])
@endpoint
def delete_all_images():
    """Permanently delete all of today's output. Irreversible — archive first
    if the files still matter."""
    return jsonify({'deleted': ws._api_delete_today(None)})


@rest.route('/images/<path:filename>', methods=['GET'])
@endpoint
def get_image(filename):
    """Image bytes. Unlike the legacy /images/<name>, this requires the API key
    like every other REST endpoint; pass ?api_key=... when the consumer is an
    <img> tag that cannot set headers."""
    return send_from_directory(ws.OUTPUT_DIR, filename)


@rest.route('/images/<filename>', methods=['DELETE'])
@endpoint
def delete_image(filename):
    """Delete one of today's images and its .prompt sidecar."""
    return jsonify({'deleted': ws._api_delete_today(filename), 'filename': filename})


@rest.route('/images/<filename>/save', methods=['POST'])
@endpoint
def save_image(filename):
    """Copy an image into .saved/, where archive and delete-today can't reach
    it."""
    return jsonify({'saved': ws._api_save_hidden(filename)})


@rest.route('/archive', methods=['POST'])
@endpoint
def archive():
    """Move today's output into web-generated/archive/."""
    return jsonify({'moved': ws._api_archive_today()})


@rest.route('/filmstrips', methods=['POST'])
@endpoint
def create_filmstrip():
    """Finish an edit-loop run: preserve every iteration in .saved/ and compose
    a film strip of the reference plus each edit, saved as a normal output."""
    result = ws._api_build_loop_strip(_body())
    return jsonify(result), 201


# ------------------------------------------------- reference-image ingestion --

@rest.route('/imports/raw', methods=['POST'])
@endpoint
def import_raw():
    """Convert an uploaded camera RAW file (NEF/DNG/CR3/...) into a JPEG data
    URL usable as a reference image. multipart/form-data, field `file`."""
    f = request.files.get('file')
    if f is None:
        return _error('multipart form field "file" is required', 400, 'invalid_request')
    image = ws._api_convert_raw(f.read(), f.filename)
    return jsonify(ws._api_reference_payload(image))


@rest.route('/imports/url', methods=['POST'])
@endpoint
def import_url():
    """Fetch a remote image server-side (no browser CORS limits) as a
    reference. Body: {"url": "https://..."}."""
    image = ws._api_fetch_url_image(_body().get('url'))
    return jsonify(ws._api_reference_payload(image))


@rest.route('/imports/path', methods=['POST'])
@endpoint
def import_path():
    """Load a reference image from the server's own filesystem. Body:
    {"path": "..."} — absolute, ~-prefixed, or relative to web-generated/."""
    image = ws._api_load_path_image(_body().get('path'))
    return jsonify(ws._api_reference_payload(image))


@rest.route('/files', methods=['GET'])
@endpoint
def browse_files():
    """List subfolders and displayable images in a server-side directory, for
    a reference picker. Query: `dir` (defaults to the archive folder)."""
    return jsonify(ws._api_browse_dir(request.args.get('dir')))


@rest.route('/files/thumbnail', methods=['GET'])
@endpoint
def file_thumbnail():
    """A small JPEG thumbnail of any server-side image. Query: `path`."""
    data = ws._api_thumbnail_bytes(request.args.get('path'))
    return Response(data, mimetype='image/jpeg')


# -------------------------------------------------------------- VLM jobs ----

# The legacy API exposes three near-identical async endpoints whose only real
# difference is which runner they start and what the result carries. Here they
# are one resource discriminated by `task`, so a client writes one poll loop.
VLM_TASKS = ('describe', 'boost', 'critique')


@rest.route('/vlm/jobs', methods=['POST'])
@endpoint
def create_vlm_job():
    """Start a vision-model job and return 202 with an id to poll.

    Body: {"task": "describe" | "boost" | "critique", ...task fields}.
      describe  images[] (base64/data URLs, up to MAX_REFERENCE_IMAGES),
                think, model              -> result.prompt
      boost     prompt, level 1-5, think, has_image, negative_prompt,
                variant_index, variant_count, model
                                          -> result.prompt, result.negative_prompt
      critique  direction, prompt, ref_image, output_filename, history, model
                                          -> result.critique, result.revised_prompt,
                                             result.score, result.metrics

    These calls can run for minutes on a local vision model, which is why they
    are jobs and not synchronous requests.
    """
    body = _body()
    task = (body.get('task') or '').strip().lower()
    if task not in VLM_TASKS:
        return _error(f'task must be one of {list(VLM_TASKS)}', 400, 'invalid_request')
    starter = {
        'describe': ws._api_start_describe,
        'boost': ws._api_start_boost,
        'critique': ws._api_start_critique,
    }[task]
    job_id = starter(body)
    location = url_for('rest.get_vlm_job', job_id=job_id, _external=False)
    return jsonify({
        'id': job_id,
        'task': task,
        'done': False,
        'url': location,
    }), 202, {'Location': location}


@rest.route('/vlm/jobs/<job_id>', methods=['GET'])
@endpoint
def get_vlm_job(job_id):
    """Poll a vision-model job.

    While running: 200 {"id", "done": false}. On success: 200 with `result`.
    If the job itself failed, the status is 502 and `error` explains why — the
    request was fine, the vision model was not.
    """
    result, done = ws._api_vlm_result(job_id)
    if not done:
        return jsonify({'id': job_id, 'done': False, 'result': None})
    if not result.get('success'):
        return _error(result.get('error') or 'vlm job failed', 502, 'vlm_failed',
                      id=job_id, done=True)
    result.pop('success', None)
    return jsonify({'id': job_id, 'done': True, 'result': result})


# ------------------------------------------------------------ multi-model ----

@rest.route('/multi-runs', methods=['POST'])
@endpoint
def create_multi_run():
    """Run one prompt across several configs, one after another.

    Each config change is a supervised process restart, so the run is
    file-backed and survives them; there is at most one active run at a time.
    Text-to-image only. Unless a seed is supplied one is drawn and shared by
    every model so the outputs are comparable.
    """
    state = ws._api_start_multi_run(_body())
    location = URL_PREFIX + '/multi-runs/current'
    return jsonify({
        'id': state['id'],
        'configs': state['configs'],
        'seed': state['params']['seed'],
        'url': location,
    }), 201, {'Location': location}


@rest.route('/multi-runs/current', methods=['GET'])
@endpoint
def get_multi_run():
    """The active run, or the last finished one until it is dismissed."""
    state = ws._api_multi_run_state()
    if state['run'] is None:
        return _error('no multi-model run', 404, 'not_found')
    return jsonify(state)


@rest.route('/multi-runs/current', methods=['DELETE'])
@endpoint
def cancel_multi_run():
    """Cancel the active run (interrupting its in-flight job), or dismiss a
    finished one's results. Idempotent: 204 either way."""
    ws._api_cancel_multi_run()
    return '', 204


# --------------------------------------------------------------- openapi ----

@rest.route('/openapi.json', methods=['GET'])
@endpoint
def openapi():
    """Machine-readable description of this API."""
    return jsonify(_openapi_document())


@rest.route('/docs', methods=['GET'])
@endpoint
def docs():
    """A human-readable endpoint reference, rendered from the OpenAPI document
    so the two can't drift."""
    spec = _openapi_document()
    rows = []
    for path, ops in spec['paths'].items():
        for method, op in ops.items():
            rows.append(
                f"<tr><td class=m><span class='v {method}'>{method.upper()}</span></td>"
                f"<td class=p><code>{path}</code></td>"
                f"<td>{op.get('summary', '')}</td></tr>")
    return Response(_DOCS_HTML.format(rows='\n'.join(rows), version=ws.VERSION),
                    mimetype='text/html')


def _op(summary, *, body=None, params=None, responses=None, tag='general'):
    """One OpenAPI operation, kept terse — this spec is a client-generation and
    discovery aid, not an exhaustive schema definition."""
    op = {'summary': summary, 'tags': [tag],
          'responses': responses or {'200': {'description': 'OK'}}}
    if params:
        op['parameters'] = [
            {'name': n, 'in': loc, 'required': req,
             'schema': {'type': typ}, 'description': desc}
            for n, loc, req, typ, desc in params]
    if body is not None:
        op['requestBody'] = {
            'required': True,
            'content': {'application/json': {'schema': body}},
        }
    return op


def _openapi_document():
    """The OpenAPI 3.1 document for /api/v1."""
    err = {'description': 'Error',
           'content': {'application/json': {'schema': {'$ref': '#/components/schemas/Error'}}}}

    generate_body = {
        'type': 'object',
        'required': ['prompt'],
        'properties': {
            'prompt': {'type': 'string', 'description': 'Required. The text prompt.'},
            'steps': {'type': 'integer', 'default': 25, 'minimum': 1, 'maximum': 200},
            'batch': {'type': 'integer', 'default': 1, 'minimum': 1, 'maximum': 128},
            'seed': {'type': ['integer', 'null'],
                     'description': 'Omit for a fresh random seed per image.'},
            'guidance': {'type': ['number', 'null'], 'minimum': 0},
            'strength': {'type': 'number', 'default': 0.5, 'minimum': 0, 'maximum': 1,
                         'description': 'img2img denoising strength; ignored without a reference.'},
            'orientation': {'type': 'string', 'default': 'landscape',
                            'enum': sorted(ws.ORIENTATIONS_1K)},
            'size': {'type': 'string', 'default': '1mp', 'enum': sorted(ws.SIZES)},
            'negative_prompt': {'type': ['string', 'null'],
                                'description': 'SDXL backend only.'},
            'input_images': {'type': 'array', 'items': {'type': 'string'},
                             'maxItems': ws.MAX_REFERENCE_IMAGES,
                             'description': 'Reference images as base64 or data URLs. '
                                            'More than one requires Kontext or FLUX.2.'},
            'input_paths': {'type': 'array', 'items': {'type': 'string'},
                            'description': 'Server-side paths, loaded and folded into '
                                           'input_images.'},
            'mask_image': {'type': ['string', 'null'],
                           'description': 'Inpainting mask; requires exactly one input '
                                          'image and a FLUX.2 or SDXL backend.'},
            'aspect_mode': {'type': 'string', 'default': 'keep',
                            'description': '"keep" derives output dims from the reference.'},
            'show_preview': {'type': 'boolean', 'default': False,
                             'description': 'Decode latent previews during generation.'},
            'save_previews': {'type': 'boolean', 'default': False,
                              'description': 'Also write each preview frame to steps/; '
                                             'requires show_preview.'},
            'spectrum_grid': {'type': 'boolean', 'default': False,
                              'description': 'Sweep guidance/strength into a matrix.'},
            'spectrum_same_seed': {'type': 'boolean', 'default': True},
            'selected_cells': {'type': 'array', 'items': {'type': 'integer'}},
        },
    }

    return {
        'openapi': '3.1.0',
        'info': {
            'title': 'FLUX Image Generator API',
            'version': ws.VERSION,
            'description': 'Self-hosted FLUX.1/FLUX.2 image generation. All endpoints '
                           'except /health and the discovery documents require an API '
                           'key via the X-API-Key header or an api_key query parameter.',
        },
        'servers': [{'url': URL_PREFIX}],
        'components': {
            'securitySchemes': {
                'ApiKeyHeader': {'type': 'apiKey', 'in': 'header', 'name': 'X-API-Key'},
                'ApiKeyQuery': {'type': 'apiKey', 'in': 'query', 'name': 'api_key'},
            },
            'schemas': {
                'Error': {
                    'type': 'object',
                    'properties': {'error': {
                        'type': 'object',
                        'properties': {
                            'code': {'type': 'string',
                                     'description': 'Stable machine-readable identifier.'},
                            'message': {'type': 'string'},
                        }}},
                },
                'Job': {
                    'type': 'object',
                    'properties': {
                        'id': {'type': 'string'},
                        'state': {'type': 'string',
                                  'enum': ['queued', 'running', 'done', 'failed', 'canceled']},
                        'prompt': {'type': 'string'},
                        'current': {'type': 'integer', 'description': 'Image index in the batch.'},
                        'batch': {'type': 'integer'},
                        'step': {'type': 'integer'},
                        'total_steps': {'type': 'integer'},
                        'images': {'type': 'array', 'items': {
                            'type': 'object',
                            'properties': {
                                'filename': {'type': 'string'},
                                'seed': {'type': 'integer'},
                                'timings': {'type': 'object'},
                            }}},
                        'error': {'type': ['string', 'null']},
                        'generation_time': {'type': 'number'},
                    },
                },
            },
        },
        'security': [{'ApiKeyHeader': []}, {'ApiKeyQuery': []}],
        'paths': {
            '/': {'get': _op('Service index and resource map.', tag='discovery')},
            '/health': {'get': _op(
                'Model-load readiness. Public. 503 until ready.', tag='discovery',
                responses={'200': {'description': 'Ready'},
                           '503': {'description': 'Still loading'}})},
            '/openapi.json': {'get': _op('This document.', tag='discovery')},
            '/docs': {'get': _op('Human-readable endpoint reference.', tag='discovery')},

            '/model': {'get': _op('Capabilities of the loaded backend.', tag='model')},
            '/models': {'get': _op('Available launcher configs.', tag='model')},
            '/models/current': {
                'get': _op('The config this process is running.', tag='model'),
                'put': _op(
                    'Switch config via a supervised restart. 202 = restart scheduled.',
                    tag='model',
                    body={'type': 'object', 'required': ['config'],
                          'properties': {'config': {
                              'type': 'integer',
                              'description': 'Config id from GET /models.'}}},
                    responses={'202': {'description': 'Restart scheduled'},
                               '400': err, '409': err}),
            },
            '/telemetry': {'get': _op('GPU power draw and VLM residency.', tag='model')},

            '/jobs': {
                'post': _op(
                    'Queue a generation job. Always asynchronous — poll GET /jobs/{id}.',
                    tag='generation', body=generate_body,
                    responses={'201': {'description': 'Queued',
                                       'content': {'application/json': {
                                           'schema': {'$ref': '#/components/schemas/Job'}}}},
                               '400': err, '429': err, '503': err}),
                'get': _op('Running, queued, and recently finished jobs.', tag='generation'),
                'delete': _op('Unsupported on the collection.', tag='generation'),
            },
            '/queue': {'get': _op(
                'The current queue, ordered, with positions and a wait estimate.',
                tag='generation')},
            '/queue.html': {'get': _op(
                'Self-refreshing HTML view of the queue (open with ?api_key=...).',
                tag='generation')},
            '/jobs/{job_id}/previews.html': {'get': _op(
                'Self-refreshing HTML view of a job\'s intermediate images.',
                tag='generation',
                params=[('job_id', 'path', True, 'string', 'Job id')])},
            '/jobs/recent': {'delete': _op(
                'Clear the recently-finished list.', tag='generation',
                responses={'204': {'description': 'Cleared'}})},
            '/jobs/{job_id}': {
                'get': _op('One job by id.', tag='generation',
                           params=[('job_id', 'path', True, 'string', 'Job id')],
                           responses={'200': {'description': 'OK', 'content': {
                               'application/json': {
                                   'schema': {'$ref': '#/components/schemas/Job'}}}},
                                      '404': err}),
                'delete': _op('Cancel a queued job or interrupt the running one.',
                              tag='generation',
                              params=[('job_id', 'path', True, 'string', 'Job id')],
                              responses={'200': {'description': 'Canceled'}, '404': err}),
            },
            '/jobs/{job_id}/previews': {'get': _op(
                'Intermediate images: the live latent preview plus any saved '
                'per-step frames.', tag='generation',
                params=[('job_id', 'path', True, 'string', 'Job id')],
                responses={'200': {'description': 'OK'}, '400': err})},

            '/images': {
                'get': _op("Today's generated images, newest first.", tag='images'),
                'delete': _op("Permanently delete all of today's output.", tag='images'),
            },
            '/images/{filename}': {
                'get': _op('Image bytes. Requires auth, unlike legacy /images/.',
                           tag='images',
                           params=[('filename', 'path', True, 'string', 'Output filename')]),
                'delete': _op("Delete one of today's images and its sidecar.", tag='images',
                              params=[('filename', 'path', True, 'string', 'Output filename')],
                              responses={'200': {'description': 'Deleted'}, '400': err}),
            },
            '/images/{filename}/save': {'post': _op(
                'Copy an image into .saved/, beyond archive and delete-today.',
                tag='images',
                params=[('filename', 'path', True, 'string', 'Output filename')])},
            '/archive': {'post': _op("Move today's output into archive/.", tag='images')},
            '/filmstrips': {'post': _op(
                'Compose an edit-loop film strip and preserve its iterations.',
                tag='images',
                body={'type': 'object', 'required': ['filenames'],
                      'properties': {
                          'filenames': {'type': 'array', 'items': {'type': 'string'}},
                          'direction': {'type': 'string'},
                          'prompts': {'type': 'array', 'items': {'type': 'string'}},
                          'ref_image': {'type': 'string'}}},
                responses={'201': {'description': 'Created'}, '404': err})},

            '/imports/raw': {'post': _op(
                'Convert an uploaded camera RAW file to a reference data URL '
                '(multipart/form-data, field "file").', tag='imports',
                responses={'200': {'description': 'OK'}, '400': err, '501': err})},
            '/imports/url': {'post': _op(
                'Fetch a remote image server-side as a reference.', tag='imports',
                body={'type': 'object', 'required': ['url'],
                      'properties': {'url': {'type': 'string', 'format': 'uri'}}},
                responses={'200': {'description': 'OK'}, '400': err})},
            '/imports/path': {'post': _op(
                "Load a reference from the server's filesystem.", tag='imports',
                body={'type': 'object', 'required': ['path'],
                      'properties': {'path': {'type': 'string'}}},
                responses={'200': {'description': 'OK'}, '400': err})},
            '/files': {'get': _op(
                'Browse a server-side directory for reference images.', tag='imports',
                params=[('dir', 'query', False, 'string',
                         'Directory; relative paths resolve against web-generated/.')])},
            '/files/thumbnail': {'get': _op(
                'JPEG thumbnail of a server-side image.', tag='imports',
                params=[('path', 'query', True, 'string', 'Image path')],
                responses={'200': {'description': 'JPEG bytes'}, '404': err})},

            '/vlm/jobs': {'post': _op(
                'Start a vision-model job: describe, boost, or critique.', tag='vlm',
                body={'type': 'object', 'required': ['task'],
                      'properties': {
                          'task': {'type': 'string', 'enum': list(VLM_TASKS)},
                          'images': {'type': 'array', 'items': {'type': 'string'},
                                     'description': 'describe: base64 reference images.'},
                          'prompt': {'type': 'string',
                                     'description': 'boost/critique: the draft prompt.'},
                          'level': {'type': 'integer', 'minimum': 1, 'maximum': 5,
                                    'default': 3, 'description': 'boost: rewrite boldness.'},
                          'has_image': {'type': 'boolean',
                                        'description': 'boost: references are attached.'},
                          'negative_prompt': {'type': 'string'},
                          'variant_index': {'type': 'integer'},
                          'variant_count': {'type': 'integer'},
                          'direction': {'type': 'string',
                                        'description': 'critique: the edit instruction.'},
                          'ref_image': {'type': 'string',
                                        'description': 'critique: the "before" image.'},
                          'output_filename': {'type': 'string',
                                              'description': 'critique: the "after" output.'},
                          'history': {'type': 'array', 'items': {'type': 'object'}},
                          'think': {'type': 'boolean', 'default': False},
                          'model': {'type': 'string'}}},
                responses={'202': {'description': 'Accepted'}, '400': err, '404': err})},
            '/vlm/jobs/{job_id}': {'get': _op(
                'Poll a vision-model job. 502 if the model failed.', tag='vlm',
                params=[('job_id', 'path', True, 'string', 'VLM job id')],
                responses={'200': {'description': 'Pending or complete'},
                           '404': err, '502': err})},

            '/multi-runs': {'post': _op(
                'Run one prompt across several configs in turn.', tag='multi-run',
                body={'type': 'object', 'required': ['configs', 'prompt'],
                      'properties': {
                          'configs': {'type': 'array', 'items': {'type': 'integer'}},
                          'prompt': {'type': 'string'},
                          'orientation': {'type': 'string'},
                          'size': {'type': 'string'},
                          'steps': {'type': 'integer'},
                          'seed': {'type': 'integer'},
                          'guidance': {'type': 'number'},
                          'batch': {'type': 'integer'},
                          'show_preview': {'type': 'boolean'}}},
                responses={'201': {'description': 'Started'}, '400': err, '409': err,
                           '503': err})},
            '/multi-runs/current': {
                'get': _op('The active or last-finished run.', tag='multi-run',
                           responses={'200': {'description': 'OK'}, '404': err}),
                'delete': _op('Cancel or dismiss the run.', tag='multi-run',
                              responses={'204': {'description': 'Cleared'}}),
            },
        },
    }


# Shared chrome for the two live views. Kept as plain strings rather than
# Jinja templates: the server has no template directory, and these pages fetch
# their own data, so there is nothing to interpolate server-side beyond the
# URL prefix and the job id.
_VIEW_CSS = """
  :root { color-scheme: light dark;
          --fg:#111; --dim:#666; --line:#e3e3e6; --bg:#fff; --card:#fafafb;
          --accent:#d9480f; --ok:#16a34a; --warn:#ca8a04; }
  @media (prefers-color-scheme: dark) {
    :root { --fg:#e8e8e8; --dim:#9a9a9a; --line:#2c2c2c; --bg:#141414;
            --card:#1c1c1c; --accent:#ff7a45; }
  }
  * { box-sizing: border-box; }
  body { margin:0 auto; padding:24px 20px 60px; max-width:1100px;
         background:var(--bg); color:var(--fg);
         font:15px/1.55 ui-sans-serif,system-ui,-apple-system,sans-serif; }
  h1 { font-size:20px; margin:0 0 2px; }
  .sub { color:var(--dim); font-size:13px; margin:0 0 22px; }
  .sub a { color:var(--accent); }
  .card { background:var(--card); border:1px solid var(--line);
          border-radius:8px; padding:14px 16px; margin-bottom:16px; }
  .card h2 { font-size:13px; text-transform:uppercase; letter-spacing:.06em;
             color:var(--dim); margin:0 0 10px; font-weight:600; }
  .idle { color:var(--dim); font-style:italic; }
  code, .mono { font:12.5px ui-monospace,SFMono-Regular,Menlo,monospace; }
  .bar { height:7px; background:var(--line); border-radius:4px; overflow:hidden;
         margin:9px 0 4px; }
  .bar > i { display:block; height:100%; background:var(--accent);
             transition:width .3s ease; }
  table { border-collapse:collapse; width:100%; font-size:14px; }
  th { text-align:left; color:var(--dim); font-size:12px; font-weight:600;
       text-transform:uppercase; letter-spacing:.05em;
       border-bottom:1px solid var(--line); padding:6px 8px; }
  td { border-bottom:1px solid var(--line); padding:8px; vertical-align:top; }
  td.pos { width:38px; color:var(--accent); font-weight:700; }
  .pill { display:inline-block; padding:2px 9px; border-radius:99px;
          font-size:12px; font-weight:600; background:var(--line); }
  .pill.ok { background:rgba(22,163,74,.16); color:var(--ok); }
  .pill.warn { background:rgba(202,138,4,.18); color:var(--warn); }
  .err { color:#dc2626; }
"""

_QUEUE_PAGE = """<!doctype html>
<meta charset=utf-8><meta name=viewport content="width=device-width,initial-scale=1">
<title>FLUX queue</title>
<style>""" + _VIEW_CSS + """
  .stats { display:flex; gap:26px; flex-wrap:wrap; }
  .stat b { display:block; font-size:22px; font-weight:700; }
  .stat span { color:var(--dim); font-size:12px; text-transform:uppercase;
               letter-spacing:.05em; }
  .running { display:flex; gap:16px; align-items:flex-start; }
  .running img { width:180px; border-radius:6px; border:1px solid var(--line);
                 background:var(--line); }
  .running .meta { flex:1; min-width:0; }
  .prompt { margin:2px 0 0; overflow-wrap:anywhere; }
</style>
<h1>Generation queue</h1>
<p class=sub>Live, refreshing every 2s &middot;
  <a href="__PREFIX__/queue">JSON</a> &middot; <a href="__PREFIX__/docs">API docs</a></p>
<div id=app><p class=idle>Loading...</p></div>
<script>
// The key travels in the page URL because a plain navigation cannot set
// headers; reuse it for polling. Thumbnails use the unauthenticated /images/
// route, so they need no key at all.
const KEY = new URLSearchParams(location.search).get('api_key') || '';
const esc = s => String(s ?? '').replace(/[&<>"]/g, c =>
    ({ '&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;' }[c]));

function renderRunning(job) {
    if (!job) return '<p class=idle>Nothing generating right now.</p>';
    const total = job.total_steps || 0;
    const pct = total ? Math.round(100 * job.step / total) : 0;
    const preview = job.preview
        ? `<img src="/images/${encodeURI(job.preview)}?t=${job.preview_ts}" alt="">`
        : '';
    return `<div class=running>${preview}<div class=meta>
        <div class=mono>${esc(job.id)} &middot; image ${job.current || 1}/${job.batch || 1}</div>
        <p class=prompt>${esc(job.prompt)}</p>
        <div class=bar><i style="width:${pct}%"></i></div>
        <div class=mono>step ${job.step}/${total || '?'} (${pct}%)</div>
      </div></div>`;
}

function renderWaiting(list) {
    if (!list.length) return '<p class=idle>Nothing waiting.</p>';
    return `<table><thead><tr><th>#</th><th>Job</th><th>Prompt</th>
        <th>Images</th><th>Size</th></tr></thead><tbody>` +
      list.map(j => `<tr>
        <td class=pos>${j.position}</td>
        <td class=mono>${esc(j.id)}</td>
        <td>${esc(j.prompt)}</td>
        <td>${j.batch || 1}</td>
        <td class=mono>${esc(j.size || '')} ${esc(j.orientation || '')}</td>
      </tr>`).join('') + '</tbody></table>';
}

async function tick() {
    try {
        const res = await fetch(`__PREFIX__/queue?api_key=${encodeURIComponent(KEY)}`,
                                { cache: 'no-store' });
        if (!res.ok) {
            const body = await res.json().catch(() => ({}));
            document.getElementById('app').innerHTML =
                `<div class=card><p class=err>${esc(body.error?.message || res.status)}</p>
                 <p class=sub>Open this page with ?api_key=YOUR_KEY</p></div>`;
            return;
        }
        const q = await res.json();
        const wait = q.estimated_wait_s != null
            ? `${Math.round(q.estimated_wait_s)}s` : '—';
        document.getElementById('app').innerHTML = `
          <div class=card><h2>Now generating</h2>${renderRunning(q.running)}</div>
          <div class=card><h2>Waiting</h2>${renderWaiting(q.waiting)}</div>
          <div class="card stats">
            <div class=stat><b>${q.depth}/${q.capacity}</b><span>queued</span></div>
            <div class=stat><b>${q.images_pending}</b><span>images pending</span></div>
            <div class=stat><b>${wait}</b><span>est. wait</span></div>
            <div class=stat><b class="pill ${q.accepting ? 'ok' : 'warn'}">
                ${q.accepting ? 'accepting' : 'full'}</b><span>new jobs</span></div>
          </div>`;
    } catch (e) {
        // A dropped connection usually means a model switch is restarting the
        // server; keep polling rather than giving up.
        document.getElementById('app').innerHTML =
            '<div class=card><p class=idle>Server unreachable (restarting?)...</p></div>';
    }
}
tick();
setInterval(tick, 2000);
</script>
"""

_PREVIEWS_PAGE = """<!doctype html>
<meta charset=utf-8><meta name=viewport content="width=device-width,initial-scale=1">
<title>FLUX intermediates</title>
<style>""" + _VIEW_CSS + """
  .live { text-align:center; }
  .live img { max-width:100%; max-height:62vh; border-radius:6px;
              border:1px solid var(--line); }
  .grid { display:grid; grid-template-columns:repeat(auto-fill,minmax(116px,1fr));
          gap:9px; }
  .frame { text-align:center; }
  .frame img { width:100%; border-radius:4px; border:1px solid var(--line);
               display:block; cursor:pointer; }
  .frame span { font:11px ui-monospace,monospace; color:var(--dim); }
  .group-label { color:var(--dim); font-size:12px; margin:14px 0 7px;
                 text-transform:uppercase; letter-spacing:.05em; font-weight:600; }
</style>
<h1>Intermediate images</h1>
<p class=sub>Job <code>__JOB_ID__</code> &middot; refreshing every 1.5s while it runs &middot;
  <a href="__PREFIX__/jobs/__JOB_ID__/previews">JSON</a> &middot;
  <a href="__PREFIX__/queue.html">queue</a></p>
<div id=app><p class=idle>Loading...</p></div>
<script>
const KEY = new URLSearchParams(location.search).get('api_key') || '';
const JOB = '__JOB_ID__';
const esc = s => String(s ?? '').replace(/[&<>"]/g, c =>
    ({ '&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;' }[c]));
let timer = null;

function renderLive(live) {
    if (!live) {
        return '<p class=idle>No live preview. It appears only while the job runs ' +
               'with show_preview enabled.</p>';
    }
    const pct = live.total_steps ? Math.round(100 * live.step / live.total_steps) : 0;
    return `<div class=live>
        <img src="/images/${encodeURI(live.path)}?t=${live.ts}" alt="">
        <div class=bar><i style="width:${pct}%"></i></div>
        <div class=mono>image ${live.image || 1} &middot; step ${live.step}/${live.total_steps}</div>
      </div>`;
}

function renderFrames(frames) {
    if (!frames.length) {
        return '<p class=idle>No saved frames. Generate with save_previews to keep ' +
               'every step on disk.</p>';
    }
    // Group by batch image so a multi-image job reads as several strips.
    const groups = new Map();
    for (const f of frames) {
        const key = f.image ?? 1;
        if (!groups.has(key)) groups.set(key, []);
        groups.get(key).push(f);
    }
    return [...groups.entries()].map(([image, list]) => `
        <div class=group-label>Image ${image} — ${list.length} frame(s)</div>
        <div class=grid>` + list.map(f => `
          <div class=frame>
            <img loading=lazy src="/images/${encodeURI(f.path)}"
                 onclick="window.open(this.src,'_blank')" alt="">
            <span>step ${f.step ?? '?'}</span>
          </div>`).join('') + '</div>').join('');
}

async function tick() {
    try {
        const res = await fetch(
            `__PREFIX__/jobs/${JOB}/previews?api_key=${encodeURIComponent(KEY)}`,
            { cache: 'no-store' });
        if (!res.ok) {
            const body = await res.json().catch(() => ({}));
            document.getElementById('app').innerHTML =
                `<div class=card><p class=err>${esc(body.error?.message || res.status)}</p>
                 <p class=sub>Open this page with ?api_key=YOUR_KEY</p></div>`;
            return;
        }
        const data = await res.json();
        document.getElementById('app').innerHTML = `
          <div class=card><h2>Generating now</h2>${renderLive(data.live)}</div>
          <div class=card><h2>Saved frames (${data.count})</h2>
            ${renderFrames(data.frames)}</div>`;

        // Once the job has settled nothing more will change, so stop polling
        // rather than hammering the server behind a forgotten open tab.
        if (data.state && !['queued', 'running'].includes(data.state)) {
            clearInterval(timer);
            document.querySelector('.sub').insertAdjacentHTML('beforeend',
                ` &middot; <b>${esc(data.state)}</b>, refresh stopped`);
        }
    } catch (e) {
        document.getElementById('app').innerHTML =
            '<div class=card><p class=idle>Server unreachable (restarting?)...</p></div>';
    }
}
tick();
timer = setInterval(tick, 1500);
</script>
"""

_DOCS_HTML = """<!doctype html>
<meta charset=utf-8>
<meta name=viewport content="width=device-width,initial-scale=1">
<title>FLUX API v1</title>
<style>
  :root {{ color-scheme: light dark; --fg:#111; --dim:#666; --line:#e2e2e2; --bg:#fff;
           --code:#f4f4f5; }}
  @media (prefers-color-scheme: dark) {{
    :root {{ --fg:#e8e8e8; --dim:#9a9a9a; --line:#2c2c2c; --bg:#141414; --code:#1f1f1f; }}
  }}
  body {{ margin:0 auto; padding:32px 20px 64px; max-width:900px; background:var(--bg);
          color:var(--fg); font:15px/1.6 ui-sans-serif,system-ui,-apple-system,sans-serif; }}
  h1 {{ font-size:22px; margin:0 0 4px; }}
  p.sub {{ color:var(--dim); margin:0 0 28px; }}
  table {{ border-collapse:collapse; width:100%; }}
  td {{ border-top:1px solid var(--line); padding:9px 8px; vertical-align:top; }}
  td.m {{ width:74px; }} td.p {{ width:40%; }}
  code {{ background:var(--code); padding:2px 6px; border-radius:4px;
          font:13px ui-monospace,SFMono-Regular,Menlo,monospace; }}
  .v {{ font:11px ui-monospace,monospace; font-weight:700; padding:2px 6px;
        border-radius:4px; background:var(--code); }}
  .get {{ color:#2563eb; }} .post {{ color:#16a34a; }}
  .put {{ color:#ca8a04; }} .delete {{ color:#dc2626; }}
  .note {{ color:var(--dim); font-size:13px; margin-top:28px; }}
</style>
<h1>FLUX Image Generator — REST API v1</h1>
<p class=sub>Server {version}. Authenticate with an <code>X-API-Key</code> header
or an <code>api_key</code> query parameter. Machine-readable spec:
<code><a href="openapi.json">openapi.json</a></code>.</p>
<table>{rows}</table>
<p class=note>Generation is always asynchronous: <code>POST /jobs</code> returns 201 with a
job id, then poll <code>GET /jobs/{{id}}</code> until <code>state</code> is
<code>done</code>, <code>failed</code>, or <code>canceled</code>. Errors are
always <code>{{"error": {{"code", "message"}}}}</code>.</p>
"""


def init_app(app, server_module):
    """Register the /api/v1 blueprint on the Flask app.

    `server_module` is the live web_server module (pass sys.modules[__name__]
    from inside it) — importing it by name here would load a second copy with
    its own queue and model state.
    """
    global ws
    ws = server_module
    app.register_blueprint(rest)

    @app.errorhandler(404)
    def _rest_404(e):
        if request.path.startswith(URL_PREFIX):
            return _error(f'no such endpoint: {request.path}', 404, 'not_found')
        return e

    @app.errorhandler(405)
    def _rest_405(e):
        if request.path.startswith(URL_PREFIX):
            return _error(f'{request.method} is not allowed on {request.path}',
                          405, 'method_not_allowed')
        return e

    @app.errorhandler(413)
    def _rest_413(e):
        if request.path.startswith(URL_PREFIX):
            return _error('request body exceeds the 64MB limit', 413, 'too_large')
        return e

    return rest
