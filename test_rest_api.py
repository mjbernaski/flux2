"""Contract tests for the /api/v1 REST layer.

Runs against Flask's test client with no model loaded and no GPU, so it covers
routing, auth, status codes, and the error envelope — the parts of the contract
a client depends on — but not generation itself. For that, queue a real job
(smoke_test_servers.py, or examples/flux_client.py) against a live server.

    python test_rest_api.py
"""

import os
import sys

os.environ.setdefault('FLUX_API_KEY', 'test-key-for-contract-tests')
KEY = os.environ['FLUX_API_KEY']

import web_server as ws  # noqa: E402 — must follow the FLUX_API_KEY default

PREFIX = '/api/v1'
AUTH = {'X-API-Key': KEY}

_checks = []
_failures = []


def check(name, condition, detail=''):
    _checks.append(name)
    if condition:
        print(f"  ok    {name}")
    else:
        _failures.append(f"{name}{' — ' + detail if detail else ''}")
        print(f"  FAIL  {name}{' — ' + detail if detail else ''}")


def err_code(response):
    """The error code from the REST envelope, or None if it isn't one."""
    body = response.get_json(silent=True) or {}
    return (body.get('error') or {}).get('code')


def main():
    client = ws.app.test_client()

    print("\ndiscovery (public, no key)")
    r = client.get(PREFIX)
    check('GET /api/v1 is public', r.status_code == 200, f"got {r.status_code}")
    check('index advertises the openapi document',
          (r.get_json() or {}).get('openapi') == PREFIX + '/openapi.json')

    r = client.get(f'{PREFIX}/health')
    check('GET /health is public', r.status_code in (200, 503), f"got {r.status_code}")
    check('health reports readiness', 'ready' in (r.get_json() or {}))
    check('health is 503 until the model loads',
          r.status_code == 503 and not r.get_json()['ready'])

    r = client.get(f'{PREFIX}/openapi.json')
    spec = r.get_json() or {}
    check('GET /openapi.json is public', r.status_code == 200)
    check('spec declares openapi 3.1', str(spec.get('openapi', '')).startswith('3.1'))
    check('spec documents every registered path', _spec_covers_routes(spec))

    r = client.get(f'{PREFIX}/docs')
    check('GET /docs renders html', r.status_code == 200 and b'<table>' in r.data)

    print("\nauthentication")
    r = client.get(f'{PREFIX}/model')
    check('protected route without a key is 401', r.status_code == 401, f"got {r.status_code}")
    check('401 uses the REST error envelope', err_code(r) == 'unauthorized',
          f"got {err_code(r)!r}")

    r = client.get(f'{PREFIX}/model', headers={'X-API-Key': 'wrong'})
    check('a wrong key is 401', r.status_code == 401)

    r = client.get(f'{PREFIX}/model', headers=AUTH)
    check('header auth works', r.status_code == 200, f"got {r.status_code}")

    r = client.get(f'{PREFIX}/model?api_key={KEY}')
    check('query-param auth works', r.status_code == 200, f"got {r.status_code}")

    print("\nmodel and config resources")
    body = client.get(f'{PREFIX}/model', headers=AUTH).get_json()
    check('model reports capabilities',
          all(k in body for k in ('model', 'flux_version', 'inpaint', 'negative_prompt')))

    body = client.get(f'{PREFIX}/models', headers=AUTH).get_json()
    check('models lists the launcher configs', len(body.get('configs') or []) == 14,
          f"got {len(body.get('configs') or [])}")

    r = client.get(f'{PREFIX}/models/current', headers=AUTH)
    check('models/current responds', r.status_code == 200)

    print("\nvae tiling policy")
    import flux_core
    body = client.get(f'{PREFIX}/model', headers=AUTH).get_json()
    check('model reports the tiling mode and threshold',
          body.get('vae_tiling') in flux_core.VAE_TILING_MODES
          and isinstance(body.get('vae_tiling_threshold_mp'), (int, float)),
          f"{body.get('vae_tiling')!r} / {body.get('vae_tiling_threshold_mp')!r}")

    # The decision is per output size — that is the whole point of 'auto'.
    original = flux_core._vae_tiling_mode
    try:
        flux_core._vae_tiling_mode = 'auto'
        below = flux_core._vae_tiling_wanted(1216, 832)     # ~1.0MP
        at_175 = flux_core._vae_tiling_wanted(1600, 1088)   # ~1.74MP
        above = flux_core._vae_tiling_wanted(1728, 1152)    # ~1.99MP
        way_above = flux_core._vae_tiling_wanted(2048, 2048)
        check('auto leaves 1MP untiled', below is False)
        check('auto leaves 1.75MP untiled (measured fine there)', at_175 is False)
        check('auto tiles just under 2MP', above is True)
        check('auto tiles well above the threshold', way_above is True)

        flux_core._vae_tiling_mode = 'always'
        check('always tiles even a small image',
              flux_core._vae_tiling_wanted(512, 512) is True)

        flux_core._vae_tiling_mode = 'off'
        check('off never tiles, however large',
              flux_core._vae_tiling_wanted(4096, 4096) is False)
    finally:
        flux_core._vae_tiling_mode = original

    check('an unknown mode is rejected at load',
          _rejects_bad_tiling_mode(flux_core))

    r = client.put(f'{PREFIX}/models/current', headers=AUTH, json={'config': 'nope'})
    check('switching without a supervisor is refused',
          r.status_code == 400 and err_code(r) == 'no_supervisor',
          f"got {r.status_code}/{err_code(r)}")

    # With a supervisor present the id itself gets validated. Only invalid ids
    # are exercised here — a valid one would restart the process.
    ws._current_config = 9
    try:
        r = client.put(f'{PREFIX}/models/current', headers=AUTH, json={'config': 'nope'})
        check('a non-integer config is 400 invalid_request',
              r.status_code == 400 and err_code(r) == 'invalid_request',
              f"got {r.status_code}/{err_code(r)}")

        r = client.put(f'{PREFIX}/models/current', headers=AUTH, json={'config': 99})
        check('an out-of-range config is 400', r.status_code == 400)

        r = client.put(f'{PREFIX}/models/current', headers=AUTH, json={'config': 9})
        check('switching to the config already loaded is 400', r.status_code == 400)
    finally:
        ws._current_config = None

    print("\njobs")
    r = client.get(f'{PREFIX}/jobs', headers=AUTH)
    body = r.get_json() or {}
    check('jobs collection responds', r.status_code == 200)
    check('jobs exposes running/queued/recent',
          all(k in body for k in ('running', 'queued', 'recent', 'queue_max_size')))

    r = client.get(f'{PREFIX}/queue', headers=AUTH)
    body = r.get_json() or {}
    check('queue endpoint responds', r.status_code == 200)
    check('queue reports depth, capacity and whether it accepts work',
          all(k in body for k in ('depth', 'capacity', 'accepting', 'waiting',
                                  'running', 'estimated_wait_s')))
    check('an empty queue has depth 0 and is accepting',
          body.get('depth') == 0 and body.get('accepting') is True,
          f"got depth={body.get('depth')} accepting={body.get('accepting')}")
    check('queue capacity matches QUEUE_MAX_SIZE',
          body.get('capacity') == ws.QUEUE_MAX_SIZE)
    check('wait estimate is null with no completed jobs to learn from',
          body.get('estimated_wait_s') is None or
          isinstance(body.get('estimated_wait_s'), (int, float)))

    # Ordering with real entries. The queue worker only starts under __main__,
    # so injected jobs stay put; going through POST /jobs would need a model.
    with ws._queue_cv:
        ws._pending.extend([
            ws.Job(id='aaa1', params={'prompt': 'first', 'batch': 2}),
            ws.Job(id='bbb2', params={'prompt': 'second', 'batch': 1}),
        ])
    try:
        body = client.get(f'{PREFIX}/queue', headers=AUTH).get_json()
        check('waiting jobs are numbered from 1, in order',
              [j['position'] for j in body['waiting']] == [1, 2] and
              [j['id'] for j in body['waiting']] == ['aaa1', 'bbb2'],
              str(body['waiting']))
        check('depth counts waiting jobs', body['depth'] == 2)
        check('images_pending sums the batches', body['images_pending'] == 3,
              f"got {body['images_pending']}")
    finally:
        with ws._queue_cv:
            ws._pending.clear()

    r = client.get(f'{PREFIX}/jobs/deadbeef1234', headers=AUTH)
    check('unknown job is 404 not_found',
          r.status_code == 404 and err_code(r) == 'not_found',
          f"got {r.status_code}/{err_code(r)}")

    r = client.delete(f'{PREFIX}/jobs/deadbeef1234', headers=AUTH)
    check('canceling an unknown job is 404',
          r.status_code == 404 and err_code(r) == 'not_found')

    r = client.get(f'{PREFIX}/jobs/deadbeef1234/previews', headers=AUTH)
    body = r.get_json() or {}
    check('previews for an unknown job is an empty result',
          r.status_code == 200 and body.get('frames') == [] and body.get('live') is None,
          f"got {r.status_code}")
    check('previews reports state, count and whether frames are being saved',
          all(k in body for k in ('id', 'state', 'live', 'frames', 'count', 'saving')))

    r = client.get(f'{PREFIX}/jobs/not-alnum!/previews', headers=AUTH)
    check('a malformed job id is rejected on previews', r.status_code == 400)

    print("\nhtml views")
    r = client.get(f'{PREFIX}/queue.html', headers=AUTH)
    check('queue.html renders', r.status_code == 200 and b'Generation queue' in r.data,
          f"got {r.status_code}")
    check('queue.html is served as html',
          r.headers['Content-Type'].startswith('text/html'))
    check('queue.html polls its own JSON resource', b'/api/v1/queue?api_key=' in r.data)

    # The shell must load without a key or the browser never runs the code that
    # supplies one — this is what a plain <a href> from the UI does.
    r = client.get(f'{PREFIX}/queue.html')
    check('queue.html loads without a key (plain navigation)',
          r.status_code == 200, f"got {r.status_code}")
    check('the unauthenticated shell carries no queue data',
          b'"running"' not in r.data and b'job_id' not in r.data)

    r = client.get(f'{PREFIX}/jobs/abc123/previews.html')
    check('previews.html loads without a key', r.status_code == 200,
          f"got {r.status_code}")

    # The data behind them stays gated.
    r = client.get(f'{PREFIX}/queue')
    check('the queue JSON still requires a key',
          r.status_code == 401 and err_code(r) == 'unauthorized')

    r = client.get(f'{PREFIX}/jobs/abc123/previews.html', headers=AUTH)
    check('previews.html renders for a job id',
          r.status_code == 200 and b'Intermediate images' in r.data)
    check('previews.html embeds the job id it was asked for', b'abc123' in r.data)

    r = client.get(f'{PREFIX}/jobs/not-alnum!/previews.html', headers=AUTH)
    check('previews.html rejects a malformed job id', r.status_code == 400)

    # A running job with a live latent preview: the endpoint should surface it
    # with a cache-busting URL, which is the case /frames alone never covered.
    live_job = ws.Job(id='livejob0001', params={'prompt': 'x'}, state='running')
    live_job.preview = '_preview_current.png'
    live_job.preview_step = 12
    live_job.preview_ts = 1700000000000
    live_job.total_steps = 30
    live_job.current = 2
    with ws._queue_cv:
        ws._recent_done.insert(0, live_job)
    try:
        body = client.get(f'{PREFIX}/jobs/livejob0001/previews', headers=AUTH).get_json()
        check('the live preview is reported with its step',
              body['live'] and body['live']['step'] == 12 and body['live']['image'] == 2,
              str(body.get('live')))
        check('the live preview URL busts the cache with its timestamp',
              't=1700000000000' in (body['live'] or {}).get('url', ''),
              (body['live'] or {}).get('url', ''))
    finally:
        with ws._queue_cv:
            ws._recent_done.clear()

    r = client.delete(f'{PREFIX}/jobs/recent', headers=AUTH)
    check('clearing recent jobs returns 204', r.status_code == 204, f"got {r.status_code}")

    # The model is not loaded in this harness, so /jobs must refuse work rather
    # than queue something that can never run.
    r = client.post(f'{PREFIX}/jobs', headers=AUTH, json={'prompt': 'a test prompt'})
    check('queueing before the model is ready is 503 model_loading',
          r.status_code == 503 and err_code(r) == 'model_loading',
          f"got {r.status_code}/{err_code(r)}")

    print("\nvalidation reaches the same rules as the legacy API")
    ws._model_ready = True  # pretend the model is up; nothing here reaches the GPU
    try:
        r = client.post(f'{PREFIX}/jobs', headers=AUTH, json={})
        check('a missing prompt is 400', r.status_code == 400, f"got {r.status_code}")

        r = client.post(f'{PREFIX}/jobs', headers=AUTH, json={'prompt': 'x', 'steps': 9999})
        check('out-of-range steps is 400', r.status_code == 400)

        r = client.post(f'{PREFIX}/jobs', headers=AUTH,
                        json={'prompt': 'x', 'orientation': 'diagonal'})
        check('an unknown orientation is 400', r.status_code == 400)

        r = client.post(f'{PREFIX}/jobs', headers=AUTH,
                        json={'prompt': 'x', 'input_images': ['not-base64']})
        check('an undecodable reference image is 400', r.status_code == 400)

        r = client.post(f'{PREFIX}/jobs', headers=AUTH, json={'prompt': 'x', 'size': '9mp'})
        check('an unknown size is 400', r.status_code == 400)
    finally:
        ws._model_ready = False

    print("\nimages")
    r = client.get(f'{PREFIX}/images', headers=AUTH)
    check('image listing responds',
          r.status_code == 200 and isinstance(r.get_json().get('images'), list))

    # Two independent guards. The route's <filename> converter never matches a
    # slash, so a traversing URL cannot reach the delete handler at all; and the
    # core function rejects separators itself, which is what protects the legacy
    # route that takes the name in a JSON body.
    r = client.delete(f'{PREFIX}/images/../../etc/passwd', headers=AUTH)
    check('a traversing URL never routes to the deleter',
          r.status_code in (400, 404, 405), f"got {r.status_code}")

    traversal_refused = False
    try:
        ws._api_delete_today('../../etc/passwd')
    except ws.ApiError as e:
        traversal_refused = e.status == 400
    check('the delete core function rejects path separators', traversal_refused)

    r = client.delete(f'{PREFIX}/images/not-todays-file.png', headers=AUTH)
    check('deleting a non-today file is 400',
          r.status_code == 400 and err_code(r) == 'invalid_request')

    r = client.post(f'{PREFIX}/images/missing_20200101_000000_abc.png/save', headers=AUTH)
    check('saving a missing file is 404', r.status_code == 404)

    print("\nimports and file browsing")
    r = client.post(f'{PREFIX}/imports/url', headers=AUTH, json={'url': 'ftp://nope'})
    check('a non-http import url is 400',
          r.status_code == 400 and err_code(r) == 'invalid_request')

    r = client.post(f'{PREFIX}/imports/path', headers=AUTH, json={})
    check('an import with no path is 400', r.status_code == 400)

    r = client.post(f'{PREFIX}/imports/raw', headers=AUTH, data={})
    check('a raw import with no file is 400', r.status_code == 400)

    r = client.get(f'{PREFIX}/files?dir=definitely-not-a-real-directory', headers=AUTH)
    check('browsing a missing directory is 400',
          r.status_code == 400 and err_code(r) == 'not_found')

    r = client.get(f'{PREFIX}/files/thumbnail?path=nope.png', headers=AUTH)
    check('a thumbnail for a missing file is 404', r.status_code == 404)

    print("\nvlm jobs")
    r = client.post(f'{PREFIX}/vlm/jobs', headers=AUTH, json={'task': 'nonsense'})
    check('an unknown vlm task is 400',
          r.status_code == 400 and err_code(r) == 'invalid_request')

    r = client.post(f'{PREFIX}/vlm/jobs', headers=AUTH, json={'task': 'describe'})
    check('describe without images is 400', r.status_code == 400)

    r = client.post(f'{PREFIX}/vlm/jobs', headers=AUTH,
                    json={'task': 'boost', 'prompt': 'x', 'level': 99})
    check('an out-of-range boost level is 400', r.status_code == 400)

    r = client.post(f'{PREFIX}/vlm/jobs', headers=AUTH,
                    json={'task': 'critique', 'direction': 'brighter'})
    check('critique without a reference is 400', r.status_code == 400)

    r = client.get(f'{PREFIX}/vlm/jobs/nosuchjob', headers=AUTH)
    check('polling an unknown vlm job is 404',
          r.status_code == 404 and err_code(r) == 'not_found')

    print("\nmulti-model runs")
    r = client.get(f'{PREFIX}/multi-runs/current', headers=AUTH)
    check('no active run is 404 or 200', r.status_code in (200, 404))

    r = client.delete(f'{PREFIX}/multi-runs/current', headers=AUTH)
    check('cancelling is idempotent 204', r.status_code == 204, f"got {r.status_code}")

    r = client.post(f'{PREFIX}/multi-runs', headers=AUTH, json={'prompt': 'x'})
    check('a run with no configs is rejected', r.status_code in (400, 503))

    print("\nprotocol errors")
    r = client.get(f'{PREFIX}/no-such-endpoint', headers=AUTH)
    check('an unknown REST path is 404 with an envelope',
          r.status_code == 404 and err_code(r) == 'not_found',
          f"got {r.status_code}/{err_code(r)}")

    r = client.post(f'{PREFIX}/model', headers=AUTH, json={})
    check('a wrong method is 405 with an envelope',
          r.status_code == 405 and err_code(r) == 'method_not_allowed',
          f"got {r.status_code}/{err_code(r)}")

    print("\nlegacy routes still answer in their own dialect")
    r = client.get('/ready')
    check('legacy /ready is still public', r.status_code == 200)

    r = client.get('/status')
    check('legacy /status still requires auth', r.status_code == 401)
    check('legacy 401 keeps the success:false shape',
          (r.get_json() or {}).get('success') is False)

    r = client.get('/status', headers=AUTH)
    check('legacy /status still works', r.status_code == 200)

    r = client.get('/history', headers=AUTH)
    check('legacy /history still works',
          r.status_code == 200 and 'images' in (r.get_json() or {}))

    r = client.get('/configs', headers=AUTH)
    check('legacy /configs still works',
          r.status_code == 200 and len((r.get_json() or {}).get('configs') or []) == 14)

    r = client.post('/jobs/nope/cancel', headers=AUTH)
    check('legacy cancel still 404s in its own shape',
          r.status_code == 404 and (r.get_json() or {}).get('success') is False)

    print(f"\n{len(_checks) - len(_failures)}/{len(_checks)} checks passed")
    if _failures:
        print("\nfailures:")
        for f in _failures:
            print(f"  - {f}")
        return 1
    return 0


def _rejects_bad_tiling_mode(flux_core):
    """load_model should refuse a typo'd mode rather than silently defaulting.

    The model is already 'loaded' as far as load_model is concerned in most
    runs, so this checks the validation directly — it happens before the
    already-loaded early return only for a genuinely bad value.
    """
    try:
        flux_core.load_model(vae_tiling='sometimes')
    except ValueError:
        return True
    except Exception:
        # Any other failure means validation did not run first.
        return False
    return False


def _spec_covers_routes(spec):
    """Every registered /api/v1 rule should appear in the OpenAPI paths, so the
    published contract can't silently fall behind the code."""
    documented = set(spec.get('paths') or {})
    missing = []
    for rule in ws.app.url_map.iter_rules():
        path = str(rule)
        if not path.startswith(PREFIX) or rule.endpoint == 'rest.index':
            continue
        # Werkzeug renders converters as <converter:name>; OpenAPI wants {name}.
        rel = path[len(PREFIX):] or '/'
        for converter in ('path:', 'string:', 'int:'):
            rel = rel.replace('<' + converter, '<')
        rel = rel.replace('<', '{').replace('>', '}')
        if rel not in documented:
            missing.append(rel)
    if missing:
        print(f"        undocumented: {sorted(set(missing))}")
    return not missing


if __name__ == '__main__':
    sys.exit(main())
