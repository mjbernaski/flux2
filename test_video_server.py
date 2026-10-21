"""Check video_server.py's request validation, workflow patching and mode gate.

No GPU, no ComfyUI and no model: it runs against Flask's test client with the
ports faked, so it covers the parts that must not get it wrong and that are
otherwise only reachable by actually switching the box into video mode.

Three things here are worth a test of their own:

- the mode gate, because the failure it prevents is silent. Two models on one
  32GB card does not raise - the Windows driver pages one out and every step
  runs 8-200x slow with no error anywhere (see CLAUDE.md on sysmem fallback).
- frame snapping, because H3 is trained on a 17k+5 grid and an off-grid length
  is accepted by the sampler and then drifts at the tail.
- workflow patching, because the map file is hand-written against an exported
  graph and every way it can be stale should name itself rather than throwing a
  KeyError or, worse, quietly detaching a link.

Run it after touching validate_request, patch_workflow or the mode endpoints.
"""
import json
import os
import sys
import tempfile

REPO = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO)
os.environ.setdefault('FLUX_API_KEY', 'test-key-not-real')
# video_server reads .mode/.switching from its own directory and POST /mode
# would launch the real switch script. Run from a temp dir so neither this test
# nor a bug in it can stop the live server.
WORKDIR = tempfile.mkdtemp(prefix='flux-video-test-')
os.chdir(WORKDIR)

fails = []


def check(label, cond, detail=''):
    print(('  ok    ' if cond else '  FAIL  ') + label + (f'  [{detail}]' if detail and not cond else ''))
    if not cond:
        fails.append(label)


import video_server as vs

vs.HERE = WORKDIR
vs.MODE_FILE = os.path.join(WORKDIR, '.mode')
vs.SWITCHING_FILE = os.path.join(WORKDIR, '.switching')
vs.WORKFLOW_DIR = os.path.join(WORKDIR, 'workflows')
vs.VIDEO_DIR = os.path.join(WORKDIR, 'video')
os.makedirs(vs.WORKFLOW_DIR, exist_ok=True)
os.makedirs(vs.VIDEO_DIR, exist_ok=True)

KEY = {'X-API-Key': 'test-key-not-real'}
client = vs.app.test_client()

# Nothing is actually listening in a test run, so port probes are driven from
# this dict instead of the network.
PORTS = {vs.FLUX_PORT: False, vs.COMFY_PORT: False}
vs.port_is_up = lambda port, host='127.0.0.1': PORTS.get(port, False)


print('request validation')
for payload, want in (
    ({}, 'prompt is required'),
    ({'prompt': '   '}, 'prompt is required'),
    ({'prompt': 'x' * 4001}, 'at most 4000'),
    ({'prompt': 'a', 'steps': 0}, 'between 1 and'),
    ({'prompt': 'a', 'steps': 999}, 'between 1 and'),
    ({'prompt': 'a', 'steps': 'many'}, 'must be an integer'),
    ({'prompt': 'a', 'seed': 'abc'}, 'must be an integer'),
    ({'prompt': 'a', 'width': 10}, 'at least 32'),
    ({'prompt': 'a', 'width': 4096}, 'at most 1344'),
    ({'prompt': 'a', 'width': 1344, 'height': 1344}, 'caps at'),
    ({'prompt': 'a', 'frames': 5}, 'must land between'),
    ({'prompt': 'a', 'frames': 9999}, 'must land between'),
):
    try:
        vs.validate_request(payload)
        check(f'{json.dumps(payload)[:46]} -> {want}', False, 'accepted')
    except vs.WorkflowError as exc:
        check(f'{json.dumps(payload)[:46]} -> {want}', want in str(exc), str(exc))

ok = vs.validate_request({'prompt': 'a fox'})
check('defaults are 864x480, 124 frames, 10 steps',
      (ok['width'], ok['height'], ok['frames'], ok['steps']) == (864, 480, 124, 10), repr(ok))
check('a seedless request draws one', isinstance(ok['seed'], int) and ok['seed'] >= 0, repr(ok['seed']))
check('an empty negative becomes None', ok['negative'] is None, repr(ok['negative']))

print('\ndimension snapping (multiple of 32, inside 768x1344)')
for given, want in ((863, 864), (865, 864), (480, 480), (100, 96), (1344, 1344), (32, 32)):
    got = vs._snap_dim(given, 'width')
    check(f'{given} -> {want}', got == want, repr(got))

print('\nframe snapping (the 17k+5 grid H3 is trained on)')
for given, want in ((124, 124), (120, 124), (130, 124), (243, 243), (362, 362), (200, 192)):
    got = vs._snap_frames(given)
    check(f'{given} -> {want}', got == want, repr(got))
check('every snapped value is on the grid',
      all((vs._snap_frames(n) - vs.FRAME_BASE) % vs.FRAME_STEP == 0
          for n in range(vs.MIN_FRAMES, vs.MAX_FRAMES + 1)))

print('\nworkflow loading')
try:
    vs.load_workflow('nope')
    check('a missing graph names the path and Export (API)', False, 'no error')
except vs.WorkflowError as exc:
    check('a missing graph names the path and Export (API)',
          'Export (API)' in str(exc) and 'h3_t2va' not in str(exc).split('nope')[0], str(exc))

# The UI-format export is the mistake worth catching by name: it is also valid
# JSON, so without this check it fails much later as a confusing patch error.
with open(os.path.join(vs.WORKFLOW_DIR, 'ui.json'), 'w') as fh:
    json.dump({'nodes': [], 'links': []}, fh)
try:
    vs.load_workflow('ui')
    check('a UI-format export is rejected by name', False, 'accepted')
except vs.WorkflowError as exc:
    check('a UI-format export is rejected by name', 'Export (API)' in str(exc), str(exc))

GRAPH = {
    '6': {'class_type': 'CLIPTextEncode', 'inputs': {'text': '', 'clip': ['4', 1]},
          '_meta': {'title': 'Positive'}},
    '7': {'class_type': 'CLIPTextEncode', 'inputs': {'text': '', 'clip': ['4', 1]}},
    '40': {'class_type': 'EmptyH3LatentVideo',
           'inputs': {'width': 512, 'height': 512, 'length': 124, 'batch_size': 1}},
    '17': {'class_type': 'KSampler',
           'inputs': {'steps': 20, 'noise_seed': 0, 'model': ['41', 0]}},
}
MAP = {'prompt': '6.text', 'negative': '7.text', 'width': '40.width',
       'height': '40.height', 'frames': '40.length', 'steps': '17.steps',
       'seed': '17.noise_seed'}


def write_workflow(name, graph, field_map):
    with open(os.path.join(vs.WORKFLOW_DIR, name + '.json'), 'w') as fh:
        json.dump(graph, fh)
    if field_map is not None:
        with open(os.path.join(vs.WORKFLOW_DIR, name + '.map.json'), 'w') as fh:
            json.dump(field_map, fh)


write_workflow('h3_t2va', GRAPH, MAP)

print('\nworkflow patching')
graph, field_map = vs.load_workflow('h3_t2va')
params = vs.validate_request({'prompt': 'a fox', 'negative': 'blurry', 'width': 864,
                              'height': 480, 'frames': 243, 'steps': 10, 'seed': 42})
applied = vs.patch_workflow(graph, field_map, dict(params))
check('the prompt lands in the mapped node', graph['6']['inputs']['text'] == 'a fox',
      repr(graph['6']['inputs']['text']))
check('the negative lands in its own node', graph['7']['inputs']['text'] == 'blurry')
check('size and length land in the latent node',
      (graph['40']['inputs']['width'], graph['40']['inputs']['height'],
       graph['40']['inputs']['length']) == (864, 480, 243), repr(graph['40']['inputs']))
check('steps and seed land in the sampler',
      (graph['17']['inputs']['steps'], graph['17']['inputs']['noise_seed']) == (10, 42))
check('untouched inputs are left alone', graph['40']['inputs']['batch_size'] == 1)
check('links are not disturbed', graph['6']['inputs']['clip'] == ['4', 1])
check('every applied field is reported back', set(applied) >= set(MAP), repr(sorted(applied)))

print('\nstale and wrong map entries name themselves')
graph, _ = vs.load_workflow('h3_t2va')
for bad_map, want in (
    ({**MAP, 'prompt': '999.text'}, 'not in the graph'),
    ({**MAP, 'prompt': '6.nosuchinput'}, 'has inputs'),
    # A link is a wire to another node, not a literal. Overwriting one detaches
    # the graph and the clip silently ignores the setting.
    ({**MAP, 'prompt': '6.clip'}, 'wired to'),
    ({**MAP, 'prompt': 'notapair'}, 'must look like'),
):
    try:
        vs.patch_workflow(json.loads(json.dumps(GRAPH)), bad_map, dict(params))
        check(f'-> {want}', False, 'accepted')
    except vs.WorkflowError as exc:
        check(f'-> {want}', want in str(exc), str(exc))

for drop in ('prompt', 'seed', 'frames'):
    thin = {k: v for k, v in MAP.items() if k != drop}
    try:
        vs.patch_workflow(json.loads(json.dumps(GRAPH)), thin, dict(params))
        check(f'a map missing {drop!r} is an error', False, 'accepted')
    except vs.WorkflowError as exc:
        check(f'a map missing {drop!r} is an error',
              'missing required field' in str(exc) and drop in str(exc), str(exc))

print('\nauth')
check('/mode is readable without a key', client.get('/mode').status_code == 200)
check('/status needs a key', client.get('/status').status_code == 401)
check('/generate needs a key', client.post('/generate', json={'prompt': 'a'}).status_code == 401)
check('a wrong key is still 401',
      client.get('/status', headers={'X-API-Key': 'wrong'}).status_code == 401)

print('\nthe mode gate (the one that prevents a silent 8-200x slowdown)')
PORTS[vs.FLUX_PORT], PORTS[vs.COMFY_PORT] = True, False
res = client.post('/generate', json={'prompt': 'a fox'}, headers=KEY)
check('flux up -> 409, not a submission', res.status_code == 409, str(res.status_code))
check('... and it says what to do next', 'mode' in res.get_json().get('error', ''),
      res.get_json().get('error'))

PORTS[vs.FLUX_PORT], PORTS[vs.COMFY_PORT] = False, False
res = client.post('/generate', json={'prompt': 'a fox'}, headers=KEY)
check('comfy down -> 503', res.status_code == 503, str(res.status_code))

# A switch in flight is both ports down for a minute or two; submitting into
# that window would queue against a ComfyUI that is still loading.
with open(vs.SWITCHING_FILE, 'w') as fh:
    fh.write('video 2026-10-21 00:00:00')
PORTS[vs.COMFY_PORT] = True
res = client.post('/generate', json={'prompt': 'a fox'}, headers=KEY)
check('a switch in flight -> 409', res.status_code == 409, str(res.status_code))
check('/mode reports the target while switching',
      client.get('/mode').get_json().get('switching') == 'video',
      repr(client.get('/mode').get_json().get('switching')))
res = client.post('/mode', json={'mode': 'flux'}, headers=KEY)
check('a second switch is refused while one is in flight', res.status_code == 409, str(res.status_code))
os.remove(vs.SWITCHING_FILE)

print('\nmode file')
check('an absent .mode reads as flux', vs.read_mode() == 'flux')
for raw, want in (('video', 'video'), ('flux', 'flux'), ('VIDEO\n', 'video'),
                  ('banana', 'flux'), ('', 'flux')):
    with open(vs.MODE_FILE, 'w') as fh:
        fh.write(raw)
    check(f'.mode {raw!r} -> {want}', vs.read_mode() == want, vs.read_mode())
os.remove(vs.MODE_FILE)

res = client.post('/mode', json={'mode': 'sideways'}, headers=KEY)
check('an unknown mode is a 400', res.status_code == 400, str(res.status_code))

print('\nserved filenames')
for name, allowed in (('h3_x.mp4', True), ('h3_x.webm', True), ('h3_x.png', False),
                      ('../secrets.mp4', False), ('.hidden/x.mp4', False),
                      ('.mp4', False), ('', False), ('sub\\x.mp4', False)):
    got = vs._safe_video_name(name) is not None
    check(f'{name!r} -> {"served" if allowed else "rejected"}', got == allowed, repr(got))

# The shipped pairs are real files generated against a specific ComfyUI, so
# they can rot: a node id that moved in an upgrade, or a typo in a map, would
# otherwise only show up as a failed submission. Check them here instead.
print('\nprompt boost')
# The VLM lives on another box, so only the paths that fail before it is
# reached are exercised here; the rewrite itself needs a live endpoint.
check('/boost needs a key', client.post('/boost', json={'prompt': 'a'}).status_code == 401)
for payload, want in (
    ({}, 'prompt is required'),
    ({'prompt': '  '}, 'prompt is required'),
    ({'prompt': 'a', 'level': 0}, 'level must be an integer 1-5'),
    ({'prompt': 'a', 'level': 6}, 'level must be an integer 1-5'),
    ({'prompt': 'a', 'level': 'lots'}, 'level must be an integer 1-5'),
    ({'prompt': 'a', 'steer': 'x' * (vs.BOOST_STEER_MAX + 1)}, 'at most'),
    ({'prompt': 'a', 'input_images': [123]}, 'not a base64 image'),
    ({'prompt': 'a', 'input_images': ['not base64 at all']}, 'not a decodable base64 image'),
    ({'prompt': 'a', 'frames': 3}, 'must land between'),
    ({'prompt': 'a', 'width': 9999}, 'at most'),
):
    res = client.post('/boost', json=payload, headers=KEY)
    body = res.get_json() or {}
    check(f'{json.dumps(payload)[:44]} -> 400 {want!r}',
          res.status_code == 400 and want in (body.get('error') or ''),
          f'{res.status_code} {body.get("error")}')

# A steer at exactly the cap is allowed; it is the character after it that is
# a draft prompt in the wrong field.
res = client.post('/boost', json={'prompt': 'a', 'steer': 'x' * vs.BOOST_STEER_MAX}, headers=KEY)
check('a steer at exactly the cap is not rejected for length',
      'at most' not in ((res.get_json() or {}).get('error') or ''),
      (res.get_json() or {}).get('error'))

import edit_loop
check('edit_loop has both video families',
      'h3' in edit_loop.BOOST_GUIDANCE and 'h3-i2v' in edit_loop.BOOST_GUIDANCE)
check('the video guidance asks for timecoded shots and audio',
      all(w in edit_loop.BOOST_GUIDANCE['h3'].lower()
          for w in ('timecode', 'audio', 'storyboard')))
check('the i2v family builds on the t2v one and adds the first frame',
      edit_loop.BOOST_GUIDANCE['h3-i2v'].startswith(edit_loop.BOOST_GUIDANCE['h3'])
      and 'first frame' in edit_loop.BOOST_GUIDANCE['h3-i2v'])
import inspect
params = inspect.signature(edit_loop.vlm_boost).parameters
check("vlm_boost takes medium, defaulting to 'image' so image callers are unchanged",
      'medium' in params and params['medium'].default == 'image')

print('\nthe workflows shipped in the repo')
REAL = os.path.join(REPO, 'workflows')
vs.WORKFLOW_DIR = REAL
shipped = sorted(f[:-5] for f in os.listdir(REAL)
                 if f.endswith('.json') and not f.endswith('.map.json'))
check('both pairs are present', shipped == ['h3_fl2va', 'h3_t2va'], repr(shipped))

for name in shipped:
    try:
        graph, field_map = vs.load_workflow(name)
    except vs.WorkflowError as exc:
        check(f'{name} loads', False, str(exc))
        continue
    check(f'{name} loads ({len(graph)} nodes)', True)
    check(f'{name} map has no note keys left in it',
          not any(k.startswith('_') for k in field_map), repr(sorted(field_map)))
    try:
        vs.patch_workflow(graph, field_map,
                          dict(vs.validate_request({'prompt': 'a fox', 'seed': 7})))
        check(f'{name} patches cleanly', True)
    except vs.WorkflowError as exc:
        check(f'{name} patches cleanly', False, str(exc))
        continue

    # Whatever the ids are, the values have to have reached real H3 nodes.
    h3 = [n for n in graph.values() if n['class_type'] == 'MiniMaxH3ImageToVideo']
    check(f'{name} has exactly one MiniMaxH3ImageToVideo', len(h3) == 1, str(len(h3)))
    if h3:
        check(f'{name} prompt reached it', h3[0]['inputs'].get('prompt') == 'a fox',
              repr(h3[0]['inputs'].get('prompt'))[:60])
    noise = [n for n in graph.values() if n['class_type'] == 'RandomNoise']
    check(f'{name} seed reached RandomNoise',
          bool(noise) and noise[0]['inputs'].get('noise_seed') == 7)
    # steps must land on a literal. Pointing it at a wire would mean the switch
    # was followed to the wrong end and the step count silently ignored.
    node_id, _, input_name = field_map['steps'].partition('.')
    check(f'{name} steps target is a literal',
          not isinstance(graph[node_id]['inputs'][input_name], list))
    check(f'{name} has a SaveVideo',
          any(n['class_type'] == 'SaveVideo' for n in graph.values()))
    dangling = [(nid, k) for nid, n in graph.items() for k, v in n['inputs'].items()
                if isinstance(v, list) and v[0] not in graph]
    check(f'{name} has no dangling links', not dangling, repr(dangling[:3]))

check('only h3_fl2va takes a first frame',
      'start_image' in vs.load_workflow('h3_fl2va')[1]
      and 'start_image' not in vs.load_workflow('h3_t2va')[1])

print(f"\n{'FAILED: ' + ', '.join(fails) if fails else 'all checks passed'}")
sys.exit(1 if fails else 0)
