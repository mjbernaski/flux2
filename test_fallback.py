"""Check the load-failure fallback's decision table.

No GPU and no model: it only exercises _fallback_after_load_failure's decision
table, which is the part that must never get it wrong — a fallback that fires
when it should not relaunches the server forever, and one that declines when it
should fire leaves the box serving nothing. Run it after touching
FALLBACK_CONFIG or the model-load error path.
"""
import os, sys, tempfile

REPO = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO)
os.environ.setdefault('FLUX_API_KEY', 'test-key-not-real')
# The fallback writes .next_config in the CWD, and that file is a live switch
# request: left in the repo, the supervisor would act on it at the next restart.
# Run from a temp directory so the test cannot hand the server an instruction.
os.chdir(tempfile.mkdtemp(prefix='flux-fallback-test-'))

fails = []
def check(label, cond, detail=''):
    print(('  ok    ' if cond else '  FAIL  ') + label + (f'  [{detail}]' if detail and not cond else ''))
    if not cond:
        fails.append(label)

import web_server as ws

print('FLUX_FALLBACK_CONFIG parsing')
check('defaults to 15', ws._fallback_config_from_env() == 15, repr(ws._fallback_config_from_env()))
for raw, want in (('0', None), ('off', None), ('none', None), ('false', None),
                  ('9', 9), ('15', 15), ('99', 15), ('banana', 15), ('', None)):
    os.environ['FLUX_FALLBACK_CONFIG'] = raw
    got = ws._fallback_config_from_env()
    check(f'{raw!r} -> {want}', got == want, repr(got))
os.environ.pop('FLUX_FALLBACK_CONFIG', None)

# Never let the real os._exit run: the fallback arms a 3s timer onto it.
exits, timers = [], []
real_timer = ws.threading.Timer
class FakeTimer:
    def __init__(self, delay, fn): timers.append(delay); self.delay = delay
    def start(self): exits.append('armed')
ws.threading.Timer = FakeTimer

def attempt(current, fallback):
    del exits[:], timers[:]
    ws._current_config = current
    ws.FALLBACK_CONFIG = fallback
    if os.path.exists(ws.SWITCH_CONFIG_FILE):
        os.remove(ws.SWITCH_CONFIG_FILE)
    acted = ws._fallback_after_load_failure(RuntimeError('out of memory'))
    wrote = (open(ws.SWITCH_CONFIG_FILE).read().strip()
             if os.path.exists(ws.SWITCH_CONFIG_FILE) else None)
    return acted, wrote, list(exits), list(timers)

print('\nfallback decision table')
acted, wrote, ex, tm = attempt(9, 15)
check('config 9 failing falls back to 15', acted and wrote == '15' and ex, f'{acted=} {wrote=} {ex=}')
check('the exit is delayed, not immediate', tm == [3.0], repr(tm))

acted, wrote, ex, _ = attempt(15, 15)
check('config 15 failing does NOT loop back into itself',
      acted is False and wrote is None and not ex, f'{acted=} {wrote=}')

acted, wrote, ex, _ = attempt(None, 15)
check('unsupervised (no FLUX_CONFIG) stays up to report the error',
      acted is False and wrote is None and not ex, f'{acted=} {wrote=}')

acted, wrote, ex, _ = attempt(9, None)
check('FLUX_FALLBACK_CONFIG=0 disables it',
      acted is False and wrote is None and not ex, f'{acted=} {wrote=}')

ws.threading.Timer = real_timer
if os.path.exists(ws.SWITCH_CONFIG_FILE):
    os.remove(ws.SWITCH_CONFIG_FILE)

print(f"\n{'FAILED: ' + ', '.join(fails) if fails else 'all checks passed'}")
sys.exit(1 if fails else 0)
