"""Tests for the sysmem-fallback detector (web_server's SLOW_STEP_* block).

Runs with no model and no GPU: the detector is pure arithmetic over per-step
wall times, so it can be fed the timings from real incidents instead of a GPU.
The numbers below are measured, not invented — 0.55s/step is this box's healthy
1024x1024 pace, and 47.4 and 83.0 are two of the runs that made the server look
frozen (2026-09-11 21:51 and 2026-09-07 08:15).

    python test_gpu_pace.py
"""

import os
import sys

os.environ.setdefault('FLUX_API_KEY', 'test-key-for-pace-tests')
os.environ.setdefault('NOTE_URL', 'off')

import web_server as ws  # noqa: E402 — must follow the FLUX_API_KEY default

HEALTHY = 0.55   # s/step at 1024x1024
FROZEN = 47.4    # s/step, same size, 2026-09-11
_failures = []
_count = 0


def check(name, condition, detail=''):
    global _count
    _count += 1
    if condition:
        print(f"  ok    {name}")
    else:
        _failures.append(name)
        print(f"  FAIL  {name}{' — ' + detail if detail else ''}")


def job(total_steps=25):
    j = ws.Job(id='pace', params={})
    j.total_steps = total_steps
    return j


def run(j, width, height, per_step, steps):
    """Feed `steps` identical steps in. Returns the step number it failed on,
    or None if the job was allowed to finish."""
    state = {'t': None, 'slow': 0}
    for _ in range(steps):
        j.step_times.append(per_step)
        try:
            ws._check_step_pace(j, width, height, per_step, state)
        except ws.GpuDegraded as e:
            run.message = str(e)
            return len(j.step_times)
    return None


def main():
    ws._step_baseline.clear()

    print("\na healthy run")
    j = job()
    check("25 steps at the normal pace are never failed",
          run(j, 1024, 1024, HEALTHY, 25) is None)
    ws._record_baseline(1024, 1024, ws._median(j.step_times[ws.SLOW_STEP_WARMUP:]))
    check("it teaches the baseline for its resolution",
          ws._baseline_for(1024, 1024) == HEALTHY)

    print("\nthe run that looked like a freeze")
    tripped = run(job(), 1024, 1024, FROZEN, 25)
    check("47.4s/step against a 0.55s baseline is caught", tripped is not None)
    check("caught once past warmup and the streak, not on the first slow step",
          tripped == ws.SLOW_STEP_WARMUP + ws.SLOW_STEP_STREAK)
    check("the job dies in under a minute of wasted GPU, not 20",
          tripped * FROZEN < 5 * 60,
          f"{tripped} steps x {FROZEN}s")
    check("the error says what to do about it",
          'close GPU-heavy apps' in getattr(run, 'message', ''))

    print("\nthe baseline cannot be poisoned")
    ws._record_baseline(1024, 1024, FROZEN)
    check("a degraded run does not raise the bar for the next one",
          ws._baseline_for(1024, 1024) == HEALTHY)

    print("\nfalse positives")
    j = job()
    state = {'t': None, 'slow': 0}
    survived = True
    for dt in [HEALTHY] * 5 + [40.0] + [HEALTHY] * 5:
        j.step_times.append(dt)
        try:
            ws._check_step_pace(j, 1024, 1024, dt, state)
        except ws.GpuDegraded:
            survived = False
    check("one slow step (another process spiking) is tolerated", survived)
    check("an unmeasured 4MP size is not judged against the backstop",
          run(job(), 2048, 2048, 25.0, 10) is None)

    print("\nthe first run at a size, with no baseline yet")
    check("83s/step is still caught by the absolute backstop",
          run(job(), 1360, 768, 83.0, 10) is not None)
    check("a slow-but-plausible first run is left alone",
          run(job(), 1360, 768, ws.SLOW_STEP_ABS_S - 1, 10) is None)

    print("\nthe kill switch")
    factor = ws.SLOW_STEP_FACTOR
    ws.SLOW_STEP_FACTOR = 0.0
    try:
        check("FLUX_SLOW_STEP_FACTOR=0 disables the check entirely",
              run(job(), 1024, 1024, FROZEN, 10) is None)
    finally:
        ws.SLOW_STEP_FACTOR = factor

    print()
    if _failures:
        print(f"{len(_failures)} of {_count} checks FAILED: {', '.join(_failures)}")
        return 1
    print(f"{_count}/{_count} checks passed")
    return 0


if __name__ == '__main__':
    sys.exit(main())
