#!/usr/bin/env python3
"""Smoke-test every model option in run_server.sh, with a live HTML tracker.

Each menu configuration is loaded in an isolated worker subprocess (matching
how the real server runs: one process per model), generates one sample image
with a fixed prompt, and reports pass/fail with timings. Kontext (editor)
options are exercised with a synthetic reference image and an edit instruction.

While the sweep runs, server_smoke_test/index.html is a live tracking page:
it auto-refreshes and shows each option as pending / running (with phase and
elapsed time) / pass / fail. When the sweep finishes the page becomes a static
report.

Usage:
    python smoke_test_servers.py                # test all 12 menu options
    python smoke_test_servers.py 1 3 9          # test only options 1, 3, 9

Results merge into prior runs, so re-running a single option updates just its
card in the report.
"""

import sys
import os
import json
import time
import html
import datetime
import traceback
import subprocess

PROMPT = "a frog on a log in a bog"
EDIT_PROMPT = "make the red circle blue"   # used for Kontext (editor) options
WIDTH, HEIGHT = 1024, 768                  # landscape smoke-test size
DEFAULT_STEPS = 20                         # schnell/turbo override this internally
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "server_smoke_test")
RESULTS_JSON = os.path.join(OUT_DIR, "results.json")
POLL_INTERVAL_S = 2.0

# The menu options exactly as defined in run_server.sh start_server().
# raw_args are the flags run_server.sh passes to web_server.py.
# kind: 'txt2img' for normal generation, 'edit' for Kontext instruction editing.
MENU = [
    (1,  "FLUX.1 4-bit BNB",              [],                                        "txt2img"),
    (2,  "FLUX.1 Full",                   ["--full-model"],                          "txt2img"),
    (3,  "FLUX.1 GGUF Q8",                ["--gguf", "q8", "--local-encoder"],       "txt2img"),
    (4,  "FLUX.1-schnell",                ["--schnell", "--local-encoder"],          "txt2img"),
    (5,  "FLUX.1 + Uncensored",           ["--uncensored"],                          "txt2img"),
    (6,  "FLUX.2 4-bit BNB",              ["--flux2"],                               "txt2img"),
    (7,  "FLUX.2 Full + Turbo",           ["--flux2", "--full-model", "--turbo"],    "txt2img"),
    (8,  "FLUX.2 Full (no Turbo)",        ["--flux2", "--full-model", "--no-turbo"], "txt2img"),
    (9,  "FLUX.2-klein-9B",               ["--klein"],                               "txt2img"),
    (10, "FLUX.1 Kontext (editor)",       ["--kontext"],                             "edit"),
    (11, "FLUX.1 Kontext Full (bf16)",    ["--kontext", "--full-model"],             "edit"),
    (12, "Kontext Full + Uncensored",     ["--kontext", "--full-model", "--uncensored"], "edit"),
]


def derive_flags(raw_args):
    """Replicate web_server.py's argument derivation for a set of raw flags."""
    full_model = "--full-model" in raw_args
    flux2 = "--flux2" in raw_args
    schnell = "--schnell" in raw_args
    uncensored = "--uncensored" in raw_args
    klein = "--klein" in raw_args
    kontext = "--kontext" in raw_args
    local_encoder_flag = "--local-encoder" in raw_args
    turbo_flag = "--turbo" in raw_args
    no_turbo = "--no-turbo" in raw_args

    gguf = None
    if "--gguf" in raw_args:
        gguf = raw_args[raw_args.index("--gguf") + 1]

    # klein implies flux2 + full
    if klein:
        flux2 = True
        full_model = True

    local_encoder = local_encoder_flag or full_model or schnell or uncensored or kontext
    if uncensored and not full_model:
        full_model = True
    # Turbo LoRA is a FLUX.2-dev LoRA; not auto-enabled for klein (different arch)
    turbo = (turbo_flag or (flux2 and not klein)) and not no_turbo

    return {
        "full_model": full_model,
        "gguf_quant": gguf,
        "flux2": flux2,
        "schnell": schnell,
        "uncensored": uncensored,
        "klein": klein,
        "kontext": kontext,
        "local_encoder": local_encoder,
        "turbo": turbo,
    }


def _worker_result_path(num):
    return os.path.join(OUT_DIR, f"_result_{num:02d}.json")


def _worker_progress_path(num):
    return os.path.join(OUT_DIR, f"_progress_{num:02d}.json")


def _report_progress(num, phase):
    """Worker side: record the current phase so the parent's live page can show it."""
    try:
        with open(_worker_progress_path(num), "w") as f:
            json.dump({"phase": phase, "ts": time.time()}, f)
    except Exception:
        pass


def _make_edit_reference():
    """Synthetic reference image for Kontext: a red circle on a plain field."""
    from PIL import Image, ImageDraw
    img = Image.new("RGB", (WIDTH, HEIGHT), (235, 235, 220))
    draw = ImageDraw.Draw(img)
    cx, cy, r = WIDTH // 2, HEIGHT // 2, min(WIDTH, HEIGHT) // 4
    draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=(200, 30, 30))
    return img


def run_config_inprocess(num, name, raw_args, kind):
    """Load one config and generate the sample image (runs inside a worker
    subprocess so model/GPU state is fully isolated per config)."""
    _report_progress(num, "importing torch/diffusers")
    from flux_core import load_model, generate_image, load_turbo_lora, load_uncensored_lora

    flags = derive_flags(raw_args)
    cmd = "python web_server.py " + " ".join(raw_args) if raw_args else "python web_server.py"
    prompt = EDIT_PROMPT if kind == "edit" else PROMPT
    result = {
        "num": num, "name": name, "command": cmd, "flags": flags, "kind": kind,
        "prompt": prompt, "status": "fail", "error": None, "image": None,
        "load_time": None, "gen_time": None, "seed": None, "steps": None,
    }

    print(f"\n{'='*64}\n[{num}] {name}\n  {cmd}\n  flags: {flags}\n{'='*64}")
    try:
        _report_progress(num, "loading model")
        t0 = time.perf_counter()
        load_model(
            local_encoder=flags["local_encoder"], full_model=flags["full_model"],
            gguf_quant=flags["gguf_quant"], flux2=flags["flux2"],
            schnell=flags["schnell"], for_lora=flags["uncensored"],
            klein=flags["klein"], kontext=flags["kontext"],
        )
        if flags["turbo"]:
            _report_progress(num, "loading turbo LoRA")
            load_turbo_lora()
        if flags["uncensored"]:
            _report_progress(num, "loading uncensored LoRA")
            load_uncensored_lora()
        result["load_time"] = time.perf_counter() - t0
        print(f"  loaded in {result['load_time']:.1f}s")

        input_image = _make_edit_reference() if kind == "edit" else None
        _report_progress(num, "generating")
        t0 = time.perf_counter()
        image, seed, _ = generate_image(
            prompt, steps=DEFAULT_STEPS, width=WIDTH, height=HEIGHT,
            local_encoder=flags["local_encoder"], input_image=input_image,
        )
        result["gen_time"] = time.perf_counter() - t0
        result["seed"] = seed
        fname = f"option_{num:02d}.png"
        image.save(os.path.join(OUT_DIR, fname))
        result["image"] = fname
        result["status"] = "pass"
        print(f"  generated {image.size} in {result['gen_time']:.1f}s -> {fname}")
    except Exception as e:
        result["error"] = f"{type(e).__name__}: {e}"
        print(f"  FAILED: {result['error']}")
        traceback.print_exc()

    with open(_worker_result_path(num), "w") as f:
        json.dump(result, f)
    return result


def _fallback_result(num, name, raw_args, kind, error):
    flags = derive_flags(raw_args)
    cmd = "python web_server.py " + " ".join(raw_args) if raw_args else "python web_server.py"
    return {
        "num": num, "name": name, "command": cmd, "flags": flags, "kind": kind,
        "prompt": EDIT_PROMPT if kind == "edit" else PROMPT,
        "status": "fail", "error": error, "image": None,
        "load_time": None, "gen_time": None, "seed": None, "steps": None,
    }


def test_one(num, name, raw_args, kind, by_num, planned):
    """Parent side: run one config in an isolated subprocess, keeping the live
    HTML tracker updated while it runs. A crash (OOM kill, segfault) is
    captured as a failure rather than taking down the whole sweep."""
    rpath = _worker_result_path(num)
    ppath = _worker_progress_path(num)
    for p in (rpath, ppath):
        if os.path.exists(p):
            os.remove(p)
    logpath = os.path.join(OUT_DIR, f"_worker_{num:02d}.log")

    print(f"\n>>> [{num}] {name} — launching isolated worker...")
    started = time.time()
    with open(logpath, "w") as logf:
        proc = subprocess.Popen(
            [sys.executable, os.path.abspath(__file__), "--worker", str(num)],
            stdout=logf, stderr=subprocess.STDOUT,
        )
        while proc.poll() is None:
            phase = "starting worker"
            try:
                with open(ppath) as f:
                    phase = json.load(f).get("phase", phase)
            except Exception:
                pass
            running = {"num": num, "phase": phase, "elapsed": time.time() - started}
            write_html(by_num, planned, running=running)
            time.sleep(POLL_INTERVAL_S)

    if os.path.exists(rpath):
        with open(rpath) as f:
            return json.load(f)

    # Worker died before writing a result (OOM/segfault). Salvage the log tail.
    tail = ""
    try:
        with open(logpath) as f:
            tail = "".join(f.readlines()[-6:]).strip()
    except Exception:
        pass
    sig = f"worker exited with code {proc.returncode}"
    if proc.returncode in (-9, 137):
        sig = "worker killed (OOM, exit 137)"
    return _fallback_result(num, name, raw_args, kind, f"{sig}. Last output:\n{tail}")


def gpu_name():
    """GPU name via nvidia-smi (avoids creating a CUDA context in the parent)."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10,
        )
        name = out.stdout.strip().splitlines()[0].strip()
        return name or "GPU"
    except Exception:
        return "GPU"


_GPU_NAME = None


def _card_for(entry, result, running):
    """Render one option card in whichever state it is in."""
    num, name, raw_args, kind = entry
    cmd = "python web_server.py " + " ".join(raw_args) if raw_args else "python web_server.py"

    if running and running["num"] == num:
        state, badge = "running", "RUNNING"
        media = (
            '<div class="run"><div class="spinner"></div>'
            f'<div class="phase">{html.escape(running["phase"])}</div>'
            f'<div class="elapsed">{running["elapsed"]:.0f}s elapsed</div></div>'
        )
        meta_rows = [("Command", cmd)]
    elif result is None:
        state, badge = "pending", "PENDING"
        media = '<div class="pend">waiting…</div>'
        meta_rows = [("Command", cmd)]
    elif result["status"] == "pass":
        state, badge = "pass", "PASS"
        media = f'<img src="{html.escape(result["image"])}" alt="{html.escape(name)}">'
        meta_rows = [("Command", result["command"])]
        if result.get("load_time") is not None:
            meta_rows.append(("Load time", f"{result['load_time']:.1f}s"))
        if result.get("gen_time") is not None:
            meta_rows.append(("Generate", f"{result['gen_time']:.1f}s"))
        if result.get("seed") is not None:
            meta_rows.append(("Seed", str(result["seed"])))
        flags = result["flags"]
        meta_rows.append(("Encoder", "local" if flags["local_encoder"] else "remote API"))
        flag_bits = [k for k in ("turbo", "uncensored", "schnell", "klein", "kontext") if flags.get(k)]
        if flags.get("gguf_quant"):
            flag_bits.append(f"gguf:{flags['gguf_quant']}")
        meta_rows.append(("Extras", ", ".join(flag_bits) if flag_bits else "—"))
    else:
        state, badge = "fail", "FAIL"
        media = (
            '<div class="err"><div class="errlabel">ERROR</div>'
            f'<pre>{html.escape(result.get("error") or "unknown error")}</pre></div>'
        )
        meta_rows = [("Command", result["command"])]

    if kind == "edit":
        meta_rows.append(("Mode", "instruction edit (synthetic reference)"))

    meta_html = "".join(
        f'<tr><td class="k">{html.escape(k)}</td><td class="v">{html.escape(str(v))}</td></tr>'
        for k, v in meta_rows
    )
    return f"""
        <div class="card {state}">
          <div class="head">
            <span class="num">{num}</span>
            <span class="title">{html.escape(name)}</span>
            <span class="badge {state}">{badge}</span>
          </div>
          <div class="media">{media}</div>
          <table class="meta">{meta_html}</table>
        </div>"""


def write_html(by_num, planned, running=None, done=False):
    """(Re)write the tracking page. While the sweep runs (done=False) the page
    auto-refreshes every few seconds; the final write drops the refresh."""
    global _GPU_NAME
    if _GPU_NAME is None:
        _GPU_NAME = gpu_name()

    generated = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cards = [_card_for(entry, by_num.get(entry[0]), running) for entry in planned]

    results = [by_num[n] for n in sorted(by_num)]
    n_pass = sum(1 for r in results if r["status"] == "pass")
    n_fail = sum(1 for r in results if r["status"] != "pass")
    n_left = sum(1 for entry in planned if entry[0] not in by_num)
    if done:
        state_line = f"<b>{n_pass} passed</b>, {n_fail} failed — sweep complete"
    else:
        state_line = f"<b>{n_pass} passed</b>, {n_fail} failed, {n_left} to go — <b class='live'>RUNNING</b>"
    refresh = "" if done else '<meta http-equiv="refresh" content="3">'

    doc = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
{refresh}
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>FLUX server smoke test</title>
<style>
  :root {{ color-scheme: dark; }}
  body {{ margin:0; padding:24px; background:#0f1115; color:#e6e6e6;
         font:14px/1.5 -apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif; }}
  h1 {{ margin:0 0 4px; font-size:22px; }}
  .sub {{ color:#9aa4b2; margin-bottom:20px; }}
  .sub b {{ color:#e6e6e6; }}
  .sub b.live {{ color:#6ab7ff; }}
  .grid {{ display:grid; gap:18px;
           grid-template-columns:repeat(auto-fill,minmax(340px,1fr)); }}
  .card {{ background:#171a21; border:1px solid #262b36; border-radius:12px;
           overflow:hidden; display:flex; flex-direction:column; }}
  .card.fail {{ border-color:#5a2230; }}
  .card.running {{ border-color:#1f4468; }}
  .card.pending {{ opacity:.65; }}
  .head {{ display:flex; align-items:center; gap:10px; padding:12px 14px;
           border-bottom:1px solid #262b36; }}
  .num {{ width:24px; height:24px; flex:0 0 auto; border-radius:6px; background:#262b36;
          color:#cbd3df; font-weight:600; display:flex; align-items:center; justify-content:center; }}
  .title {{ font-weight:600; flex:1; }}
  .badge {{ font-size:11px; font-weight:700; letter-spacing:.5px; padding:3px 8px; border-radius:999px; }}
  .badge.pass {{ background:#10391f; color:#5fd98a; }}
  .badge.fail {{ background:#3d1620; color:#ff7a93; }}
  .badge.running {{ background:#12314d; color:#6ab7ff; }}
  .badge.pending {{ background:#262b36; color:#8b94a3; }}
  .media {{ background:#0b0d11; aspect-ratio:4/3; display:flex; align-items:center; justify-content:center; }}
  .media img {{ width:100%; height:100%; object-fit:cover; display:block; }}
  .err {{ padding:16px; width:100%; box-sizing:border-box; }}
  .errlabel {{ color:#ff7a93; font-weight:700; font-size:12px; margin-bottom:8px; }}
  .err pre {{ margin:0; white-space:pre-wrap; word-break:break-word; color:#ffb3c1; font-size:12px; }}
  .pend {{ color:#5c6676; font-size:13px; }}
  .run {{ text-align:center; }}
  .run .phase {{ color:#9fc9f2; margin-top:10px; }}
  .run .elapsed {{ color:#5c88b3; font-size:12px; margin-top:2px;
                   font-variant-numeric:tabular-nums; }}
  .spinner {{ width:28px; height:28px; margin:0 auto; border-radius:50%;
              border:3px solid #1f4468; border-top-color:#6ab7ff;
              animation:spin 1s linear infinite; }}
  @keyframes spin {{ to {{ transform:rotate(360deg); }} }}
  table.meta {{ width:100%; border-collapse:collapse; }}
  table.meta td {{ padding:6px 14px; border-top:1px solid #20242e; vertical-align:top; }}
  td.k {{ color:#8b94a3; width:90px; }}
  td.v {{ color:#dde3ec; word-break:break-word; font-variant-numeric:tabular-nums; }}
</style>
</head>
<body>
  <h1>FLUX server smoke test</h1>
  <div class="sub">
    Prompt: <b>&ldquo;{html.escape(PROMPT)}&rdquo;</b>
    (edits: &ldquo;{html.escape(EDIT_PROMPT)}&rdquo;) &nbsp;·&nbsp;
    {WIDTH}&times;{HEIGHT} &nbsp;·&nbsp;
    {state_line} &nbsp;·&nbsp;
    {html.escape(_GPU_NAME)} &nbsp;·&nbsp; {generated}
  </div>
  <div class="grid">{''.join(cards)}</div>
</body>
</html>"""

    path = os.path.join(OUT_DIR, "index.html")
    with open(path, "w") as f:
        f.write(doc)
    return path


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # Worker mode: run exactly one config in this (isolated) process and exit.
    if len(sys.argv) >= 3 and sys.argv[1] == "--worker":
        num = int(sys.argv[2])
        entry = next((m for m in MENU if m[0] == num), None)
        if entry is None:
            print(f"unknown option {num}")
            return 1
        run_config_inprocess(*entry)
        return 0

    selected = MENU
    if len(sys.argv) > 1:
        wanted = {int(a) for a in sys.argv[1:]}
        selected = [m for m in MENU if m[0] in wanted]
        if not selected:
            print(f"No matching options. Valid: {[m[0] for m in MENU]}")
            return 1

    # Merge into any prior results so single-option re-runs keep the full report.
    by_num = {}
    if os.path.exists(RESULTS_JSON):
        try:
            with open(RESULTS_JSON) as f:
                by_num = {r["num"]: r for r in json.load(f)}
        except Exception:
            by_num = {}
    # Re-selected options run fresh; drop their stale result from the tracker.
    for num, *_ in selected:
        by_num.pop(num, None)

    # The tracking page always shows the full menu, so a partial sweep still
    # renders prior/pending state for the other options.
    planned = MENU

    print(f"Smoke-testing {len(selected)} option(s). Live report -> {os.path.join(OUT_DIR, 'index.html')}")
    write_html(by_num, planned)
    for num, name, raw_args, kind in selected:
        by_num[num] = test_one(num, name, raw_args, kind, by_num, planned)
        results = [by_num[n] for n in sorted(by_num)]
        with open(RESULTS_JSON, "w") as f:
            json.dump(results, f, indent=2)
        write_html(by_num, planned)

    report = write_html(by_num, planned, done=True)
    results = [by_num[n] for n in sorted(by_num)]
    print("\n" + "=" * 64)
    print("SUMMARY")
    for r in results:
        line = f"  [{r['status'].upper():4}] {r['num']:>2}. {r['name']}"
        if r["status"] == "pass":
            line += f"  (load {r['load_time']:.0f}s, gen {r['gen_time']:.0f}s)"
        else:
            line += f"  -> {r['error']}"
        print(line)
    n_pass = sum(1 for r in results if r["status"] == "pass")
    print(f"\n{n_pass}/{len(results)} passed")
    print(f"Report: {report}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
