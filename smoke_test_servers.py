#!/usr/bin/env python3
"""Smoke-test every model option in run_server.sh.

Loads each menu configuration in-process (same arg derivation as web_server.py),
generates one sample image with a fixed prompt, then writes an HTML report
showing pass/fail, timings, and the resulting image for each option.

Usage:
    python smoke_test_servers.py                # test all menu options
    python smoke_test_servers.py 1 3 10         # test only options 1, 3, 10
"""

import sys
import os
import json
import time
import html
import datetime
import traceback
import subprocess

PROMPT = "a frog on a log in bog"
WIDTH, HEIGHT = 1024, 768          # landscape smoke-test size
DEFAULT_STEPS = 20                 # schnell/turbo override this internally
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "server_smoke_test")
RESULTS_JSON = os.path.join(OUT_DIR, "results.json")

# The menu options exactly as defined in run_server.sh start_server().
# raw_args are the flags run_server.sh passes to web_server.py.
MENU = [
    (1,  "FLUX.1 4-bit BNB",       []),
    (2,  "FLUX.1 Full",            ["--full-model"]),
    (3,  "FLUX.1 GGUF Q8",         ["--gguf", "q8", "--local-encoder"]),
    (4,  "FLUX.1-schnell",         ["--schnell", "--local-encoder"]),
    (5,  "FLUX.2 4-bit BNB",       ["--flux2"]),
    (6,  "FLUX.2 Full",            ["--flux2", "--full-model"]),
    (7,  "FLUX.2 Full + Turbo",    ["--flux2", "--full-model", "--turbo"]),
    (8,  "FLUX.1 + Uncensored",    ["--uncensored"]),
    (9,  "FLUX.2 Full (no Turbo)", ["--flux2", "--full-model", "--no-turbo"]),
    (10, "FLUX.2-klein-9B",        ["--klein"]),
]


def derive_flags(raw_args):
    """Replicate web_server.py's argument derivation for a set of raw flags."""
    full_model = "--full-model" in raw_args
    flux2 = "--flux2" in raw_args
    schnell = "--schnell" in raw_args
    uncensored = "--uncensored" in raw_args
    klein = "--klein" in raw_args
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

    local_encoder = local_encoder_flag or full_model or schnell or uncensored
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
        "local_encoder": local_encoder,
        "turbo": turbo,
    }


def _worker_result_path(num):
    return os.path.join(OUT_DIR, f"_result_{num:02d}.json")


def run_config_inprocess(num, name, raw_args):
    """Load one config and generate the sample image (runs inside a worker
    subprocess so model/GPU state is fully isolated per config — this matches
    how the real server runs: one process per model)."""
    import torch
    from fl24bit import load_model, generate_image, load_turbo_lora, load_uncensored_lora

    flags = derive_flags(raw_args)
    cmd = "python web_server.py " + " ".join(raw_args) if raw_args else "python web_server.py"
    result = {
        "num": num, "name": name, "command": cmd, "flags": flags,
        "status": "fail", "error": None, "image": None,
        "load_time": None, "gen_time": None, "seed": None, "steps": None,
    }

    print(f"\n{'='*64}\n[{num}] {name}\n  {cmd}\n  flags: {flags}\n{'='*64}")
    try:
        t0 = time.perf_counter()
        load_model(
            local_encoder=flags["local_encoder"], full_model=flags["full_model"],
            gguf_quant=flags["gguf_quant"], flux2=flags["flux2"],
            schnell=flags["schnell"], for_lora=flags["uncensored"], klein=flags["klein"],
        )
        if flags["turbo"]:
            load_turbo_lora()
        if flags["uncensored"]:
            load_uncensored_lora()
        result["load_time"] = time.perf_counter() - t0
        print(f"  loaded in {result['load_time']:.1f}s")

        t0 = time.perf_counter()
        image, seed, _ = generate_image(
            PROMPT, steps=DEFAULT_STEPS, width=WIDTH, height=HEIGHT,
            local_encoder=flags["local_encoder"],
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


def test_one(num, name, raw_args):
    """Parent side: run one config in an isolated subprocess and return its
    result. A crash (OOM kill, segfault) is captured as a failure rather than
    taking down the whole sweep."""
    flags = derive_flags(raw_args)
    cmd = "python web_server.py " + " ".join(raw_args) if raw_args else "python web_server.py"
    rpath = _worker_result_path(num)
    if os.path.exists(rpath):
        os.remove(rpath)
    logpath = os.path.join(OUT_DIR, f"_worker_{num:02d}.log")

    print(f"\n>>> [{num}] {name} — launching isolated worker...")
    with open(logpath, "w") as logf:
        proc = subprocess.run(
            [sys.executable, os.path.abspath(__file__), "--worker", str(num)],
            stdout=logf, stderr=subprocess.STDOUT,
        )

    if os.path.exists(rpath):
        with open(rpath) as f:
            return json.load(f)

    # Worker died before writing a result (OOM/segfault). Salvage the tail.
    tail = ""
    try:
        with open(logpath) as f:
            tail = "".join(f.readlines()[-6:]).strip()
    except Exception:
        pass
    sig = f"worker exited with code {proc.returncode}"
    if proc.returncode == -9 or proc.returncode == 137:
        sig = "worker killed (OOM, exit 137)"
    return {
        "num": num, "name": name, "command": cmd, "flags": flags,
        "status": "fail", "error": f"{sig}. Last output:\n{tail}",
        "image": None, "load_time": None, "gen_time": None, "seed": None, "steps": None,
    }


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


def write_html(results):
    generated = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    gpu = gpu_name()
    n_pass = sum(1 for r in results if r["status"] == "pass")

    cards = []
    for r in results:
        ok = r["status"] == "pass"
        badge = "PASS" if ok else "FAIL"
        badge_cls = "pass" if ok else "fail"

        if ok and r["image"]:
            media = f'<img src="{html.escape(r["image"])}" alt="{html.escape(r["name"])}">'
        else:
            media = (
                '<div class="err"><div class="errlabel">ERROR</div>'
                f'<pre>{html.escape(r["error"] or "unknown error")}</pre></div>'
            )

        meta_rows = [("Command", r["command"])]
        if r["load_time"] is not None:
            meta_rows.append(("Load time", f"{r['load_time']:.1f}s"))
        if r["gen_time"] is not None:
            meta_rows.append(("Generate", f"{r['gen_time']:.1f}s"))
        if r["seed"] is not None:
            meta_rows.append(("Seed", str(r["seed"])))
        meta_rows.append(("Encoder", "local" if r["flags"]["local_encoder"] else "remote API"))
        flag_bits = [k for k in ("turbo", "uncensored", "schnell", "klein") if r["flags"][k]]
        if r["flags"]["gguf_quant"]:
            flag_bits.append(f"gguf:{r['flags']['gguf_quant']}")
        meta_rows.append(("Extras", ", ".join(flag_bits) if flag_bits else "—"))

        meta_html = "".join(
            f'<tr><td class="k">{html.escape(k)}</td><td class="v">{html.escape(str(v))}</td></tr>'
            for k, v in meta_rows
        )

        cards.append(f"""
        <div class="card {badge_cls}">
          <div class="head">
            <span class="num">{r['num']}</span>
            <span class="title">{html.escape(r['name'])}</span>
            <span class="badge {badge_cls}">{badge}</span>
          </div>
          <div class="media">{media}</div>
          <table class="meta">{meta_html}</table>
        </div>""")

    doc = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>FLUX server smoke test</title>
<style>
  :root {{ color-scheme: dark; }}
  body {{ margin:0; padding:24px; background:#0f1115; color:#e6e6e6;
         font:14px/1.5 -apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif; }}
  h1 {{ margin:0 0 4px; font-size:22px; }}
  .sub {{ color:#9aa4b2; margin-bottom:20px; }}
  .sub b {{ color:#e6e6e6; }}
  .grid {{ display:grid; gap:18px;
           grid-template-columns:repeat(auto-fill,minmax(340px,1fr)); }}
  .card {{ background:#171a21; border:1px solid #262b36; border-radius:12px;
           overflow:hidden; display:flex; flex-direction:column; }}
  .card.fail {{ border-color:#5a2230; }}
  .head {{ display:flex; align-items:center; gap:10px; padding:12px 14px;
           border-bottom:1px solid #262b36; }}
  .num {{ width:24px; height:24px; flex:0 0 auto; border-radius:6px; background:#262b36;
          color:#cbd3df; font-weight:600; display:flex; align-items:center; justify-content:center; }}
  .title {{ font-weight:600; flex:1; }}
  .badge {{ font-size:11px; font-weight:700; letter-spacing:.5px; padding:3px 8px; border-radius:999px; }}
  .badge.pass {{ background:#10391f; color:#5fd98a; }}
  .badge.fail {{ background:#3d1620; color:#ff7a93; }}
  .media {{ background:#0b0d11; aspect-ratio:4/3; display:flex; align-items:center; justify-content:center; }}
  .media img {{ width:100%; height:100%; object-fit:cover; display:block; }}
  .err {{ padding:16px; width:100%; box-sizing:border-box; }}
  .errlabel {{ color:#ff7a93; font-weight:700; font-size:12px; margin-bottom:8px; }}
  .err pre {{ margin:0; white-space:pre-wrap; word-break:break-word; color:#ffb3c1; font-size:12px; }}
  table.meta {{ width:100%; border-collapse:collapse; }}
  table.meta td {{ padding:6px 14px; border-top:1px solid #20242e; vertical-align:top; }}
  td.k {{ color:#8b94a3; width:90px; }}
  td.v {{ color:#dde3ec; word-break:break-word; font-variant-numeric:tabular-nums; }}
</style>
</head>
<body>
  <h1>FLUX server smoke test</h1>
  <div class="sub">
    Prompt: <b>&ldquo;{html.escape(PROMPT)}&rdquo;</b> &nbsp;·&nbsp;
    {WIDTH}&times;{HEIGHT} &nbsp;·&nbsp;
    <b>{n_pass}/{len(results)}</b> options passed &nbsp;·&nbsp;
    {html.escape(gpu)} &nbsp;·&nbsp; {generated}
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

    print(f"Smoke-testing {len(selected)} option(s). Output -> {OUT_DIR}")
    for num, name, raw_args in selected:
        by_num[num] = test_one(num, name, raw_args)
        # Persist + rewrite the full report after each config so partial
        # progress is viewable and re-runs merge into prior results.
        results = [by_num[n] for n in sorted(by_num)]
        with open(RESULTS_JSON, "w") as f:
            json.dump(results, f, indent=2)
        report = write_html(results)

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
