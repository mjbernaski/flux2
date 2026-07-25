#!/usr/bin/env python3
"""Iterative image-edit loop against the FLUX web server.

Takes an input image and an edit direction, runs the edit through the server
(Kontext or FLUX.2), inspects the output, revises the instruction, and tries
again — up to N iterations, stoppable between any two.

The "look at the output" step is pluggable:
  - a local ollama vision model (--vlm; default from the CRITIQUE_MODEL env
    var, falling back to qwen3.6:latest) compares the reference and the
    output, grades the result 0-10, and proposes a revised instruction,
    entirely on-box (no cloud), so any content stays local. It sees the
    session's prompt trajectory (what was tried and why it failed) and runs
    with thinking enabled + structured JSON output;
  - with --vlm none (or if ollama is unreachable) the loop falls back to
    pixel metrics plus your own typed feedback.

Each iteration re-edits the ORIGINAL image with a refined instruction, so
failed attempts don't compound. Pass --chain to instead feed each output in
as the next iteration's input (for progressive multi-step edits).

Usage:
  python edit_loop.py photo.png "make her hair brown" -n 5
  python edit_loop.py photo.png "remove the background clutter" --auto
  python edit_loop.py photo.png "add sunglasses" --chain --vlm none
"""

import argparse
import base64
import io
import json
import os
import shutil
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime

from PIL import Image

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SESSIONS_DIR = os.path.join(SCRIPT_DIR, "edit-loop-sessions")

KONTEXT_PROMPT_TIPS = (
    "Kontext instruction tips: use a direct imperative ('Change X to Y', "
    "'Remove X'), name the subject concretely, describe the desired result "
    "explicitly rather than what to undo, and end with 'keep everything else "
    "exactly the same' to preserve the rest of the scene."
)

SDXL_PROMPT_TIPS = (
    "SDXL prompt tips: this model has NO instruction understanding — the "
    "prompt must be a DESCRIPTION of the desired final image (subject, "
    "setting, clothing/appearance, lighting, 'photorealistic'), never a "
    "command like 'remove X' or 'change Y'. Naming an object in the prompt "
    "pulls it INTO the image, so describe what should be there instead of "
    "what to take away."
)

PROMPT_TIPS = {"instruction": KONTEXT_PROMPT_TIPS, "description": SDXL_PROMPT_TIPS}


def load_api_key():
    key = os.environ.get("FLUX_API_KEY")
    if key:
        return key
    env_path = os.path.join(SCRIPT_DIR, ".env")
    if os.path.exists(env_path):
        for line in open(env_path):
            if line.startswith("FLUX_API_KEY="):
                return line.split("=", 1)[1].strip()
    return None


def api(server, path, key, payload=None, timeout=30):
    req = urllib.request.Request(
        server.rstrip("/") + path,
        data=json.dumps(payload).encode() if payload is not None else None,
        headers={"Content-Type": "application/json", "X-API-Key": key},
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def fetch_image(server, filename, key):
    req = urllib.request.Request(
        server.rstrip("/") + "/images/" + filename,
        headers={"X-API-Key": key},
    )
    with urllib.request.urlopen(req, timeout=60) as r:
        return Image.open(io.BytesIO(r.read())).convert("RGB")


def submit_edit(server, key, prompt, image, steps, guidance, seed):
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    payload = {
        "prompt": prompt,
        "input_images": [base64.b64encode(buf.getvalue()).decode()],
        "steps": steps,
        "batch": 1,
    }
    if guidance is not None:
        payload["guidance"] = guidance
    if seed is not None:
        payload["seed"] = seed
    resp = api(server, "/generate", key, payload)
    if not resp.get("success"):
        raise RuntimeError(f"generate failed: {resp.get('error')}")
    return resp["job_id"]


def wait_for_job(server, key, job_id, poll=5):
    """Poll /status until the job finishes; returns the job record."""
    spinner = "|/-\\"
    tick = 0
    while True:
        st = api(server, "/status", key)
        for j in st.get("recent_done") or []:
            if j["id"] == job_id:
                print()  # end the progress line
                if j.get("state") != "done" or j.get("error"):
                    raise RuntimeError(f"job {job_id} {j.get('state')}: {j.get('error')}")
                return j
        running = st.get("running")
        if running and running.get("id") == job_id:
            step, total = running.get("step", 0), running.get("total_steps", 0)
            msg = f"generating {step}/{total}"
        else:
            pos = next((i + 1 for i, q in enumerate(st.get("queued") or [])
                        if q.get("id") == job_id), None)
            msg = f"queued (position {pos})" if pos else "waiting"
        print(f"\r  {spinner[tick % 4]} {msg}   ", end="", flush=True)
        tick += 1
        time.sleep(poll)


def edit_metrics(reference, output):
    """Cheap structural metrics: how anchored is the output, how much changed."""
    import numpy as np

    small = (64, 64)
    a = np.asarray(reference.resize(small).convert("L"), float).ravel()
    b = np.asarray(output.resize(small).convert("L"), float).ravel()
    corr = float(np.corrcoef(a, b)[0, 1])

    out_r = output.resize(reference.size)
    diff = abs(
        np.asarray(reference, float) - np.asarray(out_r, float)
    ).mean(axis=2)
    changed_pct = float((diff > 20).mean() * 100)
    return {"anchoring": round(corr, 3), "changed_pct": round(changed_pct, 1)}


def describe_metrics(m):
    notes = []
    if m["changed_pct"] < 1.0:
        notes.append("the output is nearly identical to the input — the edit "
                      "was likely NOT applied")
    elif m["anchoring"] < 0.5:
        notes.append("the output shares little structure with the input — it "
                      "looks regenerated rather than edited")
    else:
        notes.append("the output is anchored to the input with a localized change")
    return f"{m['changed_pct']}% of pixels changed, structural anchoring " \
           f"{m['anchoring']} (1.0 = identical layout). " + "; ".join(notes)


def img_b64(image, max_side=896):
    im = image.copy()
    im.thumbnail((max_side, max_side))
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


# Structured-output schema for the critique: ollama constrains generation to
# this shape, so the reply is guaranteed-parseable JSON (no regex scraping).
CRITIQUE_SCHEMA = {
    "type": "object",
    "properties": {
        "applied": {"type": "boolean"},
        "score": {"type": "integer", "minimum": 0, "maximum": 10},
        "critique": {"type": "string"},
        "revised_prompt": {"type": "string"},
    },
    "required": ["applied", "score", "critique", "revised_prompt"],
}


def _ollama_chat(ollama_url, payload):
    req = urllib.request.Request(
        ollama_url.rstrip("/") + "/api/chat",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=300) as r:
        return json.loads(r.read().decode())


def _chat_text(ollama_url, payload):
    """Chat with retries for two ollama quirks: models that reject the
    "think" flag (HTTP error → retry without it), and 200 replies with empty
    content — a request racing a model load/unload returns done_reason
    "load" (retried with a short wait until the load settles, which can take
    a while right after server startup while the warm-up pull runs), and a
    thinking model occasionally ends its turn after the thinking phase
    without emitting the schema-constrained reply (retry, dropping "think"
    in that case to force the JSON out directly)."""
    try:
        resp = _ollama_chat(ollama_url, payload)
    except urllib.error.HTTPError:
        payload.pop("think", None)
        resp = _ollama_chat(ollama_url, payload)
    text = resp["message"]["content"]
    for _ in range(5):
        if text.strip():
            break
        if resp.get("done_reason") == "load":
            time.sleep(3)
        else:
            payload.pop("think", None)
        resp = _ollama_chat(ollama_url, payload)
        text = resp["message"]["content"]
    return text


def vlm_critique(model, direction, prompt, reference, output, metrics,
                 ollama_url="http://127.0.0.1:11434", style="instruction",
                 history=None):
    """Ask a local ollama vision model whether the edit landed, grade it
    0-10, and propose the next prompt. `style` picks the prompting idiom of
    the backend: 'instruction' (Kontext edit commands) or 'description' (SDXL
    scene descriptions). `history` is the session's prompt trajectory — a
    list of {prompt, applied, score, critique} from earlier iterations — so
    the critic revises against what already failed instead of oscillating.
    Runs with thinking enabled (deliberation before the verdict) and a JSON
    schema constraining the reply. Returns dict or None on any failure."""
    if style == "description":
        backend_line = ("You are refining a prompt for SDXL img2img: the FIRST "
                        "image is the starting image; the SECOND is the result "
                        "of re-rendering it with the prompt.")
        revised_hint = ("an improved DESCRIPTIVE prompt of the desired final "
                        "image to try next; one sentence, no commands")
    else:
        backend_line = ("You are refining an edit instruction for an "
                        "instruction-based image editor (FLUX.1-Kontext). The "
                        "FIRST image is the reference; the SECOND is the "
                        "editor's output.")
        revised_hint = ("an improved instruction to try next; keep it one "
                        "imperative sentence")
    history_block = ""
    if history:
        lines = [
            f"  {n}. \"{h.get('prompt', '')}\" -> applied={h.get('applied')}, "
            f"score={h.get('score', '?')}/10 — {h.get('critique', '')}"
            for n, h in enumerate(history, start=1)
        ]
        history_block = (
            "Previous attempts this session (oldest first):\n"
            + "\n".join(lines) + "\n"
            "Do not re-propose a phrasing that already failed. If the goal "
            "keeps not landing, change the approach: a different verb, a more "
            "concrete name for the subject, or an explicit description of the "
            "desired result.\n"
        )
    ask = (
        f"{backend_line}\n"
        f"The user's goal: {direction}\n"
        f"The prompt used for this attempt: {prompt}\n"
        f"{history_block}"
        f"Pixel metrics: {describe_metrics(metrics)}\n"
        f"{PROMPT_TIPS.get(style, KONTEXT_PROMPT_TIPS)}\n"
        "Compare the images. Decide whether the requested edit was applied, "
        "grade the output 0-10 against the goal (10 = goal fully achieved "
        "with everything else preserved, 0 = no progress), write a one-"
        f"sentence critique, and give revised_prompt: {revised_hint}."
    )
    payload = {
        "model": model,
        "messages": [{
            "role": "user",
            "content": ask,
            "images": [img_b64(reference), img_b64(output)],
        }],
        "stream": False,
        "think": True,
        "format": CRITIQUE_SCHEMA,
        # Keep the critic resident between iterations so each critique doesn't
        # pay the model-load cost again.
        "keep_alive": "15m",
        "options": {"temperature": 0.3, "num_ctx": 8192},
    }
    try:
        text = _chat_text(ollama_url, payload)
        result = json.loads(text)
        if result.get("revised_prompt"):
            return result
        print(f"  (VLM reply missing revised_prompt: {text[:200]})")
    except Exception as e:
        print(f"  (VLM critique unavailable: {e})")
    return None


# Structured-output schema for the reverse path (photo → prompt).
DESCRIBE_SCHEMA = {
    "type": "object",
    "properties": {"prompt": {"type": "string"}},
    "required": ["prompt"],
}


def _prompt_from_reply(text):
    """Extract the prompt from a DESCRIBE_SCHEMA-formatted reply. With think
    disabled, qwen3.6 ignores the format schema and returns plain text —
    accept both shapes."""
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return (parsed.get("prompt") or "").strip()
        return str(parsed).strip()
    except json.JSONDecodeError:
        return text.strip()


def vlm_describe(model, image, think=False, ollama_url="http://127.0.0.1:11434"):
    """The reverse path: ask the local vision model to write a detailed
    text-to-image prompt that would recreate `image` from scratch. `image`
    may also be a list of images — then the prompt is a composite describing
    one coherent scene that merges the images' subjects and elements.
    `think=True` (default off) enables the deliberation phase — deeper but
    much slower. Returns the prompt string, or None on any failure."""
    images = image if isinstance(image, (list, tuple)) else [image]
    if len(images) > 1:
        ask = (
            f"You are given {len(images)} photographs. Write a detailed "
            "text-to-image generation prompt for a SINGLE new image that "
            "combines them into one coherent scene: merge their subjects, "
            "settings and distinctive elements so each photograph is "
            "recognizably represented. Describe the combined subjects and "
            "their exact appearance, pose and spatial arrangement; the "
            "unified setting and background; the composition and camera "
            "framing (angle, distance, lens feel); the lighting and time of "
            "day; the color palette and mood; and the overall style "
            "(photorealistic photo, film stock, illustration, etc.). Be "
            "specific and concrete — name colors, materials, textures and "
            "spatial relationships. Do not mention the source photographs or "
            "that you are describing images; just write the prompt as one "
            "dense paragraph describing the single combined scene."
        )
    else:
        ask = (
            "Write a detailed text-to-image generation prompt that would recreate "
            "this photograph from scratch. Describe the subject and their exact "
            "appearance, pose and expression; the setting and background; the "
            "composition and camera framing (angle, distance, lens feel); the "
            "lighting and time of day; the color palette and mood; and the overall "
            "style (photorealistic photo, film stock, illustration, etc.). Be "
            "specific and concrete — name colors, materials, textures and spatial "
            "relationships. Do not mention that you are describing an image; just "
            "write the prompt as one dense paragraph."
        )
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": ask,
                      "images": [img_b64(im) for im in images]}],
        "stream": False,
        "think": bool(think),
        "format": DESCRIBE_SCHEMA,
        "keep_alive": "15m",
        "options": {"temperature": 0.4, "num_ctx": 8192},
    }
    try:
        text = _chat_text(ollama_url, payload)
        prompt = _prompt_from_reply(text)
        if prompt:
            return prompt
        print(f"  (VLM describe reply missing prompt: {text[:200]})")
    except Exception as e:
        print(f"  (VLM describe unavailable: {e})")
    return None


# Per-backend prompting idioms for /boost — what a strong prompt looks like
# for the loaded model, so the rewrite targets the right style.
BOOST_GUIDANCE = {
    "flux2": (
        "The target model is FLUX.2, whose text encoder is a full LLM: it "
        "rewards detailed natural-language description. Write one rich "
        "paragraph of flowing prose covering the subject and its exact "
        "appearance, the setting, composition and camera framing, lighting, "
        "color palette, mood, and overall style. No keyword lists, no "
        "negative-prompt phrasing ('no X', 'without Y') — describe what "
        "SHOULD be in the image."
    ),
    "flux1": (
        "The target model is FLUX.1 (T5 text encoder): it rewards one dense "
        "descriptive paragraph of natural language — subject, setting, "
        "composition, lighting, palette, mood, style — with concrete nouns "
        "and specifics. Keep it under roughly 120 words; no keyword soup, "
        "no negative-prompt phrasing ('no X', 'without Y')."
    ),
    "kontext": (
        "The target model is FLUX.1-Kontext, an instruction-based image "
        "EDITOR: the prompt must stay a command, not a scene description. "
        + KONTEXT_PROMPT_TIPS
    ),
    "sdxl": (
        SDXL_PROMPT_TIPS + " Its CLIP encoder only reads roughly the first "
        "75 tokens, so keep it to short comma-separated descriptive phrases "
        "with the subject first and style/quality terms (e.g. "
        "'photorealistic, sharp focus, detailed') at the end."
    ),
}

# When the user has attached reference image(s), the nature of the prompt
# changes: FLUX.2 and Kontext condition on the reference and want the prompt
# as active CHANGES (edit instructions); SDXL and FLUX.1 img2img re-render
# from a description of the desired FINAL image, never the delta.
BOOST_GUIDANCE["flux2-edit"] = (
    "The target model is FLUX.2 conditioning on the user's attached "
    "reference image(s): the prompt must state the CHANGES to make as "
    "active edit instructions — imperative sentences naming concretely "
    "what to change, add or remove and the desired result, ending with "
    "what must stay the same. Do NOT describe the whole scene from "
    "scratch; the reference supplies it."
)
BOOST_GUIDANCE["sdxl-img2img"] = (
    BOOST_GUIDANCE["sdxl"] + " The user has attached a starting image "
    "(img2img): the prompt must still describe the desired FINAL image as "
    "a whole — subject, setting, lighting, style — never the delta from "
    "the starting image; anything left undescribed may drift when the "
    "image is re-rendered."
)
BOOST_GUIDANCE["flux1-img2img"] = (
    BOOST_GUIDANCE["flux1"] + " The user has attached a starting image "
    "(img2img): describe the desired FINAL image as a whole, never the "
    "changes relative to the starting image — the model re-renders from "
    "the description."
)


# How far /boost is allowed to depart from the user's draft, by level 1-5.
# Each level pairs a rewrite instruction with a sampling temperature —
# conservative levels run cold so the draft survives nearly verbatim,
# adventurous levels run hot for more inventive detail.
BOOST_LEVELS = {
    1: ("Make the lightest possible touch: fix grammar, word order and "
        "ambiguity only. Add NO new details — the result should read as the "
        "user's own prompt, just cleaner, and stay close to its original "
        "length.", 0.1),
    2: ("Tighten and clarify the wording, and add at most one or two small "
        "concrete details where the draft is vague. Keep it brief and "
        "recognizably the user's prompt.", 0.2),
    3: ("Flesh out what the draft leaves open — composition, lighting, "
        "style — into a well-formed prompt, keeping the user's framing and "
        "adding nothing that changes the scene.", 0.4),
    4: ("Expand generously: develop the scene with concrete specifics of "
        "subject appearance, setting, composition and camera, lighting, "
        "palette and mood. The result should be much richer than the draft, "
        "but every addition must fit naturally with what the user "
        "described.", 0.6),
    5: ("Reimagine boldly: treat the draft as a seed and write the most "
        "vivid, evocative prompt you can — take real creative liberties "
        "with style, atmosphere, lighting and composition, as long as the "
        "core subject and the user's explicit details remain intact.", 0.85),
}


def vlm_boost(model, prompt, family="flux2", model_desc="", level=3,
              think=False, ollama_url="http://127.0.0.1:11434", variant=None,
              negative_prompt=None):
    """Rewrite the user's draft prompt into a stronger one tuned to the
    prompting idiom of the loaded image model (`family` picks the guidance;
    `model_desc` is the human-readable model name for context; `level` 1-5
    sets how far the rewrite may depart from the draft; `think=True`,
    default off, enables the model's thinking phase — deeper but slower).
    `variant=(i, n)` marks this call as one of n independent rewrites of the
    same draft: the model is told to commit to a direction the other rewrites
    are unlikely to take, and sampling runs hot enough to actually diverge.
    `negative_prompt`, when set (SDXL), is what the user is already excluding
    via CFG — the rewrite must not add positive-prompt detail that fights or
    duplicates it (e.g. don't add "vibrant colors" when the negative prompt
    says "oversaturated"). Text-only chat — no images. Returns the improved
    prompt string, or None on any failure."""
    guidance = BOOST_GUIDANCE.get(family, BOOST_GUIDANCE["flux2"])
    degree, temperature = BOOST_LEVELS.get(level, BOOST_LEVELS[3])
    variant_line = ""
    if variant:
        v_idx, v_cnt = variant
        variant_line = (
            f"This is variation {v_idx} of {v_cnt} independent rewrites of "
            "this same draft, each generated without seeing the others. "
            "Commit to one distinctive interpretation — an angle of style, "
            "mood, composition, lighting or setting that the obvious rewrite "
            f"would not take — so the {v_cnt} variations diverge clearly "
            "while every one preserves the user's stated subject and "
            "details.\n"
        )
        # Each variation is a separate call; sampling heat is the only thing
        # that separates them, and the level-1/2 temperatures are too cold.
        temperature = max(temperature, 0.7)
    negative_line = ""
    if negative_prompt:
        negative_line = (
            f"The user is separately excluding this via a negative prompt: "
            f"\"{negative_prompt}\". Do not add positive-prompt detail that "
            "fights or duplicates it — nothing that describes, however "
            "positively framed, what the negative prompt is already ruling "
            "out.\n"
        )
    ask = (
        "You improve prompts for a local text-to-image system"
        + (f" currently running {model_desc}" if model_desc else "") + ".\n"
        f"{guidance}\n"
        f"The user's draft prompt: {prompt}\n"
        f"{negative_line}"
        f"{variant_line}"
        "Rewrite it into a stronger prompt for this model. Preserve the "
        "user's intent and every explicit detail they gave (subjects, "
        "counts, colors, names, any text to render verbatim). "
        f"Degree of rewrite: {degree} "
        "Where the degree conflicts with the style guidance above (e.g. on "
        "length or how much to add), the degree wins. "
        "Reply with the improved prompt only."
    )
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": ask}],
        "stream": False,
        "think": bool(think),
        "format": DESCRIBE_SCHEMA,
        "keep_alive": "15m",
        "options": {"temperature": temperature, "num_ctx": 8192},
    }
    try:
        text = _chat_text(ollama_url, payload)
        boosted = _prompt_from_reply(text)
        if boosted:
            return boosted
        print(f"  (VLM boost reply missing prompt: {text[:200]})")
    except Exception as e:
        print(f"  (VLM boost unavailable: {e})")
    return None


def _label_font(size):
    from PIL import ImageFont
    for path in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    ):
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size)
            except Exception:
                pass
    return ImageFont.load_default()


def build_film_strip(frames, frame_h=512, gap=8):
    """Compose labeled frames into one horizontal film strip.

    frames: list of (label, PIL.Image) in sequence order — typically
    ('input', original) followed by ('1', ...), ('2', ...). Frames are scaled
    to a common height, preserving each aspect ratio.
    """
    from PIL import ImageDraw

    scaled = [(label, im.resize((max(1, round(im.width * frame_h / im.height)), frame_h)))
              for label, im in frames]
    total_w = sum(im.width for _, im in scaled) + gap * (len(scaled) + 1)
    strip = Image.new("RGB", (total_w, frame_h + 2 * gap), (16, 16, 26))
    draw = ImageDraw.Draw(strip)
    font = _label_font(24)
    x = gap
    for label, im in scaled:
        strip.paste(im, (x, gap))
        bbox = draw.textbbox((0, 0), label, font=font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        pad = 5
        draw.rectangle([x + 6 - pad, gap + 6 - pad, x + 6 + tw + pad, gap + 6 + th + pad],
                       fill=(0, 0, 0))
        draw.text((x + 6 - bbox[0], gap + 6 - bbox[1]), label,
                  fill=(255, 255, 255), font=font)
        x += im.width + gap
    return strip


def heuristic_revision(direction, prompt, metrics):
    """No-VLM fallback: nudge the instruction based on pixel metrics."""
    revised = prompt
    if metrics["changed_pct"] < 1.0:
        base = direction.rstrip(". ")
        revised = (f"{base}. Make this change clearly and strongly visible in "
                   "the result, keep everything else exactly the same")
    elif metrics["anchoring"] < 0.5 and "keep everything else" not in prompt.lower():
        revised = prompt.rstrip(". ") + ", keep everything else exactly the same"
    return revised


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("image", help="input image to edit")
    ap.add_argument("direction", help="what you want changed, in plain words")
    ap.add_argument("-n", "--iterations", type=int, default=5,
                    help="max iterations (default 5)")
    ap.add_argument("--server", default="http://127.0.0.1:2222")
    ap.add_argument("--vlm", default=os.environ.get("CRITIQUE_MODEL", "qwen3.6:latest"),
                    help="ollama vision model for critique, or 'none' "
                         "(default: CRITIQUE_MODEL env or qwen3.6:latest)")
    ap.add_argument("--ollama", default="http://127.0.0.1:11434",
                    help="ollama endpoint")
    ap.add_argument("--auto", action="store_true",
                    help="don't pause for confirmation between iterations")
    ap.add_argument("--stop-score", type=int, default=8,
                    help="in --auto mode, stop early once the critic grades an "
                         "iteration at or above this score (default 8)")
    ap.add_argument("--chain", action="store_true",
                    help="feed each output in as the next iteration's input "
                         "(default: always re-edit the original)")
    ap.add_argument("--steps", type=int, default=25)
    ap.add_argument("--guidance", type=float, default=None,
                    help="guidance scale (default: server default, 2.5 for Kontext)")
    ap.add_argument("--seed", type=int, default=None,
                    help="fixed seed (default: vary per iteration)")
    args = ap.parse_args()

    key = load_api_key()
    if not key:
        sys.exit("FLUX_API_KEY not set (env or .env)")

    prompt_style = "instruction"
    try:
        info = api(args.server, "/model-info", key)
        print(f"Server model: {info.get('description', 'unknown')}")
        if info.get("sd"):
            prompt_style = "description"
            print("  SDXL backend: prompts are treated as scene descriptions, "
                  "not edit instructions.")
        elif not info.get("kontext") and info.get("flux_version") != 2:
            print("  WARNING: loaded model is not an instruction editor "
                  "(Kontext) or FLUX.2 — plain FLUX.1 does strength-based "
                  "img2img, so results will drift from the input.")
    except Exception as e:
        sys.exit(f"Cannot reach server at {args.server}: {e}")

    original = Image.open(args.image).convert("RGB")
    session = os.path.join(SESSIONS_DIR, datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.makedirs(session, exist_ok=True)
    original.save(os.path.join(session, "iter_00_input.png"))

    log = {"direction": args.direction, "image": os.path.abspath(args.image),
           "chain": args.chain, "iterations": []}
    reference = original
    versions = [original]  # versions[0] = input, versions[i] = iteration i output
    prompt = args.direction
    use_vlm = args.vlm.lower() != "none"
    history = []  # prompt trajectory handed to the critic each iteration

    for i in range(1, args.iterations + 1):
        print(f"\n=== Iteration {i}/{args.iterations} ===")
        print(f"  instruction: {prompt}")
        job_id = submit_edit(args.server, key, prompt, reference,
                             args.steps, args.guidance, args.seed)
        job = wait_for_job(args.server, key, job_id)
        filename = job["images"][0]["filename"]
        output = fetch_image(args.server, filename, key)

        out_path = os.path.join(session, f"iter_{i:02d}.png")
        output.save(out_path)
        metrics = edit_metrics(reference, output)
        print(f"  output: {out_path}")
        print(f"  {describe_metrics(metrics)}")

        critique = None
        if use_vlm:
            print(f"  asking {args.vlm} to compare input and output...")
            critique = vlm_critique(args.vlm, args.direction, prompt,
                                    reference, output, metrics, args.ollama,
                                    style=prompt_style, history=history)
        if critique:
            score = critique.get("score")
            score_txt = f" score={score}/10" if isinstance(score, int) else ""
            print(f"  VLM: edit applied={critique.get('applied')}{score_txt} — "
                  f"{critique.get('critique', '')}")
            next_prompt = critique["revised_prompt"]
        else:
            next_prompt = heuristic_revision(args.direction, prompt, metrics)
        if next_prompt != prompt:
            print(f"  proposed next instruction: {next_prompt}")
        history.append({
            "prompt": prompt,
            "applied": critique.get("applied") if critique else None,
            "score": critique.get("score") if critique else None,
            "critique": (critique.get("critique") if critique
                         else describe_metrics(metrics)),
        })

        log["iterations"].append({
            "prompt": prompt, "filename": filename, "output": out_path,
            "metrics": metrics, "critique": critique,
        })
        with open(os.path.join(session, "session.json"), "w") as f:
            json.dump(log, f, indent=2)

        # Early stop: the critic says the goal landed and grades it highly.
        goal_met = (critique and critique.get("applied")
                    and isinstance(critique.get("score"), int)
                    and critique["score"] >= args.stop_score)
        if goal_met:
            print(f"  goal achieved (score {critique['score']}/10)"
                  + ("" if args.auto else " — accept with [a] or keep refining"))
            if args.auto:
                break

        if i == args.iterations:
            break

        versions.append(output)
        backtrack = None
        stop = False
        if args.auto:
            prompt = next_prompt
        else:
            print("\n  [Enter] retry with proposed instruction   "
                  "[type text] your own feedback/instruction   "
                  "[b N] backtrack: base next round on version N "
                  f"(0=original, 1..{i}=iterations)   "
                  "[a] accept this result and stop   [q] quit")
            while True:
                try:
                    answer = input("  > ").strip()
                except (EOFError, KeyboardInterrupt):
                    print()
                    stop = True
                    break
                if answer.lower() == "q":
                    stop = True
                    break
                if answer.lower() == "a":
                    print(f"  accepted: {out_path}")
                    stop = True
                    break
                if answer.lower().startswith("b"):
                    parts = answer.split()
                    if len(parts) == 2 and parts[1].isdigit() and int(parts[1]) < len(versions):
                        k = int(parts[1])
                        backtrack = versions[k]
                        print(f"  next round will edit {'the original' if k == 0 else f'iteration {k}'}"
                              " — now pick the instruction ([Enter] for proposed).")
                        continue
                    print(f"  usage: b N with N in 0..{len(versions) - 1}")
                    continue
                prompt = answer if answer else next_prompt
                break
        if stop:
            break

        if backtrack is not None:
            reference = backtrack
        elif args.chain:
            reference = output

    if log["iterations"]:
        frames = [("input", original)] + [
            (str(i + 1), Image.open(it["output"]).convert("RGB"))
            for i, it in enumerate(log["iterations"])]
        strip_path = os.path.join(session, "filmstrip.png")
        build_film_strip(frames).save(strip_path)
        print(f"\n  film strip: {strip_path}")

    scored = [((it.get("critique") or {}).get("score"), n + 1, it["output"])
              for n, it in enumerate(log["iterations"])]
    scored = [(s, n, p) for s, n, p in scored if isinstance(s, int)]
    if scored:
        s, n, p = max(scored)
        print(f"  best iteration: {n} (score {s}/10) — {p}")

    print(f"\nSession saved: {session}")
    print(f"  {len(log['iterations'])} iteration(s); session.json has prompts, "
          "metrics, and critiques.")


if __name__ == "__main__":
    main()
