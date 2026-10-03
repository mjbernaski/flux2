"""Image Manager — a small web UI for browsing, deleting, moving, and cropping
images under web-generated/ and its subfolders.

Run:
    ./run_image_manager.sh
or
    python image_manager.py --port 2223
"""
import argparse
import io
import os
import shutil
import socket
from urllib.parse import quote

from flask import Flask, abort, jsonify, request, send_file
from PIL import Image
from dotenv import load_dotenv

load_dotenv()

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "web-generated"))
os.makedirs(ROOT, exist_ok=True)

API_KEY = (os.environ.get("FLUX_API_KEY") or "").strip()

IMG_EXTS = (".png", ".jpg", ".jpeg", ".webp")
SKIP_FILES = {".DS_Store"}
SKIP_PREFIXES = ("_preview_",)

app = Flask(__name__)


def check_auth():
    provided = request.headers.get("X-API-Key") or request.args.get("api_key")
    return provided == API_KEY


@app.before_request
def require_auth():
    if request.endpoint in ("index", "ready"):
        return
    if not API_KEY:
        return jsonify({"success": False, "error": "Server misconfigured: FLUX_API_KEY not set"}), 500
    if not check_auth():
        return jsonify({"success": False, "error": "Unauthorized"}), 401


def safe_resolve(rel_path):
    """Resolve a path relative to ROOT. Returns absolute path, or None if it
    would escape ROOT."""
    if rel_path is None:
        rel_path = ""
    rel_path = rel_path.replace("\\", "/").lstrip("/")
    abs_path = os.path.abspath(os.path.join(ROOT, rel_path))
    if abs_path != ROOT and not abs_path.startswith(ROOT + os.sep):
        return None
    return abs_path


def rel_of(abs_path):
    return os.path.relpath(abs_path, ROOT).replace(os.sep, "/")


def sidecar_path(abs_image_path):
    """Return the .prompt sidecar path for an image path."""
    return os.path.splitext(abs_image_path)[0] + ".prompt"


def _is_image(name):
    return (name not in SKIP_FILES and not name.startswith(SKIP_PREFIXES)
            and name.lower().endswith(IMG_EXTS))


# The one hidden folder. Deliberately the same name as web_server.py's
# HIDDEN_DIR_NAME: this gallery used to keep its own .hide/ alongside the
# generator's .hidden/, which meant an image could be "hidden" in two
# different places and only one UI could see each. migrate_legacy_hide()
# folds the old tree in at startup.
HIDDEN_DIR = ".hidden"

# Subfolders of the hidden tree that are not a gallery: step frames are
# preview decodes, and dot-folders are bookkeeping. Neither is mixed into the
# normal folder listings either, so neither belongs in the hidden one.
HIDDEN_SKIP_DIRS = {"steps"}


def _hidden_walk(abs_dir):
    """Every image under the hidden tree, minus the non-gallery subfolders."""
    for dirpath, dirs, files in os.walk(abs_dir):
        dirs[:] = [d for d in dirs if d not in HIDDEN_SKIP_DIRS and not d.startswith(".")]
        for n in files:
            if _is_image(n):
                yield os.path.join(dirpath, n)


def folder_tree(abs_dir, rel=""):
    if os.path.basename(abs_dir) == HIDDEN_DIR:
        # The hidden tree presents as a single folder: no children, and the
        # count spans every subfolder (originals keep their subpaths inside
        # .hidden so unhide can restore them).
        count = sum(1 for _ in _hidden_walk(abs_dir))
        return {"rel": rel, "name": HIDDEN_DIR, "hidden": True, "count": count, "children": []}
    children = []
    try:
        for item in sorted(os.listdir(abs_dir), key=str.lower):
            full = os.path.join(abs_dir, item)
            if os.path.isdir(full):
                children.append(folder_tree(full, (rel + "/" + item) if rel else item))
    except Exception:
        pass
    # Count image files directly in this folder
    count = 0
    try:
        for name in os.listdir(abs_dir):
            if _is_image(name) and os.path.isfile(os.path.join(abs_dir, name)):
                count += 1
    except Exception:
        pass
    return {
        "rel": rel,
        "name": os.path.basename(abs_dir) if rel else "(root)",
        "hidden": False,
        "count": count,
        "children": children,
    }


def migrate_legacy_hide():
    """Fold a pre-merge .hide/ tree into .hidden/, then drop the empty husk.

    Two hidden folders used to exist: this gallery's .hide/ and the
    generator's .hidden/. An image could therefore be "hidden" in two
    different places, and each UI could only see one of them. Subpaths carry
    over unchanged, so unhide still restores to the original folder.

    A file whose name is already taken on the other side is left where it is
    rather than overwritten — a merge is not worth losing an image to — and
    says so in the log, so the leftovers are visible instead of silent.
    """
    legacy = os.path.join(ROOT, ".hide")
    if not os.path.isdir(legacy):
        return
    moved = kept = 0
    for dirpath, _dirs, files in os.walk(legacy):
        for name in files:
            src = os.path.join(dirpath, name)
            rel = os.path.relpath(src, legacy).replace("\\", "/")
            dst = os.path.join(ROOT, HIDDEN_DIR, *rel.split("/"))
            if os.path.exists(dst):
                print(f"  kept (name already in {HIDDEN_DIR}/): {rel}")
                kept += 1
                continue
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.move(src, dst)
            moved += 1
    # Only prunes directories that came out empty, so anything left behind
    # (a collision above) keeps its folder and stays findable.
    for dirpath, _dirs, _files in os.walk(legacy, topdown=False):
        try:
            os.rmdir(dirpath)
        except OSError:
            pass
    if moved or kept:
        print(f"Merged legacy .hide/ into {HIDDEN_DIR}/: {moved} moved"
              + (f", {kept} left in place (name clash)" if kept else ""))


def flat_folder_list(tree, out=None):
    """Flatten the folder tree into [(rel, label)] pairs, depth-indented."""
    if out is None:
        out = []
    depth = tree["rel"].count("/") + (1 if tree["rel"] else 0)
    label = ("  " * depth) + tree["name"] + (f"  ({tree['count']})" if tree["count"] else "")
    out.append({"rel": tree["rel"], "label": label, "hidden": tree["hidden"]})
    for c in tree["children"]:
        flat_folder_list(c, out)
    return out


# ---------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------

@app.route("/ready")
def ready():
    return jsonify({"ready": True})


@app.route("/api/folders")
def api_folders():
    tree = folder_tree(ROOT)
    return jsonify({"tree": tree, "flat": flat_folder_list(tree)})


@app.route("/api/list")
def api_list():
    folder = request.args.get("folder", "")
    abs_dir = safe_resolve(folder)
    if not abs_dir or not os.path.isdir(abs_dir):
        return jsonify({"error": "Invalid folder"}), 400
    items = []
    try:
        if HIDDEN_DIR in [p for p in folder.replace("\\", "/").split("/") if p]:
            # The hidden tree shows as one folder — list it recursively.
            file_iter = ((os.path.dirname(f), os.path.basename(f))
                         for f in _hidden_walk(abs_dir))
        else:
            file_iter = ((abs_dir, n) for n in os.listdir(abs_dir))
        for parent, name in file_iter:
            if not _is_image(name):
                continue
            full = os.path.join(parent, name)
            if not os.path.isfile(full):
                continue
            try:
                stat = os.stat(full)
            except Exception:
                continue
            prompt = None
            sp = sidecar_path(full)
            if os.path.isfile(sp):
                try:
                    with open(sp, "r") as f:
                        for line in f:
                            if line.startswith("# Prompt: "):
                                prompt = line[10:].strip()
                                break
                except Exception:
                    pass
            items.append({
                "name": name,
                "rel": rel_of(full),
                "size": stat.st_size,
                "mtime": stat.st_mtime,
                "prompt": prompt,
            })
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    items.sort(key=lambda x: x["mtime"], reverse=True)
    return jsonify({"folder": folder, "items": items, "count": len(items)})


@app.route("/api/image")
def api_image():
    rel = request.args.get("path", "")
    abs_path = safe_resolve(rel)
    if not abs_path or not os.path.isfile(abs_path):
        abort(404)
    return send_file(abs_path)


@app.route("/api/thumb")
def api_thumb():
    rel = request.args.get("path", "")
    abs_path = safe_resolve(rel)
    if not abs_path or not os.path.isfile(abs_path):
        abort(404)
    try:
        size = max(64, min(512, int(request.args.get("size", 256))))
    except Exception:
        size = 256
    try:
        img = Image.open(abs_path)
        img.thumbnail((size, size))
        if img.mode in ("RGBA", "P", "LA"):
            img = img.convert("RGB")
        buf = io.BytesIO()
        img.save(buf, "JPEG", quality=82)
        buf.seek(0)
        return send_file(buf, mimetype="image/jpeg")
    except Exception:
        abort(500)


@app.route("/api/delete", methods=["POST"])
def api_delete():
    body = request.get_json(silent=True) or {}
    rel = body.get("path", "")
    abs_path = safe_resolve(rel)
    if not abs_path or not os.path.isfile(abs_path):
        return jsonify({"success": False, "error": "File not found"}), 404
    try:
        os.remove(abs_path)
        sp = sidecar_path(abs_path)
        if os.path.isfile(sp):
            os.remove(sp)
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500
    return jsonify({"success": True})


@app.route("/api/move", methods=["POST"])
def api_move():
    body = request.get_json(silent=True) or {}
    src_rel = body.get("path", "")
    dst_folder = body.get("folder", "")
    src_abs = safe_resolve(src_rel)
    dst_dir_abs = safe_resolve(dst_folder)
    if not src_abs or not os.path.isfile(src_abs):
        return jsonify({"success": False, "error": "Source not found"}), 404
    if dst_dir_abs is None:
        return jsonify({"success": False, "error": "Invalid destination folder"}), 400
    os.makedirs(dst_dir_abs, exist_ok=True)
    name = os.path.basename(src_abs)
    dst_abs = os.path.join(dst_dir_abs, name)
    if os.path.abspath(src_abs) == os.path.abspath(dst_abs):
        return jsonify({"success": False, "error": "Source and destination are the same"}), 400
    if os.path.exists(dst_abs):
        return jsonify({"success": False, "error": f"Destination already has a file named {name}"}), 409
    try:
        shutil.move(src_abs, dst_abs)
        src_prompt = sidecar_path(src_abs)
        if os.path.isfile(src_prompt):
            shutil.move(src_prompt, sidecar_path(dst_abs))
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500
    return jsonify({"success": True, "new_path": rel_of(dst_abs)})


@app.route("/api/hide", methods=["POST"])
def api_hide():
    """Toggle hidden: move into (or out of) the single .hidden/ tree at ROOT.

    The image's subfolder path is preserved inside .hidden/ (archive/foo.png
    hides to .hidden/archive/foo.png), so unhide restores it to where it came
    from — and it lands in the same tree the generator writes its hidden-mode
    output to, so "hidden" means one place for the whole stack.

    Unhide still accepts the two legacy layouts, because files hidden before
    the merge are still named that way on any box that has not run the
    startup migration: the old top-level .hide/ tree, and the older
    per-folder archive/.hide/foo.png. Either path component is dropped.
    """
    body = request.get_json(silent=True) or {}
    rel = body.get("path", "")
    abs_path = safe_resolve(rel)
    if not abs_path or not os.path.isfile(abs_path):
        return jsonify({"success": False, "error": "File not found"}), 404
    parts = rel_of(abs_path).split("/")
    if HIDDEN_DIR in parts:
        parts.remove(HIDDEN_DIR)
        action = "unhide"
    elif ".hide" in parts:                      # legacy, pre-merge
        parts.remove(".hide")
        action = "unhide"
    else:
        parts.insert(0, HIDDEN_DIR)
        action = "hide"
    dst_abs = os.path.join(ROOT, *parts)
    if os.path.exists(dst_abs):
        return jsonify({"success": False, "error": f"Destination already has a file named {parts[-1]}"}), 409
    os.makedirs(os.path.dirname(dst_abs), exist_ok=True)
    try:
        shutil.move(abs_path, dst_abs)
        src_prompt = sidecar_path(abs_path)
        if os.path.isfile(src_prompt):
            shutil.move(src_prompt, sidecar_path(dst_abs))
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500
    return jsonify({"success": True, "action": action, "new_path": rel_of(dst_abs)})


@app.route("/api/crop", methods=["POST"])
def api_crop():
    """Crop a rectangle. Body: {path, left, top, right, bottom, mode: 'replace'|'new'}
    Coords are in the original image's pixel space."""
    body = request.get_json(silent=True) or {}
    rel = body.get("path", "")
    abs_path = safe_resolve(rel)
    if not abs_path or not os.path.isfile(abs_path):
        return jsonify({"success": False, "error": "File not found"}), 404
    try:
        left = int(round(float(body.get("left", 0))))
        top = int(round(float(body.get("top", 0))))
        right = int(round(float(body.get("right", 0))))
        bottom = int(round(float(body.get("bottom", 0))))
    except Exception:
        return jsonify({"success": False, "error": "Invalid coordinates"}), 400
    mode = body.get("mode", "replace")
    try:
        img = Image.open(abs_path)
        w, h = img.size
        left = max(0, min(left, w))
        top = max(0, min(top, h))
        right = max(left + 1, min(right, w))
        bottom = max(top + 1, min(bottom, h))
        if right - left < 2 or bottom - top < 2:
            return jsonify({"success": False, "error": "Crop area is too small"}), 400
        cropped = img.crop((left, top, right, bottom))
        if mode == "new":
            base, ext = os.path.splitext(abs_path)
            dst = f"{base}_crop{ext}"
            i = 2
            while os.path.exists(dst):
                dst = f"{base}_crop{i}{ext}"
                i += 1
            cropped.save(dst)
            src_prompt = sidecar_path(abs_path)
            if os.path.isfile(src_prompt):
                shutil.copy2(src_prompt, sidecar_path(dst))
            return jsonify({"success": True, "new_path": rel_of(dst), "replaced": False})
        else:
            cropped.save(abs_path)
            return jsonify({"success": True, "new_path": rel, "replaced": True,
                            "size": {"w": cropped.width, "h": cropped.height}})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

HTML_PAGE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Image Manager</title>
<style>
  :root {
    --bg: #1a1a1a; --panel: #232323; --panel2: #2b2b2b;
    --fg: #e0e0e0; --muted: #888; --accent: #5aa8ff; --danger: #e05050;
    --border: #333;
  }
  * { box-sizing: border-box; }
  body { margin: 0; font: 14px system-ui, -apple-system, sans-serif;
         background: var(--bg); color: var(--fg); height: 100vh; display: flex; flex-direction: column; }
  header { padding: 8px 12px; background: var(--panel); border-bottom: 1px solid var(--border);
           display: flex; gap: 10px; align-items: center; flex-wrap: wrap; }
  header h1 { font-size: 16px; margin: 0; margin-right: 6px; }
  header select, header input, header button {
    background: var(--panel2); color: var(--fg); border: 1px solid var(--border);
    padding: 5px 8px; border-radius: 4px; font: inherit;
  }
  header button { cursor: pointer; }
  header button:hover { background: #363636; }
  .muted { color: var(--muted); }
  main { flex: 1; overflow: auto; padding: 12px; }
  #grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(200px, 1fr)); gap: 10px; }
  .card { background: var(--panel); border: 1px solid var(--border); border-radius: 6px; overflow: hidden;
          display: flex; flex-direction: column; }
  .card .thumb { background: #000; aspect-ratio: 1 / 1; display: flex; align-items: center; justify-content: center;
                 cursor: pointer; overflow: hidden; }
  .card .thumb img { max-width: 100%; max-height: 100%; object-fit: contain; }
  .card .meta { padding: 6px 8px; font-size: 12px; color: var(--muted); word-break: break-all; }
  .card .meta .name { color: var(--fg); font-size: 11px; }
  .card .actions { display: flex; gap: 4px; padding: 6px; border-top: 1px solid var(--border); }
  .card .actions button { flex: 1; background: var(--panel2); color: var(--fg);
                          border: 1px solid var(--border); border-radius: 3px; padding: 4px 2px;
                          font-size: 11px; cursor: pointer; }
  .card .actions button:hover { background: #363636; }
  .card .actions button.danger:hover { background: var(--danger); border-color: var(--danger); }
  .card { position: relative; }
  .card.selected { outline: 2px solid var(--accent); outline-offset: -2px; }
  #selRect { position: absolute; border: 1px solid var(--accent);
             background: rgba(90,168,255,0.12); z-index: 50;
             pointer-events: none; display: none; }
  .card .select-cb { position: absolute; top: 6px; left: 6px; width: 22px; height: 22px;
                     background: rgba(0,0,0,0.6); border: 1px solid var(--border);
                     border-radius: 4px; cursor: pointer; z-index: 2; user-select: none;
                     display: flex; align-items: center; justify-content: center;
                     font-size: 14px; font-weight: bold; color: transparent; }
  .card .select-cb:hover { background: rgba(0,0,0,0.85); }
  .card.selected .select-cb { background: var(--accent); border-color: var(--accent); color: #000; }
  header button.danger { background: var(--danger); border-color: var(--danger); color: #fff; }
  header button.danger:hover { background: #ff6a6a; }
  header button:disabled { opacity: 0.5; cursor: not-allowed; }

  #modal { position: fixed; inset: 0; background: rgba(0,0,0,0.85); display: none;
           align-items: center; justify-content: center; z-index: 10; }
  #modal.open { display: flex; }
  .modal-box { background: var(--panel); border: 1px solid var(--border); border-radius: 6px;
               max-width: 95vw; max-height: 95vh; display: flex; flex-direction: column; overflow: hidden; }
  .modal-head { display: flex; justify-content: space-between; align-items: center;
                padding: 8px 12px; border-bottom: 1px solid var(--border); gap: 10px; }
  .modal-head .title { font-weight: 600; word-break: break-all; font-size: 13px; }
  .modal-body { display: flex; flex: 1; min-height: 0; }
  .modal-img-wrap { flex: 1; min-width: 0; min-height: 0; background: #000;
                    display: flex; align-items: center; justify-content: center; position: relative;
                    overflow: hidden; }
  #modalImg { max-width: 100%; max-height: 80vh; object-fit: contain; display: block; user-select: none;
              -webkit-user-drag: none; }
  .crop-overlay { position: absolute; inset: 0; cursor: crosshair; }
  .crop-rect { position: absolute; border: 2px dashed var(--accent); background: rgba(90,168,255,0.12);
               box-shadow: 0 0 0 9999px rgba(0,0,0,0.5); pointer-events: none; }
  .modal-side { width: 280px; padding: 12px; border-left: 1px solid var(--border);
                display: flex; flex-direction: column; gap: 10px; overflow: auto; }
  .modal-side h3 { margin: 0; font-size: 13px; color: var(--muted); text-transform: uppercase; letter-spacing: 0.5px; }
  .modal-side .row { display: flex; gap: 6px; align-items: center; }
  .modal-side select, .modal-side button {
    background: var(--panel2); color: var(--fg); border: 1px solid var(--border);
    padding: 6px 8px; border-radius: 4px; font: inherit;
  }
  .modal-side button { cursor: pointer; }
  .modal-side button:hover { background: #363636; }
  .modal-side button.primary { background: var(--accent); border-color: var(--accent); color: #000; font-weight: 600; }
  .modal-side button.primary:hover { background: #7ab8ff; }
  .modal-side button.danger { background: var(--danger); border-color: var(--danger); color: #fff; }
  .modal-side .prompt { background: var(--panel2); padding: 6px 8px; border-radius: 4px;
                        font-size: 12px; white-space: pre-wrap; word-break: break-word; max-height: 120px; overflow: auto; }
  .close-btn { background: transparent; border: none; color: var(--fg); font-size: 20px; cursor: pointer; }

  /* Pixel compare overlay (header "Compare" button, two images selected) */
  #cmpOverlay { position: fixed; inset: 0; background: rgba(0,0,0,0.92); display: none;
                flex-direction: column; align-items: center; justify-content: center;
                gap: 12px; padding: 20px; z-index: 80; overflow-y: auto; }
  #cmpOverlay.open { display: flex; }
  /* 62vh, not 80: leaves room for the HUD and the explanation below it. */
  .cmp-stage { max-width: 95vw; max-height: 62vh; overflow: auto; }
  .cmp-stage canvas { display: block; max-width: 100%; height: auto;
                      /* checkerboard shows through wherever the two images disagree */
                      background: repeating-conic-gradient(#2a2a2a 0% 25%, #454545 0% 50%) 0 0 / 20px 20px; }
  .cmp-hud { display: flex; align-items: center; justify-content: center; flex-wrap: wrap;
             gap: 14px; font-size: 14px; }
  .cmp-hud label { display: flex; align-items: center; gap: 8px; }
  .cmp-help { max-width: 70ch; margin: 0; text-align: center; font-size: 12px;
              line-height: 1.5; color: #9a9a9a; }
  .cmp-help b { color: #cfcfcf; font-weight: 600; }
  #cmpClose { position: absolute; top: 14px; right: 14px; width: 36px; height: 36px;
              background: rgba(255,255,255,0.12); color: #fff; border: none; border-radius: 50%;
              font-size: 16px; cursor: pointer; }
  #cmpClose:hover { background: rgba(255,255,255,0.25); }

  /* ---- Slideshow: full-bleed review of a folder at full resolution ----
     Separate from #modal on purpose: the modal is a work surface (crop, move,
     info) sized to leave room for its side panel, where this gives the whole
     viewport to the picture. */
  #show { position: fixed; inset: 0; background: #000; display: none; z-index: 90;
          cursor: default; }
  #show.open { display: block; }
  #show.idle { cursor: none; }
  /* Two stacked layers: the next image decodes in the back one while the
     current is still on screen, and advancing just swaps which is opaque. A
     single <img> would blank between slides however fast the fetch was. */
  .show-layer { position: absolute; inset: 0; display: flex;
                align-items: center; justify-content: center;
                opacity: 0; transition: opacity 160ms linear; }
  .show-layer.front { opacity: 1; }
  .show-layer img { max-width: 100%; max-height: 100%; object-fit: contain;
                    display: block; user-select: none; -webkit-user-drag: none; }
  /* Actual-pixels mode: no scaling at all, scroll the overflow instead. The
     only way to judge render detail without the browser's resampling in the
     way - which is the whole point of a slideshow over the thumbnail grid. */
  #show.actual .show-layer { overflow: auto; align-items: safe center; justify-content: safe center; }
  #show.actual .show-layer img { max-width: none; max-height: none; width: auto; height: auto; }

  .show-top, .show-hud { position: absolute; left: 0; right: 0; z-index: 2;
                         display: flex; align-items: center; gap: 10px; flex-wrap: wrap;
                         padding: 10px 14px; font-size: 13px;
                         transition: opacity 250ms linear; }
  .show-top { top: 0; background: linear-gradient(rgba(0,0,0,0.8), transparent); }
  .show-hud { bottom: 0; background: linear-gradient(transparent, rgba(0,0,0,0.8)); }
  /* Chrome hides the controls after a few idle seconds so a running show is
     just the picture; any mouse move or key brings them back. */
  #show.idle .show-top, #show.idle .show-hud { opacity: 0; pointer-events: none; }
  .show-hud button, .show-top button, .show-hud select {
    background: rgba(255,255,255,0.1); color: var(--fg); border: 1px solid rgba(255,255,255,0.2);
    padding: 5px 10px; border-radius: 4px; font: inherit; cursor: pointer; }
  .show-hud button:hover, .show-top button:hover { background: rgba(255,255,255,0.2); }
  .show-hud button.on { background: var(--accent); border-color: var(--accent); color: #000; }
  .show-hud button.danger:hover { background: var(--danger); border-color: var(--danger); color: #fff; }
  /* Armed delete: the second press is the one that acts, so it has to look
     unmistakably different from the first. */
  .show-hud button.armed { background: var(--danger); border-color: var(--danger); color: #fff;
                           animation: armPulse 0.9s ease-in-out infinite; }
  @keyframes armPulse { 50% { opacity: 0.55; } }
  .show-count { font-variant-numeric: tabular-nums; color: var(--muted); }
  .show-name { color: var(--fg); word-break: break-all; font-size: 12px;
               max-width: 48vw; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
  .show-grow { flex: 1; }
  /* Dwell-time bar: a slide's remaining time, restarted per slide by
     re-triggering the animation. Hidden while paused. */
  #showBar { position: absolute; top: 0; left: 0; height: 3px; width: 0;
             background: var(--accent); z-index: 3; }
  #showBar.run { animation: showBarFill linear forwards; }
  @keyframes showBarFill { from { width: 0; } to { width: 100%; } }
  .show-info { position: absolute; top: 46px; left: 14px; z-index: 2; max-width: min(46ch, 60vw);
               background: rgba(0,0,0,0.78); border: 1px solid var(--border); border-radius: 6px;
               padding: 10px 12px; font-size: 12px; display: none; }
  .show-info.on { display: block; }
  .show-info dt { color: var(--muted); float: left; clear: left; width: 6.5em; }
  .show-info dd { margin: 0 0 3px 7em; word-break: break-all; }
  .show-info .sp { margin: 8px 0 0; padding-top: 8px; border-top: 1px solid var(--border);
                   white-space: pre-wrap; word-break: break-word; max-height: 30vh; overflow: auto; }
  .show-keys { position: absolute; right: 14px; top: 46px; z-index: 2;
               background: rgba(0,0,0,0.78); border: 1px solid var(--border); border-radius: 6px;
               padding: 10px 12px; font-size: 12px; display: none; }
  .show-keys.on { display: block; }
  .show-keys kbd { background: var(--panel2); border: 1px solid var(--border); border-radius: 3px;
                   padding: 1px 5px; font: 11px ui-monospace, monospace; }
  .show-keys tr td { padding: 2px 4px; }
  /* Click zones: the left and right thirds step the show, so it works with a
     mouse or a phone tap and not only the keyboard. */
  .show-zone { position: absolute; top: 0; bottom: 0; width: 28%; z-index: 1; }
  .show-zone.prev { left: 0; } .show-zone.next { right: 0; }

  #toast { position: fixed; bottom: 20px; left: 50%; transform: translateX(-50%);
           background: var(--panel); border: 1px solid var(--border); border-radius: 4px;
           padding: 8px 16px; z-index: 100; display: none; }
  #toast.error { border-color: var(--danger); }
  #toast.show { display: block; }

  .empty { color: var(--muted); text-align: center; padding: 40px; }
  .flex-grow { flex-grow: 1; }

  /* Make selected folder option easy to read */
  #folderSel option { font-family: ui-monospace, monospace; }
</style>
</head>
<body>

<header>
  <h1>Image Manager</h1>
  <label>Folder:
    <select id="folderSel"></select>
  </label>
  <span id="count" class="muted"></span>
  <button id="selectAllBtn" title="Select all images in this folder">Select all</button>
  <button id="showBtn" title="Full-screen slideshow of this folder at full resolution (or of the selection, if there is one)">&#9654; Slideshow</button>
  <span id="selCount" class="muted" style="display:none"></span>
  <button id="cmpBtn" style="display:none" title="Show only the pixels the two selected images share">Compare &#x29C9;</button>
  <button id="bulkHideBtn" style="display:none">Hide selected</button>
  <button id="bulkDeleteBtn" class="danger" style="display:none">Delete selected</button>
  <button id="clearSelBtn" style="display:none">Clear</button>
  <span class="flex-grow"></span>
  <input id="apiKey" type="password" placeholder="API key" size="24">
  <button id="refresh">Refresh</button>
</header>

<main>
  <div id="grid"></div>
  <div id="empty" class="empty" style="display:none">No images in this folder.</div>
</main>

<div id="modal">
  <div class="modal-box">
    <div class="modal-head">
      <div class="title" id="modalTitle"></div>
      <button class="close-btn" id="closeBtn">&times;</button>
    </div>
    <div class="modal-body">
      <div class="modal-img-wrap">
        <img id="modalImg" src="" alt="">
        <div class="crop-overlay" id="cropOverlay" style="display:none">
          <div class="crop-rect" id="cropRect" style="display:none"></div>
        </div>
      </div>
      <div class="modal-side">
        <h3>Info</h3>
        <div id="modalInfo" class="muted" style="font-size:12px"></div>
        <div id="modalPrompt" class="prompt" style="display:none"></div>

        <h3>Move to folder</h3>
        <div class="row">
          <select id="moveSel" class="flex-grow" style="flex:1"></select>
          <button id="moveBtn">Move</button>
        </div>

        <h3>Hide</h3>
        <button id="hideBtn">Toggle hide</button>

        <h3>Slideshow</h3>
        <button id="showFromHereBtn">Start here &#9654;</button>

        <h3>Crop</h3>
        <div class="muted" style="font-size:12px">Drag on the image to select a region.</div>
        <div id="cropCoords" class="muted" style="font-size:11px; font-family:ui-monospace,monospace"></div>
        <div class="row">
          <button id="cropResetBtn">Clear selection</button>
        </div>
        <div class="row">
          <button id="cropReplaceBtn" class="primary" disabled>Crop &amp; replace</button>
        </div>
        <div class="row">
          <button id="cropNewBtn" disabled>Crop as new file</button>
        </div>

        <div style="flex-grow:1"></div>
        <h3>Danger zone</h3>
        <button id="deleteBtn" class="danger">Delete permanently</button>
      </div>
    </div>
  </div>
</div>

<div id="show">
  <div id="showBar"></div>
  <div class="show-layer" id="showLayer0"><img alt=""></div>
  <div class="show-layer" id="showLayer1"><img alt=""></div>
  <div class="show-zone prev" id="showZonePrev" title="Previous"></div>
  <div class="show-zone next" id="showZoneNext" title="Next"></div>

  <div class="show-top">
    <span class="show-count" id="showCount"></span>
    <span class="show-name" id="showName"></span>
    <span class="show-grow"></span>
    <button id="showClose" title="Close (Esc)">&times;</button>
  </div>

  <div class="show-info" id="showInfo"></div>

  <div class="show-keys" id="showKeys">
    <table>
      <tr><td><kbd>&larr;</kbd> <kbd>&rarr;</kbd></td><td>previous / next</td></tr>
      <tr><td><kbd>Space</kbd></td><td>play / pause</td></tr>
      <tr><td><kbd>H</kbd></td><td>hide (reversible)</td></tr>
      <tr><td><kbd>X</kbd></td><td>delete &mdash; press twice</td></tr>
      <tr><td><kbd>A</kbd></td><td>fit / actual pixels</td></tr>
      <tr><td><kbd>F</kbd></td><td>full screen</td></tr>
      <tr><td><kbd>I</kbd></td><td>image info</td></tr>
      <tr><td><kbd>?</kbd></td><td>this list</td></tr>
      <tr><td><kbd>Esc</kbd></td><td>close</td></tr>
    </table>
  </div>

  <div class="show-hud">
    <button id="showPrev" title="Previous (&larr;)">&#9664;</button>
    <button id="showPlay" title="Play / pause (Space)">&#9654;</button>
    <button id="showNext" title="Next (&rarr;)">&#9654;&#9654;</button>
    <label>Every
      <select id="showEvery">
        <option value="1500">1.5s</option>
        <option value="3000">3s</option>
        <option value="5000" selected>5s</option>
        <option value="8000">8s</option>
        <option value="15000">15s</option>
      </select>
    </label>
    <span class="show-grow"></span>
    <button id="showHideBtn" title="Move to the hidden tree (H) - reversible">Hide</button>
    <button id="showDeleteBtn" class="danger" title="Delete permanently (X, press twice)">Delete</button>
    <button id="showActual" title="Fit to screen / actual pixels (A)">Fit</button>
    <button id="showFs" title="Full screen (F)">&#9974;</button>
    <button id="showInfoBtn" title="Image info (I)">Info</button>
    <button id="showKeysBtn" title="Keyboard shortcuts (?)">?</button>
  </div>
</div>

<div id="cmpOverlay">
  <button id="cmpClose" title="Close (Esc)">&times;</button>
  <div class="cmp-stage"><canvas id="cmpCanvas"></canvas></div>
  <div class="cmp-hud">
    <span id="cmpStat"></span>
    <label>Tolerance <input type="range" id="cmpTol" min="0" max="48" step="1" value="8"></label>
    <span id="cmpTolVal">8</span>
  </div>
  <p class="cmp-help">
    Only the pixels the two images agree on are drawn, taken from the
    first-selected image. A pixel is kept when its red, green and blue all
    differ by <b>&le; <span id="cmpTolEcho">8</span> of 255</b> from the other
    image's — each channel is judged on its own, so one channel drifting too far
    drops the pixel. Everything else is transparent and shows the checkerboard.
    <b>0</b> keeps only exact matches; raising it forgives the sub-level drift
    between two runs of the same seed, and past ~24 it starts merging genuinely
    different content. Alpha is ignored, and the match percentage above moves
    with this slider.
  </p>
</div>

<div id="toast"></div>

<script>
const $ = sel => document.querySelector(sel);
const state = {
  folders: [],       // flat folder list (visible subset, filtered by altHeld)
  allFolders: [],    // flat folder list, unfiltered (includes .hidden)
  altHeld: false,    // Option/Alt key currently held — reveals the .hidden folder
  items: [],         // current folder images
  current: null,     // selected image item
  naturalSize: null, // {w,h} of current image
  cropBox: null,     // {x1,y1,x2,y2} in image pixel coords
  selected: new Set(), // rel paths of selected images in the current folder
  lastClickedRel: null, // for shift-click range selection
};

function apiKey() { return $('#apiKey').value.trim(); }

function authQuery() {
  const k = apiKey();
  return k ? ('api_key=' + encodeURIComponent(k)) : '';
}

async function api(path, opts={}) {
  const url = path + (path.includes('?') ? '&' : '?') + authQuery();
  const init = { ...opts };
  init.headers = Object.assign({}, opts.headers || {}, { 'X-API-Key': apiKey() });
  if (init.body && typeof init.body === 'object' && !(init.body instanceof FormData)) {
    init.headers['Content-Type'] = 'application/json';
    init.body = JSON.stringify(init.body);
  }
  const r = await fetch(url, init);
  if (!r.ok) {
    const t = await r.text();
    try { const j = JSON.parse(t); throw new Error(j.error || r.statusText); }
    catch (e) { throw e instanceof Error ? e : new Error(t || r.statusText); }
  }
  return r.json();
}

function toast(msg, isErr=false) {
  const t = $('#toast');
  t.textContent = msg;
  t.className = 'show' + (isErr ? ' error' : '');
  clearTimeout(toast._t);
  toast._t = setTimeout(() => t.className = '', 2500);
}

function fmtBytes(n) {
  if (n < 1024) return n + ' B';
  if (n < 1024*1024) return (n/1024).toFixed(1) + ' KB';
  return (n/1024/1024).toFixed(1) + ' MB';
}

function fmtDate(ts) {
  const d = new Date(ts * 1000);
  return d.toLocaleString();
}

function imgUrl(rel, thumb=false) {
  const base = thumb ? '/api/thumb' : '/api/image';
  const k = apiKey();
  return base + '?path=' + encodeURIComponent(rel) + (k ? '&api_key=' + encodeURIComponent(k) : '');
}

async function loadFolders() {
  const data = await api('/api/folders');
  state.allFolders = data.flat;
  renderFolderOptions();
}

function visibleFolders() {
  return state.altHeld ? state.allFolders : state.allFolders.filter(f => !f.hidden);
}

function renderFolderOptions() {
  const visible = visibleFolders();
  state.folders = visible;
  const sel = $('#folderSel');
  const move = $('#moveSel');
  const prevSel = sel.value;
  const prevMove = move.value;
  sel.innerHTML = '';
  move.innerHTML = '';
  for (const f of visible) {
    const opt = document.createElement('option');
    opt.value = f.rel;
    opt.textContent = f.label + (f.hidden ? '  [hidden]' : '');
    sel.appendChild(opt);
    const opt2 = opt.cloneNode(true);
    move.appendChild(opt2);
  }
  if (prevSel && visible.some(f => f.rel === prevSel)) sel.value = prevSel;
  if (prevMove && visible.some(f => f.rel === prevMove)) move.value = prevMove;
}

// Hold Option (Alt) to reveal the .hidden folder in the folder dropdowns.
document.addEventListener('keydown', e => {
  if (e.key !== 'Alt' || state.altHeld) return;
  state.altHeld = true;
  renderFolderOptions();
});
document.addEventListener('keyup', e => {
  if (e.key !== 'Alt' || !state.altHeld) return;
  releaseAltHeld();
});
window.addEventListener('blur', () => {
  if (state.altHeld) releaseAltHeld();
});

function releaseAltHeld() {
  state.altHeld = false;
  const sel = $('#folderSel');
  const wasHidden = sel.value.split('/').includes('.hidden');
  renderFolderOptions();
  if (wasHidden) {
    sel.value = '';
    loadList('').catch(e => toast(e.message, true));
  }
}

async function loadList(folder) {
  const data = await api('/api/list?folder=' + encodeURIComponent(folder));
  state.items = data.items;
  // Drop selections for items no longer present (e.g., after a delete).
  const present = new Set(state.items.map(it => it.rel));
  for (const rel of Array.from(state.selected)) {
    if (!present.has(rel)) state.selected.delete(rel);
  }
  $('#count').textContent = `${data.count} image${data.count === 1 ? '' : 's'}`;
  renderGrid();
  updateSelectionUI();
}

function updateSelectionUI() {
  const n = state.selected.size;
  $('#selCount').style.display = n ? 'inline' : 'none';
  $('#selCount').textContent = n ? `${n} selected` : '';
  $('#bulkDeleteBtn').style.display = n ? 'inline-block' : 'none';
  $('#bulkHideBtn').style.display = n ? 'inline-block' : 'none';
  $('#bulkHideBtn').textContent =
    $('#folderSel').value.split('/').includes('.hidden') ? 'Unhide selected' : 'Hide selected';
  $('#cmpBtn').style.display = n === 2 ? 'inline-block' : 'none';
  $('#clearSelBtn').style.display = n ? 'inline-block' : 'none';
  $('#selectAllBtn').style.display = state.items.length ? 'inline-block' : 'none';
  $('#selectAllBtn').textContent =
    (n && n === state.items.length) ? 'Deselect all' : 'Select all';
}

function toggleSelect(rel, shiftKey=false) {
  if (shiftKey && state.lastClickedRel && state.lastClickedRel !== rel) {
    const rels = state.items.map(it => it.rel);
    const a = rels.indexOf(state.lastClickedRel);
    const b = rels.indexOf(rel);
    if (a >= 0 && b >= 0) {
      const [lo, hi] = a < b ? [a, b] : [b, a];
      const shouldSelect = !state.selected.has(rel);
      for (let i = lo; i <= hi; i++) {
        if (shouldSelect) state.selected.add(rels[i]);
        else state.selected.delete(rels[i]);
      }
      state.lastClickedRel = rel;
      renderGrid();
      updateSelectionUI();
      return;
    }
  }
  if (state.selected.has(rel)) state.selected.delete(rel);
  else state.selected.add(rel);
  state.lastClickedRel = rel;
  // Update just this card's class without full re-render
  const card = document.querySelector(`.card[data-rel="${CSS.escape(rel)}"]`);
  if (card) card.classList.toggle('selected', state.selected.has(rel));
  updateSelectionUI();
}

function renderGrid() {
  const grid = $('#grid');
  grid.innerHTML = '';
  if (state.items.length === 0) {
    $('#empty').style.display = 'block';
    return;
  }
  $('#empty').style.display = 'none';
  for (const it of state.items) {
    const card = document.createElement('div');
    card.className = 'card' + (state.selected.has(it.rel) ? ' selected' : '');
    card.dataset.rel = it.rel;
    card.innerHTML = `
      <div class="select-cb" title="Select (shift-click for range)">✓</div>
      <div class="thumb" title="Click to open">
        <img loading="lazy" src="${imgUrl(it.rel, true)}" alt="">
      </div>
      <div class="meta">
        <div class="name">${it.name}</div>
        <div>${fmtBytes(it.size)} · ${new Date(it.mtime*1000).toLocaleString()}</div>
      </div>
      <div class="actions">
        <button data-act="open">Open</button>
        <button data-act="hide">Hide</button>
        <button class="danger" data-act="delete">Delete</button>
      </div>`;
    const cb = card.querySelector('.select-cb');
    cb.addEventListener('click', e => {
      e.stopPropagation();
      toggleSelect(it.rel, e.shiftKey);
    });
    const thumb = card.querySelector('.thumb');
    thumb.addEventListener('click', e => {
      // Shift-click on the thumb toggles selection (range with prior click).
      if (e.shiftKey) { toggleSelect(it.rel, true); return; }
      openModal(it);
    });
    card.querySelectorAll('[data-act]').forEach(btn => {
      btn.addEventListener('click', e => {
        e.stopPropagation();
        const act = btn.getAttribute('data-act');
        if (act === 'open') openModal(it);
        else if (act === 'delete') quickDelete(it);
        else if (act === 'hide') quickHide(it);
      });
    });
    grid.appendChild(card);
  }
}

async function quickDelete(it) {
  if (!confirm(`Permanently delete ${it.name}?`)) return;
  try {
    await api('/api/delete', { method: 'POST', body: { path: it.rel } });
    toast('Deleted');
    refresh();
  } catch (e) { toast(e.message, true); }
}

async function quickHide(it) {
  try {
    const r = await api('/api/hide', { method: 'POST', body: { path: it.rel } });
    toast(r.action === 'hide' ? 'Hidden' : 'Unhidden');
    refresh();
  } catch (e) { toast(e.message, true); }
}

// ---- Drag (rubber-band) selection ----
(function setupDragSelect() {
  const grid = $('#grid');
  const rect = document.createElement('div');
  rect.id = 'selRect';
  document.body.appendChild(rect);
  let anchor = null;        // drag start, page coords
  let active = false;       // true once the pointer moved past the threshold
  let baseSelected = null;  // selection at drag start (drag adds to it)

  grid.addEventListener('mousedown', e => {
    if (e.button !== 0) return;
    if (e.target.closest('button') || e.target.closest('.select-cb')) return;
    anchor = { x: e.pageX, y: e.pageY };
    baseSelected = new Set(state.selected);
    active = false;
    e.preventDefault();  // stop native image drag / text selection
  });

  document.addEventListener('mousemove', e => {
    if (!anchor) return;
    if (!(e.buttons & 1)) {
      // Button already released (e.g. mouseup happened outside the window):
      // abandon the drag instead of letting it stick to the pointer.
      anchor = null; active = false; baseSelected = null;
      rect.style.display = 'none';
      return;
    }
    if (!active && Math.hypot(e.pageX - anchor.x, e.pageY - anchor.y) < 6) return;
    active = true;
    const x1 = Math.min(anchor.x, e.pageX), x2 = Math.max(anchor.x, e.pageX);
    const y1 = Math.min(anchor.y, e.pageY), y2 = Math.max(anchor.y, e.pageY);
    Object.assign(rect.style, { display: 'block', left: x1 + 'px', top: y1 + 'px',
                                width: (x2 - x1) + 'px', height: (y2 - y1) + 'px' });
    state.selected = new Set(baseSelected);
    for (const card of grid.children) {
      const r = card.getBoundingClientRect();
      const cx1 = r.left + scrollX, cy1 = r.top + scrollY;
      if (cx1 < x2 && cx1 + r.width > x1 && cy1 < y2 && cy1 + r.height > y1)
        state.selected.add(card.dataset.rel);
      card.classList.toggle('selected', state.selected.has(card.dataset.rel));
    }
    updateSelectionUI();
  });

  document.addEventListener('mouseup', () => {
    if (!anchor) return;
    const wasDrag = active;
    anchor = null; active = false; baseSelected = null;
    rect.style.display = 'none';
    if (wasDrag) suppressNextClick = true;  // don't open the modal under the cursor
  });

  let suppressNextClick = false;
  document.addEventListener('click', e => {
    if (suppressNextClick) {
      suppressNextClick = false;
      e.stopPropagation();
      e.preventDefault();
    }
  }, true);
})();

// ---- Modal ----
function openModal(it) {
  state.current = it;
  state.cropBox = null;
  state.naturalSize = null;
  $('#modal').classList.add('open');
  $('#modalTitle').textContent = it.name;
  $('#modalInfo').textContent = `${it.rel}  ·  ${fmtBytes(it.size)}  ·  ${fmtDate(it.mtime)}`;
  const promptEl = $('#modalPrompt');
  if (it.prompt) { promptEl.style.display = 'block'; promptEl.textContent = it.prompt; }
  else { promptEl.style.display = 'none'; }
  const img = $('#modalImg');
  img.onload = () => {
    state.naturalSize = { w: img.naturalWidth, h: img.naturalHeight };
    $('#cropOverlay').style.display = 'block';
    $('#cropRect').style.display = 'none';
    $('#cropCoords').textContent = `image: ${img.naturalWidth} × ${img.naturalHeight}`;
    $('#cropReplaceBtn').disabled = true;
    $('#cropNewBtn').disabled = true;
  };
  img.src = imgUrl(it.rel) + '&t=' + Date.now();
  // Preselect the current folder in the move dropdown
  const currentFolder = $('#folderSel').value;
  $('#moveSel').value = currentFolder;
}

function closeModal() {
  $('#modal').classList.remove('open');
  state.current = null;
  state.cropBox = null;
  $('#modalImg').src = '';
  $('#cropRect').style.display = 'none';
  $('#cropOverlay').style.display = 'none';
}

// ---- Crop interactions ----
(function setupCrop() {
  const overlay = $('#cropOverlay');
  const rect = $('#cropRect');
  let startX, startY, dragging = false;

  function imgGeom() {
    const img = $('#modalImg');
    const wrap = overlay.parentElement;
    const wrapRect = wrap.getBoundingClientRect();
    const imgRect = img.getBoundingClientRect();
    return {
      // overlay is positioned inset:0 over wrap; coords of img inside overlay:
      offX: imgRect.left - wrapRect.left,
      offY: imgRect.top - wrapRect.top,
      w: imgRect.width, h: imgRect.height,
      natW: img.naturalWidth, natH: img.naturalHeight,
    };
  }

  overlay.addEventListener('mousedown', e => {
    if (e.button !== 0) return;
    const g = imgGeom();
    const ox = e.offsetX, oy = e.offsetY;
    // Only start if inside the image area
    if (ox < g.offX || ox > g.offX + g.w || oy < g.offY || oy > g.offY + g.h) return;
    dragging = true;
    startX = ox; startY = oy;
    rect.style.left = ox + 'px'; rect.style.top = oy + 'px';
    rect.style.width = '0px'; rect.style.height = '0px';
    rect.style.display = 'block';
    e.preventDefault();
  });

  overlay.addEventListener('mousemove', e => {
    if (!dragging) return;
    const g = imgGeom();
    const x = Math.max(g.offX, Math.min(e.offsetX, g.offX + g.w));
    const y = Math.max(g.offY, Math.min(e.offsetY, g.offY + g.h));
    const x1 = Math.min(startX, x), y1 = Math.min(startY, y);
    const x2 = Math.max(startX, x), y2 = Math.max(startY, y);
    rect.style.left = x1 + 'px'; rect.style.top = y1 + 'px';
    rect.style.width = (x2 - x1) + 'px'; rect.style.height = (y2 - y1) + 'px';
  });

  overlay.addEventListener('mouseup', e => {
    if (!dragging) return;
    dragging = false;
    const g = imgGeom();
    const r = rect.getBoundingClientRect();
    const wrapRect = overlay.parentElement.getBoundingClientRect();
    const x1_px = r.left - wrapRect.left - g.offX;
    const y1_px = r.top - wrapRect.top - g.offY;
    const x2_px = x1_px + r.width;
    const y2_px = y1_px + r.height;
    if (r.width < 5 || r.height < 5) {
      rect.style.display = 'none';
      state.cropBox = null;
      $('#cropReplaceBtn').disabled = true;
      $('#cropNewBtn').disabled = true;
      return;
    }
    const scaleX = g.natW / g.w;
    const scaleY = g.natH / g.h;
    state.cropBox = {
      left: Math.round(x1_px * scaleX),
      top: Math.round(y1_px * scaleY),
      right: Math.round(x2_px * scaleX),
      bottom: Math.round(y2_px * scaleY),
    };
    $('#cropCoords').textContent =
      `crop: (${state.cropBox.left}, ${state.cropBox.top}) → (${state.cropBox.right}, ${state.cropBox.bottom})` +
      `  size: ${state.cropBox.right - state.cropBox.left} × ${state.cropBox.bottom - state.cropBox.top}`;
    $('#cropReplaceBtn').disabled = false;
    $('#cropNewBtn').disabled = false;
  });

  $('#cropResetBtn').addEventListener('click', () => {
    rect.style.display = 'none';
    state.cropBox = null;
    $('#cropReplaceBtn').disabled = true;
    $('#cropNewBtn').disabled = true;
    const img = $('#modalImg');
    $('#cropCoords').textContent = `image: ${img.naturalWidth} × ${img.naturalHeight}`;
  });
})();

async function doCrop(mode) {
  if (!state.current || !state.cropBox) return;
  if (mode === 'replace' && !confirm('Replace the original file with the cropped image?')) return;
  try {
    const r = await api('/api/crop', { method: 'POST', body: {
      path: state.current.rel, ...state.cropBox, mode,
    }});
    toast(mode === 'replace' ? 'Cropped & replaced' : 'Saved cropped copy');
    closeModal();
    refresh();
  } catch (e) { toast(e.message, true); }
}

$('#cropReplaceBtn').addEventListener('click', () => doCrop('replace'));
$('#cropNewBtn').addEventListener('click', () => doCrop('new'));

$('#moveBtn').addEventListener('click', async () => {
  if (!state.current) return;
  const dst = $('#moveSel').value;
  try {
    await api('/api/move', { method: 'POST', body: { path: state.current.rel, folder: dst }});
    toast('Moved');
    closeModal();
    refresh();
  } catch (e) { toast(e.message, true); }
});

$('#hideBtn').addEventListener('click', async () => {
  if (!state.current) return;
  try {
    const r = await api('/api/hide', { method: 'POST', body: { path: state.current.rel }});
    toast(r.action === 'hide' ? 'Hidden' : 'Unhidden');
    closeModal();
    refresh();
  } catch (e) { toast(e.message, true); }
});

$('#deleteBtn').addEventListener('click', async () => {
  if (!state.current) return;
  if (!confirm(`Permanently delete ${state.current.name}?`)) return;
  try {
    await api('/api/delete', { method: 'POST', body: { path: state.current.rel }});
    toast('Deleted');
    closeModal();
    refresh();
  } catch (e) { toast(e.message, true); }
});

$('#closeBtn').addEventListener('click', closeModal);
$('#modal').addEventListener('click', e => { if (e.target.id === 'modal') closeModal(); });
document.addEventListener('keydown', e => {
  if (e.key === 'Escape') {
    if ($('#cmpOverlay').classList.contains('open')) cmpClose();
    else closeModal();
    return;
  }
  if ((e.key === 'ArrowLeft' || e.key === 'ArrowRight') && state.current) {
    const tag = (e.target.tagName || '').toLowerCase();
    if (tag === 'input' || tag === 'textarea' || tag === 'select') return;
    const idx = state.items.findIndex(it => it.rel === state.current.rel);
    if (idx === -1) return;
    const next = idx + (e.key === 'ArrowRight' ? 1 : -1);
    if (next < 0 || next >= state.items.length) return;
    e.preventDefault();
    openModal(state.items[next]);
  }
});

$('#folderSel').addEventListener('change', () => {
  state.selected.clear();
  state.lastClickedRel = null;
  loadList($('#folderSel').value).catch(e => toast(e.message, true));
});
$('#refresh').addEventListener('click', () => refresh());

$('#selectAllBtn').addEventListener('click', () => {
  if (state.selected.size === state.items.length) {
    state.selected.clear();
  } else {
    state.selected = new Set(state.items.map(it => it.rel));
  }
  renderGrid();
  updateSelectionUI();
});

$('#clearSelBtn').addEventListener('click', () => {
  state.selected.clear();
  state.lastClickedRel = null;
  renderGrid();
  updateSelectionUI();
});

$('#bulkHideBtn').addEventListener('click', async () => {
  const targets = state.items.filter(it => state.selected.has(it.rel));
  if (targets.length === 0) return;
  const btn = $('#bulkHideBtn');
  btn.disabled = true;
  const origLabel = btn.textContent;
  const verb = origLabel.startsWith('Unhide') ? 'Unhiding' : 'Hiding';
  let ok = 0, failed = 0, lastErr = '';
  for (let i = 0; i < targets.length; i++) {
    btn.textContent = `${verb} ${i + 1}/${targets.length}...`;
    try {
      await api('/api/hide', { method: 'POST', body: { path: targets[i].rel } });
      state.selected.delete(targets[i].rel);
      ok++;
    } catch (e) { failed++; lastErr = e.message; }
  }
  btn.textContent = origLabel;
  btn.disabled = false;
  state.lastClickedRel = null;
  state.selected.clear();
  const done = verb === 'Hiding' ? 'Hid' : 'Unhid';
  toast(failed ? `${done} ${ok}, ${failed} failed: ${lastErr}` : `${done} ${ok}`, failed > 0);
  refresh();
});

$('#bulkDeleteBtn').addEventListener('click', async () => {
  const targets = state.items.filter(it => state.selected.has(it.rel));
  if (targets.length === 0) return;
  if (!confirm(`Permanently delete ${targets.length} image${targets.length === 1 ? '' : 's'}? This cannot be undone.`)) return;
  const btn = $('#bulkDeleteBtn');
  btn.disabled = true;
  const origLabel = btn.textContent;
  let ok = 0, failed = 0, lastErr = '';
  for (let i = 0; i < targets.length; i++) {
    btn.textContent = `Deleting ${i + 1}/${targets.length}...`;
    try {
      await api('/api/delete', { method: 'POST', body: { path: targets[i].rel } });
      state.selected.delete(targets[i].rel);
      ok++;
    } catch (e) { failed++; lastErr = e.message; }
  }
  btn.textContent = origLabel;
  btn.disabled = false;
  state.lastClickedRel = null;
  state.selected.clear();
  toast(failed ? `Deleted ${ok}, ${failed} failed: ${lastErr}` : `Deleted ${ok}`, failed > 0);
  refresh();
});
// ---- Pixel compare: with exactly two images selected, render only the
// pixels they share — per-channel RGB diff within the tolerance keeps the
// first image's pixel, anything else goes transparent over the stage's
// checkerboard. Same tool as the generator UI's history-card ⧉ button. ----
let cmpData = null;  // {a, b: ImageData, w, h} while the overlay is open

function cmpClose() {
  $('#cmpOverlay').classList.remove('open');
  cmpData = null;
  const c = $('#cmpCanvas');
  c.width = c.height = 0;  // frees the decoded bitmap
}
$('#cmpClose').addEventListener('click', cmpClose);
$('#cmpOverlay').addEventListener('click', e => { if (e.target.id === 'cmpOverlay') cmpClose(); });

function cmpRender() {
  if (!cmpData) return;
  const tol = parseInt($('#cmpTol').value, 10);
  $('#cmpTolVal').textContent = tol;
  $('#cmpTolEcho').textContent = tol;
  const { a, b, w, h } = cmpData;
  const out = new ImageData(w, h);
  const pa = a.data, pb = b.data, po = out.data;
  let same = 0;
  for (let i = 0; i < pa.length; i += 4) {
    if (Math.abs(pa[i] - pb[i]) <= tol &&
        Math.abs(pa[i+1] - pb[i+1]) <= tol &&
        Math.abs(pa[i+2] - pb[i+2]) <= tol) {
      po[i] = pa[i]; po[i+1] = pa[i+1]; po[i+2] = pa[i+2]; po[i+3] = 255;
      same++;
    }
  }
  $('#cmpCanvas').getContext('2d').putImageData(out, 0, 0);
  $('#cmpStat').textContent = `${w}×${h} — ${(100*same/(w*h)).toFixed(1)}% of pixels match`;
}
$('#cmpTol').addEventListener('input', () => {
  if (cmpRender._raf) cancelAnimationFrame(cmpRender._raf);
  cmpRender._raf = requestAnimationFrame(cmpRender);
});

function cmpLoadPixels(rel) {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => {
      const c = document.createElement('canvas');
      c.width = img.naturalWidth;
      c.height = img.naturalHeight;
      const ctx = c.getContext('2d', { willReadFrequently: true });
      ctx.drawImage(img, 0, 0);
      resolve(ctx.getImageData(0, 0, c.width, c.height));
    };
    img.onerror = () => reject(new Error('failed to load ' + rel));
    img.src = imgUrl(rel);
  });
}

$('#cmpBtn').addEventListener('click', async () => {
  // Grid order, so "first image" is deterministic regardless of click order.
  const rels = state.items.filter(it => state.selected.has(it.rel)).map(it => it.rel);
  if (rels.length !== 2) return;
  const btn = $('#cmpBtn');
  btn.disabled = true;
  let a, b;
  try {
    [a, b] = await Promise.all([cmpLoadPixels(rels[0]), cmpLoadPixels(rels[1])]);
  } catch (e) { btn.disabled = false; toast(e.message, true); return; }
  btn.disabled = false;
  if (a.width !== b.width || a.height !== b.height) {
    toast(`Compare needs the same resolution — got ${a.width}×${a.height} and ${b.width}×${b.height}`, true);
    return;
  }
  cmpData = { a, b, w: a.width, h: a.height };
  const c = $('#cmpCanvas');
  c.width = a.width;
  c.height = a.height;
  $('#cmpOverlay').classList.add('open');
  cmpRender();
});

$('#apiKey').addEventListener('change', () => {
  localStorage.setItem('im_api_key', apiKey());
  refresh();
});

// ---- Slideshow -------------------------------------------------------
// Review a whole folder at full resolution and triage it in place. Quality is
// the point, so three things are deliberate:
//
//  * the source is /api/image (the original file), never /api/thumb, which
//    caps at 512px and re-encodes to JPEG q82;
//  * the next slides are fetched ahead of time and a slide is only shown once
//    its bytes are fully in, so none ever paints in progressively;
//    HTMLImageElement.decode() would be the better gate - it resolves only
//    when the frame is ready to paint - but it never settles here (attached or
//    detached, Chrome leaves the promise pending), so the load event is what
//    the show waits on;
//  * the dwell timer starts when the image appears, not when it was
//    requested, so a slow decode shortens nothing.
//
// The decoded-bitmap cache is bounded: a 4MP RGBA frame is ~16MB, so holding a
// whole folder would be hundreds of MB for no benefit.
const SHOW_PRELOAD = 2;             // slides ahead/behind to decode
const SHOW_CACHE_KEEP = 5;          // decoded images retained around the cursor
const SHOW_IDLE_MS = 2600;          // hide the chrome after this much stillness
const SHOW_ARM_MS = 3000;           // how long a primed delete stays primed

const show = {
  list: [], i: 0, playing: false, intervalMs: 5000,
  timer: null, idleTimer: null, armTimer: null,
  cache: new Map(),                 // rel -> HTMLImageElement
  front: 1,                         // which layer is currently opaque
  armed: false, changed: false, token: 0,
};

function showOpen(items, startIndex) {
  if (!items.length) { toast('Nothing to show', true); return; }
  show.list = items.slice();
  show.i = Math.max(0, Math.min(startIndex || 0, show.list.length - 1));
  show.changed = false;
  show.cache.clear();
  show.front = 1;
  $('#showLayer0').classList.remove('front');
  $('#showLayer1').classList.remove('front');
  show.intervalMs = parseInt($('#showEvery').value, 10) || 5000;
  $('#show').classList.add('open');
  showGoto(show.i);
  showNudge();
}

function showClose() {
  showPause();
  clearTimeout(show.idleTimer);
  showDisarm();
  show.token++;                     // abandon any decode still in flight
  $('#show').classList.remove('open', 'idle');
  $('#showLayer0').querySelector('img').src = '';
  $('#showLayer1').querySelector('img').src = '';
  show.cache.clear();
  if (document.fullscreenElement) document.exitFullscreen().catch(() => {});
  // Deletes and hides happened server-side; the grid behind is now stale.
  if (show.changed) refresh();
}

function showImgFor(rel) {
  let img = show.cache.get(rel);
  if (!img) {
    img = new Image();
    img.src = imgUrl(rel);          // full-resolution original
    show.cache.set(rel, img);
  }
  return img;
}

// Resolves once the image's bytes are in and its dimensions are known.
// Deliberately not img.decode(): that is the better primitive on paper, but it
// does not settle in this browser for either an attached or a detached <img>,
// which left every slide blank. The load event is reliable, and a fully
// fetched image painting into a layer that is fading in over 160ms does not
// show the progressive paint that the decode gate was there to prevent.
function showReady(img) {
  if (img.complete) {
    return img.naturalWidth ? Promise.resolve() : Promise.reject(new Error('not an image'));
  }
  return new Promise((resolve, reject) => {
    img.addEventListener('load', () => resolve(), { once: true });
    img.addEventListener('error', () => reject(new Error('load failed')), { once: true });
  });
}

function showPreloadAround(i) {
  for (let d = 1; d <= SHOW_PRELOAD; d++) {
    for (const j of [i + d, i - d]) {
      if (j >= 0 && j < show.list.length) showImgFor(show.list[j].rel);
    }
  }
  // Evict anything far from the cursor so the cache stays a window, not a log.
  if (show.cache.size > SHOW_CACHE_KEEP) {
    const keep = new Set();
    for (let d = -SHOW_PRELOAD; d <= SHOW_PRELOAD; d++) {
      const it = show.list[i + d];
      if (it) keep.add(it.rel);
    }
    for (const rel of Array.from(show.cache.keys())) {
      if (!keep.has(rel)) show.cache.delete(rel);
    }
  }
}

async function showGoto(i, keepPlaying) {
  if (!show.list.length) { showClose(); return; }
  show.i = (i + show.list.length) % show.list.length;
  const it = show.list[show.i];
  const token = ++show.token;
  showDisarm();
  clearTimeout(show.timer);
  showBarStop();

  $('#showCount').textContent = (show.i + 1) + ' / ' + show.list.length;
  $('#showName').textContent = it.name;
  showRenderInfo(it, null);

  const img = showImgFor(it.rel);
  try { await showReady(img); }
  catch (e) {
    // A load failure is the image itself being unreadable (or deleted from
    // under us); skipping beats a black slide and a stalled show.
    if (token !== show.token) return;
    toast('Could not display ' + it.name, true);
    if (show.list.length === 1) { showClose(); return; }
    show.list.splice(show.i, 1);
    show.cache.delete(it.rel);
    showGoto(show.i, keepPlaying);
    return;
  }
  if (token !== show.token) return;  // a newer navigation won

  const backEl = show.front === 0 ? $('#showLayer1') : $('#showLayer0');
  const frontEl = show.front === 0 ? $('#showLayer0') : $('#showLayer1');
  const backImg = backEl.querySelector('img');
  backImg.src = img.src;
  backImg.width = img.naturalWidth;
  backImg.height = img.naturalHeight;
  backEl.classList.add('front');
  frontEl.classList.remove('front');
  show.front = show.front === 0 ? 1 : 0;
  backEl.scrollTop = 0; backEl.scrollLeft = 0;

  showRenderInfo(it, img);
  showPreloadAround(show.i);
  // Dwell starts now - after the picture is actually up.
  if (show.playing && keepPlaying !== false) showSchedule();
}

function showSchedule() {
  clearTimeout(show.timer);
  showBarRun();
  show.timer = setTimeout(() => showGoto(show.i + 1), show.intervalMs);
}

function showBarRun() {
  const bar = $('#showBar');
  bar.classList.remove('run');
  void bar.offsetWidth;             // reflow, so the animation restarts
  bar.style.animationDuration = (show.intervalMs / 1000) + 's';
  bar.classList.add('run');
}

function showBarStop() {
  const bar = $('#showBar');
  bar.classList.remove('run');
  bar.style.width = '0';
}

function showPlay() {
  show.playing = true;
  $('#showPlay').innerHTML = '&#10073;&#10073;';
  $('#showPlay').classList.add('on');
  showSchedule();
}

function showPause() {
  show.playing = false;
  clearTimeout(show.timer);
  showBarStop();
  $('#showPlay').innerHTML = '&#9654;';
  $('#showPlay').classList.remove('on');
}

function showTogglePlay() { show.playing ? showPause() : showPlay(); }

function showRenderInfo(it, img) {
  const dims = img ? (img.naturalWidth + ' x ' + img.naturalHeight + ' px') : '...';
  const mp = img ? ((img.naturalWidth * img.naturalHeight) / 1e6).toFixed(1) + ' MP' : '';
  let html = '<dl>' +
    '<dt>File</dt><dd>' + esc(it.name) + '</dd>' +
    '<dt>Folder</dt><dd>' + esc(it.rel.split('/').slice(0, -1).join('/') || '(root)') + '</dd>' +
    '<dt>Size</dt><dd>' + fmtBytes(it.size) + '</dd>' +
    '<dt>Pixels</dt><dd>' + dims + (mp ? ' &middot; ' + mp : '') + '</dd>' +
    '<dt>Modified</dt><dd>' + fmtDate(it.mtime) + '</dd>' +
    '</dl>';
  if (it.prompt) html += '<div class="sp">' + esc(it.prompt) + '</div>';
  $('#showInfo').innerHTML = html;
}

function esc(t) {
  return String(t == null ? '' : t).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
}

// Remove the current slide from the show after it has been hidden or deleted,
// and keep the grid's own state in step so closing is not a surprise.
function showDropCurrent() {
  const gone = show.list.splice(show.i, 1)[0];
  if (gone) {
    show.cache.delete(gone.rel);
    state.items = state.items.filter(x => x.rel !== gone.rel);
    state.selected.delete(gone.rel);
  }
  show.changed = true;
  if (!show.list.length) { toast('That was the last one'); showClose(); return; }
  if (show.i >= show.list.length) show.i = 0;
  showGoto(show.i);
}

async function showHideCurrent() {
  const it = show.list[show.i];
  if (!it) return;
  const wasPlaying = show.playing;
  showPause();
  try {
    const r = await api('/api/hide', { method: 'POST', body: { path: it.rel } });
    toast(r.action === 'unhide' ? 'Unhid ' + it.name : 'Hid ' + it.name);
    showDropCurrent();
    if (wasPlaying && show.list.length) showPlay();
  } catch (e) { toast(e.message, true); }
}

// Delete is irreversible and a slideshow is a fast-moving context, so it takes
// two presses rather than a confirm dialog: a modal would stall the show and
// steal the keyboard, and confirm-every-time is what trains people to click
// through without reading.
function showArmDelete() {
  show.armed = true;
  const b = $('#showDeleteBtn');
  b.classList.add('armed');
  b.textContent = 'Press again';
  clearTimeout(show.armTimer);
  show.armTimer = setTimeout(showDisarm, SHOW_ARM_MS);
}

function showDisarm() {
  if (!show.armed) return;
  show.armed = false;
  clearTimeout(show.armTimer);
  const b = $('#showDeleteBtn');
  b.classList.remove('armed');
  b.textContent = 'Delete';
}

async function showDeleteCurrent() {
  if (!show.armed) { showPause(); showArmDelete(); return; }
  showDisarm();
  const it = show.list[show.i];
  if (!it) return;
  try {
    await api('/api/delete', { method: 'POST', body: { path: it.rel } });
    toast('Deleted ' + it.name);
    showDropCurrent();
  } catch (e) { toast(e.message, true); }
}

function showToggleActual() {
  const on = $('#show').classList.toggle('actual');
  $('#showActual').textContent = on ? 'Actual' : 'Fit';
  $('#showActual').classList.toggle('on', on);
}

function showToggleFs() {
  if (document.fullscreenElement) document.exitFullscreen().catch(() => {});
  else $('#show').requestFullscreen().catch(() => toast('Full screen refused', true));
}

// Any input brings the chrome back and restarts the idle countdown.
function showNudge() {
  const el = $('#show');
  el.classList.remove('idle');
  clearTimeout(show.idleTimer);
  show.idleTimer = setTimeout(() => {
    if (el.classList.contains('open')) el.classList.add('idle');
  }, SHOW_IDLE_MS);
}

function showItemsForStart() {
  // Mirrors the bulk buttons: a selection narrows the show to it.
  const sel = state.items.filter(it => state.selected.has(it.rel));
  return sel.length ? sel : state.items;
}

$('#showBtn').addEventListener('click', () => showOpen(showItemsForStart(), 0));
$('#showFromHereBtn').addEventListener('click', () => {
  const cur = state.current;
  closeModal();
  const items = state.items;
  const idx = cur ? items.findIndex(it => it.rel === cur.rel) : 0;
  showOpen(items, idx < 0 ? 0 : idx);
});
$('#showClose').addEventListener('click', showClose);
$('#showPrev').addEventListener('click', () => { showPause(); showGoto(show.i - 1); });
$('#showNext').addEventListener('click', () => { showPause(); showGoto(show.i + 1); });
$('#showPlay').addEventListener('click', showTogglePlay);
$('#showHideBtn').addEventListener('click', showHideCurrent);
$('#showDeleteBtn').addEventListener('click', showDeleteCurrent);
$('#showActual').addEventListener('click', showToggleActual);
$('#showFs').addEventListener('click', showToggleFs);
$('#showInfoBtn').addEventListener('click', () => {
  $('#showInfoBtn').classList.toggle('on', $('#showInfo').classList.toggle('on'));
});
$('#showKeysBtn').addEventListener('click', () => {
  $('#showKeysBtn').classList.toggle('on', $('#showKeys').classList.toggle('on'));
});
$('#showEvery').addEventListener('change', () => {
  show.intervalMs = parseInt($('#showEvery').value, 10) || 5000;
  if (show.playing) showSchedule();
});
$('#showZonePrev').addEventListener('click', () => { showPause(); showGoto(show.i - 1); });
$('#showZoneNext').addEventListener('click', () => { showPause(); showGoto(show.i + 1); });
$('#show').addEventListener('mousemove', showNudge);

// Capture phase: while the show is open it owns the keyboard, so the grid's
// own Esc/arrow handling never sees these keys and cannot also act on them.
document.addEventListener('keydown', e => {
  if (!$('#show').classList.contains('open')) return;
  const tag = (e.target.tagName || '').toLowerCase();
  if (tag === 'input' || tag === 'textarea' || tag === 'select') return;
  let handled = true;
  switch (e.key) {
    case 'Escape':    showClose(); break;
    case 'ArrowRight':
    case 'PageDown':  showPause(); showGoto(show.i + 1); break;
    case 'ArrowLeft':
    case 'PageUp':    showPause(); showGoto(show.i - 1); break;
    case 'Home':      showPause(); showGoto(0); break;
    case 'End':       showPause(); showGoto(show.list.length - 1); break;
    case ' ':         showTogglePlay(); break;
    case 'h': case 'H': showHideCurrent(); break;
    case 'x': case 'X': case 'Delete': showDeleteCurrent(); break;
    case 'a': case 'A': showToggleActual(); break;
    case 'f': case 'F': showToggleFs(); break;
    case 'i': case 'I': $('#showInfoBtn').click(); break;
    case '?':           $('#showKeysBtn').click(); break;
    default: handled = false;
  }
  if (handled) { e.preventDefault(); e.stopPropagation(); showNudge(); }
}, true);

async function refresh() {
  try {
    await loadFolders();
    await loadList($('#folderSel').value || '');
  } catch (e) { toast(e.message, true); }
}

// On startup
$('#apiKey').value = localStorage.getItem('im_api_key') || '';
refresh();
</script>
</body>
</html>
"""


@app.route("/")
def index():
    from flask import Response
    return Response(HTML_PAGE, mimetype="text/html")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FLUX Image Manager")
    parser.add_argument("--port", type=int, default=int(os.environ.get("IMAGE_MANAGER_PORT", 2223)))
    parser.add_argument("--host", default="0.0.0.0")
    args = parser.parse_args()

    if not API_KEY:
        print("ERROR: FLUX_API_KEY is not set. Put it in .env or export it before starting.")
        import sys
        sys.exit(1)

    migrate_legacy_hide()
    print(f"Image Manager serving {ROOT}")
    print(f"Listening on http://{args.host}:{args.port}  (hostname: {socket.gethostname()})")
    app.run(host=args.host, port=args.port, threaded=True)
