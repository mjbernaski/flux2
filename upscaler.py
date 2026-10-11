"""Real-ESRGAN upscaling for the image manager, on the CPU by default.

Why CPU by default: this runs inside image_manager.py, which is a *separate
process* from the generation server, and the whole point is that upscaling an
image must not cost the diffusion pipeline anything. Taking VRAM on a card the
generator is resident on is exactly the condition that triggers the WDDM
sysmem-fallback stall described in CLAUDE.md — the driver pages the generator's
weights out and every diffusion step runs 8-200x slow with nothing in the log.
A 1MP image at 4x measured 2.7s on CPU with the compact model (12-core Ryzen),
which is cheap enough that there is no reason to reach for the GPU.

`device="cuda"` is still offered because it is ~3x faster for the compact model
and ~26x for the big one, and the card is free when nothing is generating. It
is opt-in per request, never the default.

Models are plain .pth files under models/upscale/, loaded through spandrel
rather than the `realesrgan` pip package: that package depends on basicsr,
which is unmaintained and fails to import against current torchvision.
Everything here imports lazily, so the image manager still starts and serves
normally when spandrel or the weights are absent — `available()` just comes
back empty and the UI hides the control.
"""

import os
import threading

MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "upscale")

# Measured on a 992x992 source, 12 threads: compact 2.7s / ESRGAN 60.8s on CPU,
# 0.9s / 2.3s on CUDA. The compact model is the default for that reason — it is
# not a quality compromise so much as the only one that feels interactive.
MODELS = {
    "realesr-general-x4v3": {
        "file": "realesr-general-x4v3.pth",
        "label": "Real-ESRGAN general x4 (fast)",
        "note": "1.2M params — ~3s per megapixel on CPU",
    },
    "RealESRGAN_x4plus": {
        "file": "RealESRGAN_x4plus.pth",
        "label": "Real-ESRGAN x4plus (slow, more detail)",
        "note": "16.7M params — ~60s per megapixel on CPU, seconds on CUDA",
    },
}

DEFAULT_MODEL = "realesr-general-x4v3"

# Tile size in *source* pixels. 192 with 16px of context on each side keeps peak
# memory flat regardless of image size, which matters more here than it looks:
# a 4x pass on a 1MP source produces a 16MP result, and doing it in one piece
# would allocate the whole thing as activations.
TILE = 192
OVERLAP = 16

_loaded = {}          # (name, device) -> spandrel descriptor
_load_lock = threading.Lock()
# One upscale at a time. Each pass already saturates every core, so overlapping
# requests would only trade throughput for latency on all of them.
_run_lock = threading.Lock()


def _model_path(name):
    spec = MODELS.get(name)
    return os.path.join(MODELS_DIR, spec["file"]) if spec else None


def available():
    """Models whose weights are actually on disk, newest API first.

    Returns [] when spandrel is not installed, so callers can treat "no
    upscaler" and "no weights" the same way.
    """
    try:
        import spandrel  # noqa: F401
    except Exception:
        return []
    out = []
    for name, spec in MODELS.items():
        path = _model_path(name)
        if path and os.path.isfile(path):
            out.append({"name": name, "label": spec["label"], "note": spec["note"]})
    return out


def cuda_available():
    try:
        import torch
        return bool(torch.cuda.is_available())
    except Exception:
        return False


def _load(name, device):
    key = (name, device)
    with _load_lock:
        if key in _loaded:
            return _loaded[key]
        import torch
        from spandrel import ModelLoader
        path = _model_path(name)
        if not path or not os.path.isfile(path):
            raise FileNotFoundError(f"no weights for {name} at {path}")
        model = ModelLoader().load_from_file(path)
        model.eval()
        model.to(torch.device(device))
        _loaded[key] = model
        return model


def _to_tensor(im):
    import numpy as np
    import torch
    a = np.asarray(im, dtype="float32") / 255.0
    return torch.from_numpy(a).permute(2, 0, 1).unsqueeze(0)


def _to_pil(t):
    import numpy as np
    from PIL import Image
    a = (t.squeeze(0).clamp(0, 1).permute(1, 2, 0).numpy() * 255.0).round().astype("uint8")
    return Image.fromarray(a)


def upscale(image, name=DEFAULT_MODEL, device="cpu", tile=TILE, overlap=OVERLAP):
    """Upscale a PIL image. Returns (new image, scale factor).

    Tiles overlap on input and are cropped back on output, so each tile is
    produced with `overlap` pixels of real context on every side it has one —
    which is what stops tile edges showing as seams. Nothing is blended, so
    there is no double-exposure where tiles meet.

    Alpha is dropped: these models are RGB-only, and silently inventing an
    alpha channel for the result would be worse than saying so here.
    """
    import torch
    from PIL import Image

    if name not in MODELS:
        raise ValueError(f"unknown upscaler {name!r}")
    if device not in ("cpu", "cuda"):
        raise ValueError(f"unknown device {device!r}")
    if device == "cuda" and not cuda_available():
        raise RuntimeError("CUDA is not available in this process")

    src = image.convert("RGB")
    with _run_lock:
        model = _load(name, device)
        scale = model.scale
        w, h = src.size
        out = Image.new("RGB", (w * scale, h * scale))
        for y in range(0, h, tile):
            for x in range(0, w, tile):
                x0, y0 = max(0, x - overlap), max(0, y - overlap)
                x1, y1 = min(w, x + tile + overlap), min(h, y + tile + overlap)
                patch = _to_tensor(src.crop((x0, y0, x1, y1)))
                with torch.no_grad():
                    res = model(patch.to(torch.device(device)))
                res = _to_pil(res.detach().cpu())
                # Cut the context back off, in output coordinates.
                cx, cy = (x - x0) * scale, (y - y0) * scale
                cw = (min(w, x + tile) - x) * scale
                ch = (min(h, y + tile) - y) * scale
                out.paste(res.crop((cx, cy, cx + cw, cy + ch)), (x * scale, y * scale))
        if device == "cuda":
            # Hand the memory straight back: the generator on the other port is
            # the one that actually needs it.
            torch.cuda.empty_cache()
    return out, scale
