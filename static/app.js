// Model readiness overlay: poll /ready until the model is loaded, then hide.
(function() {
    const overlay = document.getElementById('loadingOverlay');
    const statusEl = document.getElementById('loadingStatus');
    const elapsedEl = document.getElementById('loadingElapsed');
    const titleEl = document.getElementById('loadingTitle');
    if (!overlay) return;
    let readyPollTimer = null;

    async function checkReady() {
        try {
            const res = await fetch('/ready', { cache: 'no-store' });
            const data = await res.json();
            if (data.ready) {
                overlay.style.display = 'none';
                if (readyPollTimer) { clearInterval(readyPollTimer); readyPollTimer = null; }
                return;
            }
            if (data.error) {
                overlay.classList.add('error');
                titleEl.textContent = 'Model load failed';
                statusEl.textContent = data.error;
                if (readyPollTimer) { clearInterval(readyPollTimer); readyPollTimer = null; }
                return;
            }
            if (data.status) statusEl.textContent = data.status;
            if (typeof data.elapsed_s === 'number') {
                const s = Math.round(data.elapsed_s);
                elapsedEl.textContent = s < 60 ? `${s}s` : `${Math.floor(s/60)}m ${s%60}s`;
            }
        } catch (err) {
            statusEl.textContent = 'waiting for server…';
        }
    }
    checkReady();
    readyPollTimer = setInterval(checkReady, 1500);
})();

// Security helpers
function getAuthHeaders(extraHeaders = {}) {
    const apiKey = localStorage.getItem('flux_api_key');
    const headers = { ...extraHeaders };
    if (apiKey) {
        headers['X-API-Key'] = apiKey;
    }
    return headers;
}

// Initialize API key input
const apiKeyInput = document.getElementById('apiKeyInput');
if (apiKeyInput) {
    apiKeyInput.value = localStorage.getItem('flux_api_key') || '';
}

// Run model-info fetch first, before any other code that might throw (so UI always updates)
(function() {
    var el = document.getElementById('modelInfo');
    if (!el) return;
    var timeout = setTimeout(function() {
        if (el.textContent === 'Loading model info...') el.textContent = 'Model info unavailable (use server URL, e.g. http://localhost:2222)';
    }, 5000);
    var ac = new AbortController();
    setTimeout(function() { ac.abort(); }, 8000);
    fetch('/model-info', { 
        signal: ac.signal,
        headers: getAuthHeaders()
    })
        .then(function(r) { if (!r.ok) throw new Error(r.status); return r.json(); })
        .then(function(data) {
            clearTimeout(timeout);
            // Model name is not shown on the page, but hovering the top-left
            // corner reveals it; the fetched info also drives the capability
            // toggles below.
            var mh = document.getElementById('modelHoverName');
            if (mh) mh.textContent = data.description || data.model || '';
            var h = document.getElementById('hostname'); if (h) h.textContent = data.hostname || '';
            var v = document.getElementById('version'); if (v) v.textContent = 'v' + (data.version || '');
            window.__fluxVersion = data.flux_version || null;
            window.__inpaintCapable = !!data.inpaint;
            if (typeof refreshInpaintAvailability === 'function') refreshInpaintAvailability();
            if (data.schnell) {
                var s = document.getElementById('steps'); if (s) { s.disabled = true; s.title = 'Schnell uses fixed 4 steps'; }
                var g = document.getElementById('guidance'); if (g) { g.disabled = true; g.title = 'Schnell requires guidance_scale=0'; }
            }
            if (data.negative_prompt) {
                var np = document.getElementById('negativePromptGroup');
                if (np) np.style.display = 'block';
            }
        })
        .catch(function() { clearTimeout(timeout); el.textContent = 'Model info unavailable (use server URL, e.g. http://localhost:2222)'; });
})();

// ---- Form-state carryover across a model switch ----
// The switch ends in a full page reload (so /model-info re-fetches and the
// capability toggles match the new backend), which would otherwise wipe the
// prompt, reference images and every setting. Snapshot the form to
// sessionStorage before reloading; the restore block at the bottom of this
// file puts it back on the next load.
const SWITCH_STATE_KEY = 'flux_switch_state';

function saveSwitchState() {
    const val = function(id) { const el = document.getElementById(id); return el ? el.value : null; };
    const chk = function(id) { const el = document.getElementById(id); return el ? el.checked : null; };
    const state = {
        prompt: val('prompt'),
        negativePrompt: val('negativePrompt'),
        orientation: val('orientation'),
        size: val('size'),
        steps: val('steps'),
        seed: val('seed'),
        guidance: val('guidance'),
        batch: val('batch'),
        evolveCount: val('evolveCount'),
        allOrientations: chk('allOrientations'),
        spectrumGrid: chk('spectrumGrid'),
        spectrumSameSeed: chk('spectrumSameSeed'),
        showPreview: chk('showPreview'),
        savePreviews: chk('savePreviews'),
        boostThink: chk('boostThink'),
        describeThink: chk('describeThink'),
        strength: strengthSlider ? strengthSlider.value : null,
        aspectMode: aspectModeEl ? aspectModeEl.value : null,
        cells: Array.from(selectedCells),
        refImages: currentInputImages.slice(0, MAX_REFERENCE_IMAGES)
    };
    try {
        sessionStorage.setItem(SWITCH_STATE_KEY, JSON.stringify(state));
    } catch (err) {
        // Reference data URLs can blow the sessionStorage quota; keep at
        // least the text parameters.
        state.refImages = [];
        try { sessionStorage.setItem(SWITCH_STATE_KEY, JSON.stringify(state)); } catch (err2) {}
    }
}

// After a model switch: show the loading overlay and reload the page once the
// server has gone down and come back up ready with the new model. (A reload
// re-fetches /model-info so all capability toggles match the new backend.)
function watchServerRestart(label, progress) {
    saveSwitchState();
    // The restarting server can't answer /status; this watcher's own /ready
    // poll takes over until the page reloads.
    pollStopped = true;
    if (pollTimer) { clearTimeout(pollTimer); pollTimer = null; }
    const overlay = document.getElementById('loadingOverlay');
    const statusEl = document.getElementById('loadingStatus');
    const elapsedEl = document.getElementById('loadingElapsed');
    const titleEl = document.getElementById('loadingTitle');
    const progressEl = document.getElementById('loadingProgress');
    if (!overlay) { setTimeout(() => location.reload(), 5000); return; }
    overlay.classList.remove('error');
    if (titleEl) titleEl.textContent = 'Switching to ' + label + '…';
    if (progressEl) {
        progressEl.textContent = progress || '';
        progressEl.style.display = progress ? '' : 'none';
    }
    if (statusEl) statusEl.textContent = 'restarting server';
    if (elapsedEl) elapsedEl.textContent = '0s';
    overlay.style.display = 'flex';

    const t0 = Date.now();
    let wentDown = false;  // don't reload until the old server has actually exited
    setInterval(async function() {
        if (elapsedEl) {
            const s = Math.round((Date.now() - t0) / 1000);
            elapsedEl.textContent = s < 60 ? `${s}s` : `${Math.floor(s / 60)}m ${s % 60}s`;
        }
        try {
            const res = await fetch('/ready', { cache: 'no-store' });
            const data = await res.json();
            if (!data.ready) {
                wentDown = true;
                if (data.error) {
                    overlay.classList.add('error');
                    if (titleEl) titleEl.textContent = 'Model load failed';
                    if (statusEl) statusEl.textContent = data.error;
                } else if (statusEl) {
                    statusEl.textContent = data.status || 'loading model';
                }
            } else if (wentDown) {
                location.reload();
            }
        } catch (err) {
            wentDown = true;
            if (statusEl) statusEl.textContent = 'waiting for server…';
        }
    }, 1500);
}

// ---- Model switcher ----
// Populated from /configs (the run_server.sh menu). Picking a different
// config POSTs /switch-model: the server restarts under the supervisor with
// the new model's flags and the page reloads when it's back up.
(function() {
    const wrap = document.getElementById('modelSwitch');
    const select = document.getElementById('modelSelect');
    if (!wrap || !select) return;
    let currentConfig = null;

    fetch('/configs', { headers: getAuthHeaders() })
        .then(function(r) { if (!r.ok) throw new Error(r.status); return r.json(); })
        .then(function(data) {
            if (!data.switchable || !data.configs) return;
            currentConfig = data.current;
            data.configs.forEach(function(c) {
                const opt = document.createElement('option');
                opt.value = c.id;
                opt.textContent = c.id + ' — ' + c.label;
                if (c.id === data.current) opt.selected = true;
                select.appendChild(opt);
            });
            wrap.style.display = 'flex';
        })
        .catch(function() {});  // no API key yet or older server — keep hidden

    select.addEventListener('change', async function() {
        const target = parseInt(select.value, 10);
        if (!target || target === currentConfig) return;
        const label = select.options[select.selectedIndex].textContent;
        if (!confirm('Switch model to:\n\n' + label + '\n\nThe server restarts and loads the new model (this can take a while).')) {
            select.value = String(currentConfig);
            return;
        }
        select.disabled = true;
        try {
            const res = await fetch('/switch-model', {
                method: 'POST',
                headers: getAuthHeaders({ 'Content-Type': 'application/json' }),
                body: JSON.stringify({ config: target })
            });
            const data = await res.json().catch(function() { return {}; });
            if (!res.ok || !data.success) throw new Error(data.error || `HTTP ${res.status}`);
            watchServerRestart(data.switching_to || label);
        } catch (err) {
            alert('Model switch failed: ' + err.message);
            select.value = String(currentConfig);
            select.disabled = false;
        }
    });
})();

const form = document.getElementById('generateForm');
const submitBtn = document.getElementById('submitBtn');
const status = document.getElementById('status');
const statusText = document.getElementById('statusText');
const result = document.getElementById('result');
const imageGrid = document.getElementById('imageGrid');
const generationInfo = document.getElementById('generationInfo');
const resultCount = document.getElementById('resultCount');

// The result grid sits in a collapsed-by-default <details>; the summary's
// count is the only signal of what's inside, so refresh it on every change.
function updateResultCount() {
    const count = imageGrid ? imageGrid.children.length : 0;
    if (resultCount) resultCount.textContent = count ? `(${count})` : '';
    // Looked up here rather than closed over: this runs before the button's
    // own const is initialized later in the file.
    const inlineClear = document.getElementById('clearRecentInline');
    if (inlineClear) inlineClear.style.display = count ? '' : 'none';
}

const uploadArea = document.getElementById('uploadArea');
const inputImage = document.getElementById('inputImage');
const uploadPlaceholder = document.getElementById('uploadPlaceholder');
const refThumbs = document.getElementById('refThumbs');
const multiRefHint = document.getElementById('multiRefHint');

const MAX_REFERENCE_IMAGES = 3;
// Data URLs of the uploaded references, in order. The first is the primary
// (drives output aspect ratio and inpainting). currentInputImage mirrors the
// primary for the inpaint code paths, which only ever work on one image.
let currentInputImages = [];
let currentInputImage = null;
const strengthControl = document.getElementById('strengthControl');
const aspectModeControl = document.getElementById('aspectModeControl');
const aspectModeEl = document.getElementById('aspectMode');
const strengthSlider = document.getElementById('strength');
const strengthValue = document.getElementById('strengthValue');

let knownImageFilenames = new Set();
let lastPreviewStep = -1;

// ---- Fullscreen "watch" overlay ----
const watchOverlay = document.getElementById('watchOverlay');
const watchBtnEl = document.getElementById('watchBtn');
const watchCloseEl = document.getElementById('watchClose');
let watchOpen = false;
let watchLastPreviewTs = -1;

function openWatch() {
    if (!watchOverlay) return;
    watchOpen = true;
    watchLastPreviewTs = -1;  // force preview refresh on next poll
    watchOverlay.classList.add('visible');
    if (watchOverlay.requestFullscreen) {
        watchOverlay.requestFullscreen().catch(function() {});
    }
}
function closeWatch() {
    if (!watchOverlay) return;
    watchOpen = false;
    watchOverlay.classList.remove('visible');
    if (document.fullscreenElement) {
        document.exitFullscreen().catch(function() {});
    }
}
// Keeps the overlay's image, progress bar and caption in sync with the
// running job. `running` is null when nothing is generating.
function updateWatchOverlay(running, pct, runBatch, stepInfo) {
    if (!watchOpen || !watchOverlay) return;
    const wimg = document.getElementById('watchImg');
    const wwait = document.getElementById('watchWaiting');
    const wprog = document.getElementById('watchProgress');
    const wtext = document.getElementById('watchText');

    if (!running) {
        if (wprog) wprog.style.width = '100%';
        if (wtext) wtext.textContent = 'Done';
        return;
    }

    if (running.preview && running.preview_ts && running.preview_ts !== watchLastPreviewTs) {
        if (wimg) {
            wimg.src = `/images/${running.preview}?t=${running.preview_ts}`;
            wimg.classList.remove('hidden');
        }
        if (wwait) wwait.style.display = 'none';
        watchLastPreviewTs = running.preview_ts;
    }
    if (wimg && wimg.classList.contains('hidden') && wwait) {
        wwait.style.display = 'block';
        wwait.textContent = (running.step > 0)
            ? 'Generating (live preview off for this job)…'
            : 'Waiting for first preview…';
    }

    if (wprog) wprog.style.width = `${pct || 0}%`;
    if (wtext) {
        const batchInfo = (runBatch > 1) ? `image ${running.current} / ${runBatch} · ` : '';
        wtext.textContent = `${batchInfo}${pct || 0}% complete${stepInfo || ''}`;
    }
}

if (watchBtnEl) watchBtnEl.addEventListener('click', openWatch);
if (watchCloseEl) watchCloseEl.addEventListener('click', closeWatch);
document.addEventListener('keydown', function(e) {
    // !lbOpen: when the lightbox is up, Escape must close only it.
    if (e.key === 'Escape' && watchOpen && !lbOpen) closeWatch();
});
// If the user leaves browser fullscreen (Esc/F11), drop our overlay too.
document.addEventListener('fullscreenchange', function() {
    if (!document.fullscreenElement && watchOpen) closeWatch();
});

// ---- In-page lightbox ----
// Replaces window.open for history/result/loop images. Pinch, pan, swipe and
// double-tap zoom are implemented with pointer events: native pinch on a
// fixed overlay zooms the page's visual viewport in iOS Safari and leaks
// that zoom back to the page after closing, so it can't be relied on.
const lightboxEl = document.getElementById('lightbox');
const lightboxStage = document.getElementById('lightboxStage');
const lightboxImg = document.getElementById('lightboxImg');
const lightboxCaption = document.getElementById('lightboxCaption');
const lightboxPrevBtn = document.getElementById('lightboxPrev');
const lightboxNextBtn = document.getElementById('lightboxNext');
const lightboxCloseBtn = document.getElementById('lightboxClose');
let lbItems = [];
let lbIndex = 0;
let lbOpen = false;
let lbScale = 1, lbTx = 0, lbTy = 0;

function lbApplyTransform() {
    lightboxImg.style.transform = `translate(${lbTx}px, ${lbTy}px) scale(${lbScale})`;
}
function lbClampPan() {
    // The image fills the stage (object-fit contain), so keep the scaled
    // box covering the viewport instead of letting it fly off-screen.
    const w = lightboxStage.clientWidth, h = lightboxStage.clientHeight;
    lbTx = Math.min(0, Math.max(w - w * lbScale, lbTx));
    lbTy = Math.min(0, Math.max(h - h * lbScale, lbTy));
}
function lbResetTransform() {
    lbScale = 1; lbTx = 0; lbTy = 0;
    lbApplyTransform();
}
function lbShow(i) {
    const n = lbItems.length;
    lbIndex = ((i % n) + n) % n;
    lbResetTransform();
    lightboxImg.src = lbItems[lbIndex].src;
    lightboxCaption.textContent = lbItems[lbIndex].caption || '';
    if (n > 1) {  // preload neighbors so swipes feel instant
        new Image().src = lbItems[(lbIndex + 1) % n].src;
        new Image().src = lbItems[(lbIndex - 1 + n) % n].src;
    }
}
function openLightbox(items, index) {
    if (!lightboxEl || !items || !items.length) return;
    lbItems = items;
    lbOpen = true;
    const multi = items.length > 1;
    if (lightboxPrevBtn) lightboxPrevBtn.style.display = multi ? 'block' : 'none';
    if (lightboxNextBtn) lightboxNextBtn.style.display = multi ? 'block' : 'none';
    lbShow(index || 0);
    lightboxEl.classList.add('visible');
    document.body.style.overflow = 'hidden';
}
function closeLightbox() {
    if (!lightboxEl) return;
    lbOpen = false;
    lightboxEl.classList.remove('visible');
    lightboxImg.removeAttribute('src');  // frees the decoded bitmap on iOS
    document.body.style.overflow = '';
}

// Zoom toward a screen point (transform-origin is 0 0, so the anchor math
// keeps whatever is under the tap in place).
function lbToggleZoom(clientX, clientY) {
    if (lbScale > 1) { lbResetTransform(); return; }
    const rect = lightboxStage.getBoundingClientRect();
    const s = 2.5;
    const x = clientX - rect.left, y = clientY - rect.top;
    lbTx = x - (x - lbTx) * (s / lbScale);
    lbTy = y - (y - lbTy) * (s / lbScale);
    lbScale = s;
    lbClampPan();
    lbApplyTransform();
}

if (lightboxEl) {
    // Gesture rules: two pointers pinch; one pointer pans when zoomed and
    // swipes between images when not. Swipe and pan never overlap.
    const lbPointers = new Map();
    let lbPinchStart = null;
    let lbPanStart = null;
    let lbSwipeStart = null;
    let lbLastTap = { t: 0, x: 0, y: 0 };

    lightboxStage.addEventListener('pointerdown', function(e) {
        e.preventDefault();
        try { lightboxStage.setPointerCapture(e.pointerId); } catch (err) {}
        lbPointers.set(e.pointerId, { x: e.clientX, y: e.clientY });
        if (lbPointers.size === 2) {
            const pts = Array.from(lbPointers.values());
            lbPinchStart = {
                dist: Math.hypot(pts[0].x - pts[1].x, pts[0].y - pts[1].y),
                scale: lbScale, tx: lbTx, ty: lbTy
            };
            lbPanStart = null;
            lbSwipeStart = null;
        } else if (lbPointers.size === 1) {
            if (lbScale > 1) lbPanStart = { x: e.clientX, y: e.clientY, tx: lbTx, ty: lbTy };
            else lbSwipeStart = { x: e.clientX, y: e.clientY };
        }
    });
    lightboxStage.addEventListener('pointermove', function(e) {
        const p = lbPointers.get(e.pointerId);
        if (!p) return;
        p.x = e.clientX; p.y = e.clientY;
        if (lbPointers.size === 2 && lbPinchStart) {
            const pts = Array.from(lbPointers.values());
            const dist = Math.hypot(pts[0].x - pts[1].x, pts[0].y - pts[1].y);
            const rect = lightboxStage.getBoundingClientRect();
            const midX = (pts[0].x + pts[1].x) / 2 - rect.left;
            const midY = (pts[0].y + pts[1].y) / 2 - rect.top;
            const next = Math.min(5, Math.max(1, lbPinchStart.scale * (dist / lbPinchStart.dist)));
            lbTx = midX - (midX - lbPinchStart.tx) * (next / lbPinchStart.scale);
            lbTy = midY - (midY - lbPinchStart.ty) * (next / lbPinchStart.scale);
            lbScale = next;
            lbClampPan();
            lbApplyTransform();
        } else if (lbPanStart && lbScale > 1) {
            lbTx = lbPanStart.tx + (e.clientX - lbPanStart.x);
            lbTy = lbPanStart.ty + (e.clientY - lbPanStart.y);
            lbClampPan();
            lbApplyTransform();
        }
    });
    const lbPointerEnd = function(e) {
        lbPointers.delete(e.pointerId);
        if (lbPointers.size < 2) lbPinchStart = null;
        if (lbPointers.size > 0) return;
        if (lbSwipeStart && lbScale === 1) {
            const dx = e.clientX - lbSwipeStart.x;
            const dy = e.clientY - lbSwipeStart.y;
            if (Math.abs(dx) > 50 && Math.abs(dx) > Math.abs(dy) && lbItems.length > 1) {
                lbShow(lbIndex + (dx < 0 ? 1 : -1));
            } else if (Math.abs(dx) < 8 && Math.abs(dy) < 8) {
                const now = Date.now();
                if (now - lbLastTap.t < 300 &&
                    Math.abs(e.clientX - lbLastTap.x) < 25 && Math.abs(e.clientY - lbLastTap.y) < 25) {
                    lbToggleZoom(e.clientX, e.clientY);
                    lbLastTap = { t: 0, x: 0, y: 0 };
                } else {
                    lbLastTap = { t: now, x: e.clientX, y: e.clientY };
                }
            }
        }
        // Double-tap back to 1x also works while zoomed: a zoomed single
        // pointer starts a pan, so detect the no-movement case here.
        if (lbPanStart && Math.abs(e.clientX - lbPanStart.x) < 8 && Math.abs(e.clientY - lbPanStart.y) < 8) {
            const now = Date.now();
            if (now - lbLastTap.t < 300 &&
                Math.abs(e.clientX - lbLastTap.x) < 25 && Math.abs(e.clientY - lbLastTap.y) < 25) {
                lbToggleZoom(e.clientX, e.clientY);
                lbLastTap = { t: 0, x: 0, y: 0 };
            } else {
                lbLastTap = { t: now, x: e.clientX, y: e.clientY };
            }
        }
        lbPanStart = null;
        lbSwipeStart = null;
    };
    lightboxStage.addEventListener('pointerup', lbPointerEnd);
    lightboxStage.addEventListener('pointercancel', lbPointerEnd);

    if (lightboxCloseBtn) lightboxCloseBtn.addEventListener('click', closeLightbox);
    if (lightboxPrevBtn) lightboxPrevBtn.addEventListener('click', function() { lbShow(lbIndex - 1); });
    if (lightboxNextBtn) lightboxNextBtn.addEventListener('click', function() { lbShow(lbIndex + 1); });
    document.addEventListener('keydown', function(e) {
        if (!lbOpen) return;
        if (e.key === 'Escape') closeLightbox();
        else if (e.key === 'ArrowLeft' && lbItems.length > 1) lbShow(lbIndex - 1);
        else if (e.key === 'ArrowRight' && lbItems.length > 1) lbShow(lbIndex + 1);
    });
    // Older-Safari belt and braces: suppress its proprietary gesture events.
    document.addEventListener('gesturestart', function(e) { if (lbOpen) e.preventDefault(); });
}

// Touch path for the top-left model-name reveal (:hover never fires on
// iPad/iPhone): a tap toggles the badge, which auto-hides after a moment.
(function() {
    const zone = document.getElementById('modelHover');
    if (!zone) return;
    let hideTimer = null;
    zone.addEventListener('click', function() {
        const on = zone.classList.toggle('show');
        if (hideTimer) { clearTimeout(hideTimer); hideTimer = null; }
        if (on) hideTimer = setTimeout(function() { zone.classList.remove('show'); }, 4000);
    });
})();

if (strengthSlider) strengthSlider.addEventListener('input', function() { if (strengthValue) strengthValue.textContent = strengthSlider.value; });

function useSeed(seed) {
    var el = document.getElementById('seed'); if (el) el.value = seed;
}

// Load an already-generated image into the reference-image slot (same as
// uploading it), so it can drive img2img or be edited with the inpaint brush.
async function useAsReference(filename) {
    try {
        const res = await fetch('/images/' + filename, { headers: getAuthHeaders() });
        if (!res.ok) throw new Error('HTTP ' + res.status);
        const blob = await res.blob();
        handleImageFile(blob);  // reuses the upload pipeline (preview + controls)
        const top = document.getElementById('uploadArea');
        if (top && top.scrollIntoView) top.scrollIntoView({ behavior: 'smooth', block: 'center' });
    } catch (err) {
        alert('Could not load image as reference: ' + err.message);
    }
}

// Copy an image into the server-side .saved subdir (preserves it from
// archive/delete-today). `el` is the clicked button; we reflect status on it.
// Text-label buttons get a label swap; compact icon buttons just restyle.
async function saveHidden(el, filename) {
    const labeled = el && el.classList.contains('save-hidden-btn');
    const original = el ? el.textContent : '';
    if (el) { el.classList.add('saving'); if (labeled) el.textContent = 'Saving…'; }
    try {
        const res = await fetch('/save-hidden', {
            method: 'POST',
            headers: getAuthHeaders({ 'Content-Type': 'application/json' }),
            body: JSON.stringify({ filename: filename })
        });
        const body = await res.json();
        if (body.success) {
            if (el) {
                el.classList.remove('saving'); el.classList.add('saved');
                if (labeled) el.textContent = '★ Saved'; else el.title = 'Saved';
            }
        } else {
            if (el) { el.classList.remove('saving'); if (labeled) el.textContent = original; }
            alert('Save failed: ' + (body.error || 'unknown error'));
        }
    } catch (err) {
        if (el) { el.classList.remove('saving'); if (labeled) el.textContent = original; }
        alert('Save failed: ' + err.message);
    }
}

// Re-render the reference thumbnails and every control whose visibility
// depends on how many references are loaded.
function syncRefUI() {
    currentInputImage = currentInputImages[0] || null;
    if (refThumbs) {
        refThumbs.innerHTML = '';
        currentInputImages.forEach(function(dataUrl, i) {
            const thumb = document.createElement('div');
            thumb.className = 'ref-thumb';
            const img = document.createElement('img');
            img.src = dataUrl;
            img.alt = 'Reference ' + (i + 1);
            const badge = document.createElement('span');
            badge.className = 'ref-badge';
            badge.textContent = i + 1;
            const rm = document.createElement('button');
            rm.type = 'button';
            rm.className = 'clear-btn';
            rm.textContent = 'X';
            rm.title = 'Remove this reference';
            rm.addEventListener('click', function(e) {
                e.stopPropagation();
                currentInputImages.splice(i, 1);
                syncRefUI();
            });
            thumb.appendChild(img);
            thumb.appendChild(badge);
            thumb.appendChild(rm);
            refThumbs.appendChild(thumb);
        });
    }
    const n = currentInputImages.length;
    if (uploadArea) uploadArea.style.display = n >= MAX_REFERENCE_IMAGES ? 'none' : 'block';
    const urlImportRow = document.getElementById('urlImportRow');
    if (urlImportRow) urlImportRow.style.display = n >= MAX_REFERENCE_IMAGES ? 'none' : 'flex';
    if (uploadPlaceholder) {
        const span = uploadPlaceholder.querySelector('span');
        if (span) span.textContent = n === 0
            ? 'Tap or click to add up to 3 images'
            : `+ Add image (${n}/${MAX_REFERENCE_IMAGES})`;
    }
    // Strength only applies to single-image FLUX.1 img2img; multi-reference
    // runs through Kontext/FLUX.2 conditioning, which ignores it.
    if (strengthControl) strengthControl.style.display = n === 1 ? 'flex' : 'none';
    const describeControl = document.getElementById('describeControl');
    if (describeControl) describeControl.style.display = n > 0 ? 'block' : 'none';
    if (aspectModeControl) aspectModeControl.style.display = n > 0 ? 'block' : 'none';
    if (multiRefHint) multiRefHint.style.display = n > 1 ? 'block' : 'none';
    if (inputImage) inputImage.value = '';
    if (typeof refreshInpaintAvailability === 'function') refreshInpaintAvailability();
}

// FLUX conditions on ≤~2MP inputs, so anything bigger is wasted upload; raw
// phone photos (base64, ×3 refs) were blowing past the server's 64MB body cap.
var MAX_UPLOAD_EDGE = 2048;
var UPLOAD_JPEG_QUALITY = 0.92;

function shrinkImageDataUrl(dataUrl, cb) {
    var img = new Image();
    img.onload = function() {
        var w = img.naturalWidth, h = img.naturalHeight;
        var scale = Math.min(1, MAX_UPLOAD_EDGE / Math.max(w, h));
        if (scale === 1 && dataUrl.length < 4 * 1024 * 1024) { cb(dataUrl); return; }
        var canvas = document.createElement('canvas');
        canvas.width = Math.max(1, Math.round(w * scale));
        canvas.height = Math.max(1, Math.round(h * scale));
        canvas.getContext('2d').drawImage(img, 0, 0, canvas.width, canvas.height);
        cb(canvas.toDataURL('image/jpeg', UPLOAD_JPEG_QUALITY));
    };
    // Formats the browser can't decode (e.g. HEIC outside Safari) pass
    // through unshrunk; the server's PIL decode is the real gate.
    img.onerror = function() { cb(dataUrl); };
    img.src = dataUrl;
}

// Camera RAW files (NEF etc.) can't be decoded by the browser, so they're
// posted as-is to /convert-raw, which returns a JPEG data URL that then
// behaves like any other reference. Detection is by extension — browsers
// report an empty MIME type for RAW files.
var RAW_EXTENSIONS = ['nef', 'nrw', 'dng', 'cr2', 'cr3', 'arw', 'raf', 'orf', 'rw2'];

function isRawFile(file) {
    var ext = ((file && file.name) || '').split('.').pop().toLowerCase();
    return RAW_EXTENSIONS.indexOf(ext) !== -1;
}

function handleRawFile(file) {
    var span = uploadPlaceholder ? uploadPlaceholder.querySelector('span') : null;
    if (span) span.textContent = 'Converting ' + file.name + '…';
    var fd = new FormData();
    fd.append('file', file, file.name);
    fetch('/convert-raw', { method: 'POST', headers: getAuthHeaders(), body: fd })
        .then(function(res) {
            return res.json().catch(function() { return {}; }).then(function(data) {
                if (!res.ok || !data.success) throw new Error(data.error || ('HTTP ' + res.status));
                if (currentInputImages.length < MAX_REFERENCE_IMAGES) currentInputImages.push(data.image);
            });
        })
        .catch(function(err) { alert('RAW conversion failed: ' + err.message); })
        .then(function() { syncRefUI(); }); // also restores the placeholder label
}

// Import a reference from either a web URL or a file path on the server's
// own filesystem. URLs go through /fetch-image-url (server-side fetch, so
// browser CORS restrictions don't apply); anything else is treated as a
// server path and goes through /fetch-image-path (absolute, ~, or relative
// to web-generated/). Both return a JPEG data URL that then behaves like any
// uploaded reference.
function addImageSource(value) {
    value = (value || '').trim();
    if (!value) return;
    if (currentInputImages.length >= MAX_REFERENCE_IMAGES) return;
    var isUrl = /^https?:\/\//i.test(value);
    var span = uploadPlaceholder ? uploadPlaceholder.querySelector('span') : null;
    if (span) span.textContent = isUrl ? 'Fetching image from URL…' : 'Loading image from server path…';
    fetch(isUrl ? '/fetch-image-url' : '/fetch-image-path', {
        method: 'POST',
        headers: getAuthHeaders({ 'Content-Type': 'application/json' }),
        body: JSON.stringify(isUrl ? { url: value } : { path: value })
    })
        .then(function(res) {
            return res.json().catch(function() { return {}; }).then(function(data) {
                if (!res.ok || !data.success) throw new Error(data.error || ('HTTP ' + res.status));
                if (currentInputImages.length < MAX_REFERENCE_IMAGES) currentInputImages.push(data.image);
                var input = document.getElementById('imageUrlInput');
                if (input) input.value = '';
            });
        })
        .catch(function(err) {
            alert((isUrl ? 'Could not fetch image URL: ' : 'Could not load server image: ') + err.message);
        })
        .then(function() { syncRefUI(); }); // also restores the placeholder label
}

// Server file browser: navigate folders on the server's filesystem (starting
// at the archive folder) and click a thumbnail to attach it as a reference,
// instead of typing a path into the URL import row by hand. Thumbnails go
// through /browse-thumb (authenticated — unlike /images/, it can read any
// path on disk, so it can't be exempted from the API-key check) fetched as
// a blob and turned into an object URL; those are revoked whenever the grid
// is rebuilt or the picker closes so they don't leak.
const archivePicker = document.getElementById('archivePicker');
const archivePickerGrid = document.getElementById('archivePickerGrid');
const archivePickerPath = document.getElementById('archivePickerPath');
const archivePickerUp = document.getElementById('archivePickerUp');
const archivePickerClose = document.getElementById('archivePickerClose');
const archivePickerJumpInput = document.getElementById('archivePickerJumpInput');
const archivePickerJumpBtn = document.getElementById('archivePickerJumpBtn');
const browseArchiveBtn = document.getElementById('browseArchiveBtn');

let archiveCurrentDir = 'archive';
let archiveParentDir = null;
let archiveThumbUrls = [];

function revokeArchiveThumbUrls() {
    archiveThumbUrls.forEach(function(u) { URL.revokeObjectURL(u); });
    archiveThumbUrls = [];
}

function closeArchivePicker() {
    if (archivePicker) archivePicker.classList.remove('visible');
    revokeArchiveThumbUrls();
}

function loadArchiveThumb(img, path) {
    fetch('/browse-thumb?path=' + encodeURIComponent(path), { headers: getAuthHeaders() })
        .then(function(res) { if (!res.ok) throw new Error('HTTP ' + res.status); return res.blob(); })
        .then(function(blob) {
            const url = URL.createObjectURL(blob);
            archiveThumbUrls.push(url);
            img.src = url;
        })
        .catch(function() {});
}

function loadArchiveDir(dir) {
    if (!archivePicker || !archivePickerGrid) return;
    archivePickerGrid.innerHTML = '<div class="archive-picker-empty">Loading…</div>';
    revokeArchiveThumbUrls();
    fetch('/browse-files?dir=' + encodeURIComponent(dir), { headers: getAuthHeaders() })
        .then(function(res) { return res.json().then(function(data) { return { ok: res.ok, data: data }; }); })
        .then(function(r) {
            if (!r.ok || !r.data.success) throw new Error((r.data && r.data.error) || 'could not list folder');
            const data = r.data;
            archiveCurrentDir = data.dir;
            archiveParentDir = data.parent;
            if (archivePickerPath) archivePickerPath.textContent = data.dir;
            if (archivePickerUp) archivePickerUp.disabled = !data.parent;
            const dirs = data.dirs || [], files = data.files || [];
            archivePickerGrid.innerHTML = '';
            if (!dirs.length && !files.length) {
                archivePickerGrid.innerHTML = '<div class="archive-picker-empty">Empty folder.</div>';
                return;
            }
            dirs.forEach(function(name) {
                const btn = document.createElement('button');
                btn.type = 'button';
                btn.className = 'archive-picker-folder';
                btn.title = name;
                btn.innerHTML = '<span class="folder-icon">&#128193;</span><span class="folder-name"></span>';
                btn.querySelector('.folder-name').textContent = name;
                btn.addEventListener('click', function() { loadArchiveDir(data.dir + '/' + name); });
                archivePickerGrid.appendChild(btn);
            });
            files.forEach(function(f) {
                const btn = document.createElement('button');
                btn.type = 'button';
                btn.className = 'archive-picker-thumb';
                btn.title = f.filename;
                const img = document.createElement('img');
                img.alt = f.filename;
                const fullPath = data.dir + '/' + f.filename;
                loadArchiveThumb(img, fullPath);
                btn.appendChild(img);
                btn.addEventListener('click', function() {
                    addImageSource(fullPath);
                    closeArchivePicker();
                });
                archivePickerGrid.appendChild(btn);
            });
        })
        .catch(function(err) {
            archivePickerGrid.innerHTML = '<div class="archive-picker-empty">Could not load folder: ' + err.message + '</div>';
        });
}

function openArchivePicker() {
    if (!archivePicker) return;
    archivePicker.classList.add('visible');
    loadArchiveDir(archiveCurrentDir);
}

if (browseArchiveBtn) browseArchiveBtn.addEventListener('click', openArchivePicker);
if (archivePickerClose) archivePickerClose.addEventListener('click', closeArchivePicker);
if (archivePickerUp) archivePickerUp.addEventListener('click', function() {
    if (archiveParentDir) loadArchiveDir(archiveParentDir);
});
if (archivePickerJumpBtn) archivePickerJumpBtn.addEventListener('click', function() {
    const v = archivePickerJumpInput ? archivePickerJumpInput.value.trim() : '';
    if (v) loadArchiveDir(v);
});
if (archivePickerJumpInput) archivePickerJumpInput.addEventListener('keydown', function(e) {
    if (e.key === 'Enter') { e.preventDefault(); archivePickerJumpBtn.click(); }
});
if (archivePicker) archivePicker.addEventListener('click', function(e) {
    if (e.target === archivePicker) closeArchivePicker();
});

function handleImageFile(file) {
    if (currentInputImages.length >= MAX_REFERENCE_IMAGES) return;
    if (isRawFile(file)) { handleRawFile(file); return; }
    var reader = new FileReader();
    reader.onload = function(e) {
        shrinkImageDataUrl(e.target.result, function(shrunk) {
            if (currentInputImages.length >= MAX_REFERENCE_IMAGES) return;
            currentInputImages.push(shrunk);
            syncRefUI();
        });
    };
    reader.readAsDataURL(file);
}

function addImageFiles(fileList) {
    Array.from(fileList || [])
        .filter(function(f) { return f && (f.type.indexOf('image/') === 0 || isRawFile(f)); })
        .slice(0, Math.max(0, MAX_REFERENCE_IMAGES - currentInputImages.length))
        .forEach(handleImageFile);
}

function clearRefs() {
    currentInputImages = [];
    syncRefUI();
    if (typeof resetInpaint === 'function') resetInpaint();
}

// ---- Live progress for slow VLM workflows (describe / boost / evolve) ----
// Each workflow shows "label — 12s" on its trigger button and in the main
// status bar, ticking on every poll cycle. While any of them is active,
// renderRunning's idle reset of the status bar is suppressed (vlmActiveCount)
// so the message isn't hidden between generation polls.
let vlmActiveCount = 0;

function fmtElapsed(t0) {
    const s = Math.round((Date.now() - t0) / 1000);
    return s < 60 ? s + 's' : Math.floor(s / 60) + 'm ' + (s % 60) + 's';
}

function vlmStatusStart(btn, label) {
    const t0 = Date.now();
    vlmActiveCount += 1;
    let ended = false;
    const handle = {
        elapsed: function() { return fmtElapsed(t0); },
        // Refresh button + status bar. `detail` replaces the base label when
        // given; `btnText` overrides the button text when the bar line is too
        // long for a button.
        tick: function(detail, btnText) {
            const msg = (detail || label) + ' — ' + fmtElapsed(t0);
            if (btn) btn.textContent = btnText || msg;
            if (status && statusText) {
                status.className = 'status generating';
                statusText.textContent = msg;
            }
        },
        end: function() {
            if (ended) return;
            ended = true;
            vlmActiveCount = Math.max(0, vlmActiveCount - 1);
        }
    };
    handle.tick();
    return handle;
}

// The reverse path: have the local vision model write a detailed prompt from
// the reference photo(s) — a composite description of one combined scene
// when several are attached — drop it into the prompt box, and generate a
// fresh image from that prompt alone (the references are set aside for the
// submission so the result comes from the description, not img2img).
async function runReversePath() {
    const btn = document.getElementById('describeBtn');
    if (!currentInputImages.length) { alert('Upload a reference image first.'); return; }
    const promptEl = document.getElementById('prompt');
    const oldLabel = btn.textContent;
    btn.disabled = true;
    const prog = vlmStatusStart(btn, currentInputImages.length > 1
        ? 'Reverse: describing ' + currentInputImages.length + ' photos as one scene'
        : 'Reverse: describing photo with the vision model');
    try {
        const thinkEl = document.getElementById('describeThink');
        const res = await fetch('/describe', {
            method: 'POST',
            headers: getAuthHeaders({ 'Content-Type': 'application/json' }),
            body: JSON.stringify({
                images: currentInputImages,
                think: thinkEl ? thinkEl.checked : false
            })
        });
        const submitted = await res.json().catch(() => ({}));
        if (!res.ok || !submitted.success) throw new Error(submitted.error || `HTTP ${res.status}`);
        const data = await pollVlmJob('/describe/' + submitted.describe_id, null,
                                      function() { prog.tick(); });
        if (promptEl) promptEl.value = data.prompt;
        if (status && statusText) {
            status.className = 'status generating';
            statusText.textContent = 'Description ready in ' + prog.elapsed() + ' — generating from it…';
        }
        prog.end();  // generation progress takes over the status bar from here
        const refs = currentInputImages;
        currentInputImages = [];
        syncRefUI();
        try {
            await doGenerate();
        } finally {
            currentInputImages = refs;
            syncRefUI();
        }
    } catch (err) {
        alert('Reverse path failed: ' + err.message);
    } finally {
        prog.end();
        btn.disabled = false;
        btn.textContent = oldLabel;
    }
}

const describeBtn = document.getElementById('describeBtn');
if (describeBtn) describeBtn.addEventListener('click', runReversePath);

// Prompt boost: the local VLM rewrites whatever is in the prompt box into a
// stronger prompt for the loaded model (the server knows which backend is
// active and picks the matching prompting idiom; has_image tells it whether
// references are attached, which shifts the idiom to edit instructions on
// FLUX.2/Kontext and desired-final-image description on SDXL/FLUX.1).
// The level select (1-5) sets how far the rewrite may depart from the draft.
async function runBoost() {
    const btn = document.getElementById('boostBtn');
    const promptEl = document.getElementById('prompt');
    const levelEl = document.getElementById('boostLevel');
    const thinkEl = document.getElementById('boostThink');
    const draft = (promptEl.value || '').trim();
    if (!draft) { alert('Type a prompt to boost first.'); return; }
    const oldLabel = btn.textContent;
    btn.disabled = true;
    const level = levelEl ? parseInt(levelEl.value, 10) : 3;
    const negativeEl = document.getElementById('negativePrompt');
    const prog = vlmStatusStart(btn, 'Boosting prompt (level ' + level + ')');
    try {
        const res = await fetch('/boost', {
            method: 'POST',
            headers: getAuthHeaders({ 'Content-Type': 'application/json' }),
            body: JSON.stringify({
                prompt: draft,
                level: level,
                think: thinkEl ? thinkEl.checked : false,
                has_image: currentInputImages.length > 0,
                negative_prompt: negativeEl && negativeEl.value.trim() ? negativeEl.value.trim() : null
            })
        });
        const submitted = await res.json().catch(() => ({}));
        if (!res.ok || !submitted.success) throw new Error(submitted.error || `HTTP ${res.status}`);
        const data = await pollVlmJob('/boost/' + submitted.boost_id, null,
                                      function() { prog.tick(); });
        if (data.prompt) promptEl.value = data.prompt;
        if (data.negative_prompt && negativeEl) negativeEl.value = data.negative_prompt;
        if (status && statusText) {
            status.className = 'status generating';
            statusText.textContent = 'Prompt boosted in ' + prog.elapsed() + '.';
        }
    } catch (err) {
        alert('Prompt boost failed: ' + err.message);
    } finally {
        prog.end();
        btn.disabled = false;
        btn.textContent = oldLabel;
    }
}

const boostBtn = document.getElementById('boostBtn');
if (boostBtn) boostBtn.addEventListener('click', runBoost);

// Evolve & generate: boost the base prompt N times independently (each call
// told it is variation i of N so the VLM takes divergent directions, using
// the boost level/think settings above), and queue one generation per evolved
// prompt as soon as its boost lands. The prompt box keeps the base prompt;
// the evolved prompts show up on the finished jobs and in .prompt sidecars.
async function runEvolveGenerate() {
    const btn = document.getElementById('evolveBtn');
    const promptEl = document.getElementById('prompt');
    const levelEl = document.getElementById('boostLevel');
    const thinkEl = document.getElementById('boostThink');
    const countEl = document.getElementById('evolveCount');
    const negativeEl = document.getElementById('negativePrompt');
    const base = (promptEl.value || '').trim();
    if (!base) { alert('Type a base prompt to evolve first.'); return; }
    const n = countEl ? parseInt(countEl.value, 10) : 4;

    const built = buildGenerateFormData();
    if (!built) return;
    recordPromptHistory(base);

    const oldLabel = btn.textContent;
    btn.disabled = true;
    let evolved = 0, queued = 0, failed = 0;
    const prog = vlmStatusStart(btn, `Evolving ${n} prompt variations`);
    const tick = function() {
        const btnText = `Evolving ${evolved}/${n}… ${prog.elapsed()}`;
        // Once a generation is running, its step-by-step progress owns the
        // status bar; keep the evolve tally on the button only.
        if (runningJobId) {
            btn.textContent = btnText;
            return;
        }
        prog.tick(`Evolving ${n} prompt variations — ${evolved} evolved, `
                  + `${queued} queued` + (failed ? `, ${failed} failed` : ''),
                  btnText);
    };
    tick();
    try {
        await Promise.all(Array.from({ length: n }, async function(_, i) {
            try {
                const res = await fetch('/boost', {
                    method: 'POST',
                    headers: getAuthHeaders({ 'Content-Type': 'application/json' }),
                    body: JSON.stringify({
                        prompt: base,
                        level: levelEl ? parseInt(levelEl.value, 10) : 3,
                        think: thinkEl ? thinkEl.checked : false,
                        has_image: currentInputImages.length > 0,
                        negative_prompt: negativeEl && negativeEl.value.trim() ? negativeEl.value.trim() : null,
                        variant_index: i + 1,
                        variant_count: n
                    })
                });
                const submitted = await res.json().catch(() => ({}));
                if (!res.ok || !submitted.success) throw new Error(submitted.error || `HTTP ${res.status}`);
                const data = await pollVlmJob('/boost/' + submitted.boost_id, null, tick);
                if (!data.prompt) throw new Error('boost returned no prompt');
                evolved += 1; tick();
                const overrides = { prompt: data.prompt };
                if (data.negative_prompt) overrides.negative_prompt = data.negative_prompt;
                const gres = await fetch('/generate', {
                    method: 'POST',
                    headers: getAuthHeaders({ 'Content-Type': 'application/json' }),
                    body: JSON.stringify(Object.assign({}, built.formData, overrides))
                });
                const gdata = await gres.json().catch(() => ({}));
                if (!gres.ok || !gdata.success) throw new Error(gdata.error || `HTTP ${gres.status}`);
                queued += 1; tick();
                noteActivity();
                schedulePoll(0);
            } catch (err) {
                failed += 1; tick();
                console.warn(`evolve variation ${i + 1} failed:`, err);
            }
        }));
    } finally {
        prog.end();
        btn.disabled = false;
        btn.textContent = oldLabel;
    }
    if (status && statusText) {
        if (!queued) {
            status.className = 'status error';
            statusText.textContent = 'Evolve failed: no variation could be boosted or queued.';
        } else if (!runningJobId) {
            // With a generation already running, its live progress owns the bar.
            status.className = 'status generating';
            statusText.textContent = `Queued ${queued} evolved variation${queued > 1 ? 's' : ''}`
                + ` in ${prog.elapsed()}` + (failed ? ` (${failed} failed)` : '');
        }
    }
}

const evolveBtn = document.getElementById('evolveBtn');
if (evolveBtn) evolveBtn.addEventListener('click', runEvolveGenerate);

if (uploadArea) {
    uploadArea.addEventListener('click', function() { if (inputImage) inputImage.click(); });
    uploadArea.addEventListener('dragover', function(e) { e.preventDefault(); uploadArea.classList.add('dragover'); });
    uploadArea.addEventListener('dragleave', function() { uploadArea.classList.remove('dragover'); });
    uploadArea.addEventListener('drop', function(e) {
        e.preventDefault();
        uploadArea.classList.remove('dragover');
        if (e.dataTransfer.files && e.dataTransfer.files.length) {
            addImageFiles(e.dataTransfer.files);
            return;
        }
        // An image dragged from another browser tab arrives as a URL, not a file.
        var uri = e.dataTransfer.getData('text/uri-list') || e.dataTransfer.getData('text/plain');
        if (uri) addImageSource(uri.split('\n')[0]);
    });
}
if (inputImage) inputImage.addEventListener('change', function(e) { addImageFiles(e.target.files); });

const imageUrlInput = document.getElementById('imageUrlInput');
const addUrlBtn = document.getElementById('addUrlBtn');
if (addUrlBtn) addUrlBtn.addEventListener('click', function() { addImageSource(imageUrlInput ? imageUrlInput.value : ''); });
if (imageUrlInput) imageUrlInput.addEventListener('keydown', function(e) {
    if (e.key === 'Enter') { e.preventDefault(); addImageSource(imageUrlInput.value); }
});

// The save-each-frame toggle only means something while live previews are on
// (the saved frames ARE the preview decodes), so hide it otherwise.
function syncSavePreviewsVisibility() {
    var show = document.getElementById('showPreview');
    var label = document.getElementById('savePreviewsLabel');
    if (label) label.style.display = (show && show.checked) ? '' : 'none';
}
(function() {
    var show = document.getElementById('showPreview');
    if (show) show.addEventListener('change', syncSavePreviewsVisibility);
    syncSavePreviewsVisibility();
})();

// ---- Inpainting brush (FLUX.2 only) ----
const inpaintControl = document.getElementById('inpaintControl');
const inpaintMode = document.getElementById('inpaintMode');
const inpaintTools = document.getElementById('inpaintTools');
const inpaintBaseImg = document.getElementById('inpaintBaseImg');
const inpaintCanvas = document.getElementById('inpaintCanvas');
const brushSize = document.getElementById('brushSize');
const inpaintEraser = document.getElementById('inpaintEraser');
const inpaintClear = document.getElementById('inpaintClear');
const inpaintCtx = inpaintCanvas ? inpaintCanvas.getContext('2d') : null;
const INPAINT_MAX_DIM = 1536;
let eraserOn = false;
let painting = false;
let lastPt = null;

function refreshInpaintAvailability() {
    if (!inpaintControl) return;
    // Inpainting works on exactly one image (the mask applies to the primary).
    const ok = currentInputImages.length === 1
        && (window.__fluxVersion === 2 || window.__inpaintCapable);
    inpaintControl.style.display = ok ? 'block' : 'none';
    if (!ok && inpaintMode && inpaintMode.checked) {
        inpaintMode.checked = false;
        applyInpaintMode();
    }
}

function resetInpaint() {
    if (inpaintMode) inpaintMode.checked = false;
    if (inpaintTools) inpaintTools.style.display = 'none';
    if (inpaintCanvas && inpaintCtx) inpaintCtx.clearRect(0, 0, inpaintCanvas.width, inpaintCanvas.height);
    eraserOn = false;
    if (inpaintEraser) inpaintEraser.classList.remove('active');
}

function applyInpaintMode() {
    if (!inpaintMode) return;
    const on = inpaintMode.checked;
    if (inpaintTools) inpaintTools.style.display = on ? 'block' : 'none';
    // FLUX.2's masked diffusion ignores strength, so the slider hides in its
    // inpaint mode. On SDXL the slider IS the inpaint control — the denoise
    // level of the painted region — so it stays, with a hint to match.
    const sdxlInpaint = on && window.__fluxVersion !== 2;
    if (strengthControl) {
        strengthControl.style.display =
            ((on && !sdxlInpaint) || !currentInputImage) ? 'none' : 'flex';
        const hint = strengthControl.querySelector('.strength-hint');
        if (hint) hint.textContent = sdxlInpaint
            ? 'Denoise level for the painted region: ~1.0 fully replaces it, 0.4–0.7 keeps some of the original showing through'
            : '0 = closest to original, 0.5 = default, 1 = most change';
    }
    if (on && currentInputImage && inpaintBaseImg) {
        inpaintBaseImg.src = currentInputImage;
    }
}

if (inpaintBaseImg) inpaintBaseImg.addEventListener('load', function() {
    if (!inpaintCanvas) return;
    const nw = inpaintBaseImg.naturalWidth || inpaintBaseImg.width;
    const nh = inpaintBaseImg.naturalHeight || inpaintBaseImg.height;
    if (!nw || !nh) return;
    const scale = Math.min(1, INPAINT_MAX_DIM / Math.max(nw, nh));
    inpaintCanvas.width = Math.max(1, Math.round(nw * scale));
    inpaintCanvas.height = Math.max(1, Math.round(nh * scale));
    if (inpaintCtx) inpaintCtx.clearRect(0, 0, inpaintCanvas.width, inpaintCanvas.height);
});

if (inpaintMode) inpaintMode.addEventListener('change', applyInpaintMode);
if (inpaintEraser) inpaintEraser.addEventListener('click', function() {
    eraserOn = !eraserOn;
    inpaintEraser.classList.toggle('active', eraserOn);
});
if (inpaintClear) inpaintClear.addEventListener('click', function() {
    if (inpaintCanvas && inpaintCtx) inpaintCtx.clearRect(0, 0, inpaintCanvas.width, inpaintCanvas.height);
});

function inpaintPoint(e) {
    const rect = inpaintCanvas.getBoundingClientRect();
    const sx = inpaintCanvas.width / rect.width;
    const sy = inpaintCanvas.height / rect.height;
    return { x: (e.clientX - rect.left) * sx, y: (e.clientY - rect.top) * sy, s: (sx + sy) / 2 };
}
function inpaintStroke(a, b) {
    if (!inpaintCtx) return;
    const radius = (parseInt(brushSize ? brushSize.value : 40, 10) || 40) * b.s;
    inpaintCtx.globalCompositeOperation = eraserOn ? 'destination-out' : 'source-over';
    inpaintCtx.strokeStyle = 'rgba(255,0,0,0.55)';
    inpaintCtx.fillStyle = 'rgba(255,0,0,0.55)';
    inpaintCtx.lineWidth = radius * 2;
    inpaintCtx.lineCap = 'round';
    inpaintCtx.lineJoin = 'round';
    inpaintCtx.beginPath();
    inpaintCtx.moveTo(a.x, a.y);
    inpaintCtx.lineTo(b.x, b.y);
    inpaintCtx.stroke();
    inpaintCtx.beginPath();
    inpaintCtx.arc(b.x, b.y, radius, 0, Math.PI * 2);
    inpaintCtx.fill();
}
if (inpaintCanvas) {
    inpaintCanvas.addEventListener('pointerdown', function(e) {
        e.preventDefault();
        painting = true;
        lastPt = inpaintPoint(e);
        inpaintStroke(lastPt, lastPt);
        try { inpaintCanvas.setPointerCapture(e.pointerId); } catch (err) {}
    });
    inpaintCanvas.addEventListener('pointermove', function(e) {
        if (!painting) return;
        e.preventDefault();
        const pt = inpaintPoint(e);
        inpaintStroke(lastPt, pt);
        lastPt = pt;
    });
    const endPaint = function() { painting = false; lastPt = null; };
    inpaintCanvas.addEventListener('pointerup', endPaint);
    inpaintCanvas.addEventListener('pointercancel', endPaint);
    inpaintCanvas.addEventListener('pointerleave', endPaint);
}

function inpaintActive() {
    return !!(inpaintMode && inpaintMode.checked && currentInputImages.length === 1 && window.__fluxVersion === 2);
}
// Returns a black/white mask data URL (white = regenerate), or null if nothing painted.
function getMaskDataURL() {
    if (!inpaintCanvas || !inpaintCtx || !inpaintCanvas.width) return null;
    const w = inpaintCanvas.width, h = inpaintCanvas.height;
    const src = inpaintCtx.getImageData(0, 0, w, h).data;
    const out = inpaintCtx.createImageData(w, h);
    const dst = out.data;
    let painted = false;
    for (let i = 0; i < src.length; i += 4) {
        const on = src[i + 3] > 10;  // any painted alpha
        if (on) painted = true;
        const v = on ? 255 : 0;
        dst[i] = v; dst[i + 1] = v; dst[i + 2] = v; dst[i + 3] = 255;
    }
    if (!painted) return null;
    const tmp = document.createElement('canvas');
    tmp.width = w; tmp.height = h;
    tmp.getContext('2d').putImageData(out, 0, 0);
    return tmp.toDataURL('image/png');
}

var resetBtn = document.getElementById('resetBtn');
const spectrumGridEl = document.getElementById('spectrumGrid');
const gridContainer = document.getElementById('gridContainer');
const gridSelector = document.getElementById('gridSelector');
const selectedCells = new Set();

// Initialize grid
if (gridSelector) {
    for (let i = 0; i < 16; i++) {
        const cell = document.createElement('div');
        cell.className = 'grid-cell';
        // Default to diagonals as before if user just turns it on
        const r = Math.floor(i / 4);
        const c = i % 4;
        if (r === c || r === (3 - c)) {
            cell.classList.add('selected');
            selectedCells.add(i);
        }
        cell.onclick = () => {
            cell.classList.toggle('selected');
            if (cell.classList.contains('selected')) {
                selectedCells.add(i);
            } else {
                selectedCells.delete(i);
            }
        };
        gridSelector.appendChild(cell);
    }
}

if (spectrumGridEl) spectrumGridEl.addEventListener('change', () => {
    gridContainer.style.display = spectrumGridEl.checked ? 'block' : 'none';
});

const allOrientationsEl = document.getElementById('allOrientations');
const orientationSelectEl = document.getElementById('orientation');
if (allOrientationsEl && orientationSelectEl) {
    allOrientationsEl.addEventListener('change', () => {
        orientationSelectEl.disabled = allOrientationsEl.checked;
        orientationSelectEl.style.opacity = allOrientationsEl.checked ? '0.5' : '';
    });
}

if (resetBtn) resetBtn.addEventListener('click', async function() {
    try {
        await fetch('/reset', { method: 'POST', headers: getAuthHeaders() });
    } catch (e) { console.warn('Reset request failed:', e); }
    var p = document.getElementById('prompt'); if (p) { p.value = ''; p.dispatchEvent(new Event('input')); }
    var o = document.getElementById('orientation'); if (o) { o.value = 'landscape'; o.disabled = false; o.style.opacity = ''; }
    var ao = document.getElementById('allOrientations'); if (ao) ao.checked = false;
    var s = document.getElementById('size'); if (s) s.value = '1mp';
    var st = document.getElementById('steps'); if (st) st.value = '25';
    var sd = document.getElementById('seed'); if (sd) sd.value = '';
    var gu = document.getElementById('guidance'); if (gu) gu.value = '4';
    var b = document.getElementById('batch'); if (b) b.value = '1';
    if (strengthSlider) strengthSlider.value = '0.5';
    if (strengthValue) strengthValue.textContent = '0.5';
    if (aspectModeEl) aspectModeEl.value = 'keep';
    clearRefs();
    var sg = document.getElementById('spectrumGrid'); if (sg) sg.checked = false;
    var sss = document.getElementById('spectrumSameSeed'); if (sss) sss.checked = false;
    var spv = document.getElementById('showPreview'); if (spv) spv.checked = true;
    var svp = document.getElementById('savePreviews'); if (svp) svp.checked = false;
    if (typeof syncSavePreviewsVisibility === 'function') syncSavePreviewsVisibility();
    if (gridContainer) gridContainer.style.display = 'none';
    selectedCells.clear();
    if (gridSelector) {
        Array.from(gridSelector.children).forEach((cell, i) => {
            const r = Math.floor(i / 4);
            const c = i % 4;
            if (r === c || r === (3 - c)) {
                cell.classList.add('selected');
                selectedCells.add(i);
            } else {
                cell.classList.remove('selected');
            }
        });
    }
    // Clear results and status
    if (imageGrid) imageGrid.innerHTML = '';
    updateResultCount();
    resultLbItems.length = 0;
    if (generationInfo) generationInfo.textContent = '';
    if (stepFrames) { stepFrames.style.display = 'none'; stepFrames.innerHTML = ''; }
    if (result) result.className = 'result';
    if (status) {
        status.className = 'status';
        if (statusText) statusText.textContent = 'Generating...';
    }
    if (knownImageFilenames) knownImageFilenames.clear();
    seenDoneJobIds.clear();
    lastCompletedJobId = null;
    // Hide the floating latest-image thumb; loadHistory re-shows it when the
    // next generation finishes.
    var lt = document.getElementById('latestThumb'); if (lt) lt.style.display = 'none';
    const pt = document.getElementById('progressTracker'); if (pt) pt.style.display = 'none';
    const pb = document.getElementById('progressBar'); if (pb) pb.style.width = '0%';
});

// Clearing empties the results view and tells the server to forget its
// recently-finished list; the generated files themselves are untouched and
// stay in Today's Generations.
async function clearRecentResults() {
    try {
        await fetch('/reset', { method: 'POST', headers: getAuthHeaders() });
    } catch (e) { console.warn('Clear recent request failed:', e); }
    if (imageGrid) imageGrid.innerHTML = '';
    updateResultCount();
    resultLbItems.length = 0;
    if (generationInfo) generationInfo.textContent = '';
    if (stepFrames) { stepFrames.style.display = 'none'; stepFrames.innerHTML = ''; }
    if (result) result.className = 'result';
    if (knownImageFilenames) knownImageFilenames.clear();
    seenDoneJobIds.clear();
    lastCompletedJobId = null;
}

const clearRecentBtn = document.getElementById('clearRecentBtn');
if (clearRecentBtn) clearRecentBtn.addEventListener('click', clearRecentResults);

// The same action on the expander's own summary row, so clearing doesn't mean
// opening the panel and scrolling past every image to reach the button at the
// bottom. Inside a <summary>, a click would also toggle the panel — hence the
// stopPropagation/preventDefault pair.
const clearRecentInline = document.getElementById('clearRecentInline');
if (clearRecentInline) clearRecentInline.addEventListener('click', function(e) {
    e.stopPropagation();
    e.preventDefault();
    clearRecentResults();
});

let lastCompletedJobId = null;
let seenDoneJobIds = new Set();
// Lightbox item list for the result grid; cleared wherever the grid clears
// so indices captured by the card click handlers stay valid.
const resultLbItems = [];

// ---- Interrupt (stop the running job) ----
// Two buttons share the handler: one in the main status panel, one in the
// fullscreen watch overlay.
const interruptBtn = document.getElementById('interruptBtn');
const watchInterruptBtn = document.getElementById('watchInterruptBtn');
let runningJobId = null;
let interruptRequested = false;  // optimistic UI until /status echoes cancel_requested

function setInterruptButton(btn, visible, stopping) {
    if (!btn) return;
    btn.style.display = visible ? 'inline-block' : 'none';
    btn.disabled = !!stopping;
    btn.textContent = stopping ? 'Stopping…' : '■ Interrupt';
}

async function requestInterrupt() {
    if (!runningJobId) return;
    interruptRequested = true;
    setInterruptButton(interruptBtn, true, true);
    setInterruptButton(watchInterruptBtn, true, true);
    try {
        const res = await fetch(`/jobs/${runningJobId}/cancel`, {
            method: 'POST',
            headers: getAuthHeaders(),
        });
        if (!res.ok) {
            const err = await res.json().catch(() => ({}));
            alert('Interrupt failed: ' + (err.error || res.status));
            interruptRequested = false;
        }
    } catch (e) {
        alert('Interrupt failed: ' + e.message);
        interruptRequested = false;
    }
    noteActivity();
    schedulePoll(0);
}

if (interruptBtn) interruptBtn.addEventListener('click', requestInterrupt);
if (watchInterruptBtn) watchInterruptBtn.addEventListener('click', requestInterrupt);

function renderQueueItem(job, position) {
    const safePrompt = (job.prompt || '(empty prompt)').replace(/</g, '&lt;').replace(/>/g, '&gt;');
    const metaParts = [];
    if (job.orientation) metaParts.push(job.orientation);
    if (job.size) metaParts.push(job.size);
    if (job.steps) metaParts.push(`${job.steps} steps`);
    if (job.batch && job.batch > 1) metaParts.push(`×${job.batch}`);
    if (job.refs > 0) metaParts.push(`${job.refs} ref${job.refs > 1 ? 's' : ''}`);
    if (job.spectrum_grid) metaParts.push('spectrum');
    const meta = metaParts.join(' · ');
    const el = document.createElement('div');
    el.className = 'queue-item';
    el.innerHTML = `
        <span class="pos">#${position}</span>
        <span class="prompt" title="${safePrompt}">${safePrompt}</span>
        <span class="meta">${meta}</span>
        <button class="cancel" data-job-id="${job.id}">Cancel</button>
    `;
    el.querySelector('.cancel').addEventListener('click', () => cancelQueuedJob(job.id));
    return el;
}

function renderQueue(queued) {
    const panel = document.getElementById('queuePanel');
    const list = document.getElementById('queueList');
    const count = document.getElementById('queueCount');
    if (!panel || !list || !count) return;
    if (!queued || queued.length === 0) {
        panel.style.display = 'none';
        list.innerHTML = '';
        return;
    }
    panel.style.display = 'block';
    count.textContent = `${queued.length} waiting`;
    list.innerHTML = '';
    queued.forEach((job, i) => list.appendChild(renderQueueItem(job, i + 1)));
}

async function cancelQueuedJob(jobId) {
    try {
        const res = await fetch(`/jobs/${jobId}/cancel`, {
            method: 'POST',
            headers: getAuthHeaders(),
        });
        if (!res.ok) {
            const err = await res.json().catch(() => ({}));
            console.warn('Cancel failed:', err.error || res.status);
        }
        noteActivity();
        schedulePoll(0);
    } catch (e) {
        console.error('Cancel error:', e);
    }
}

function renderRunning(running) {
    const progressTracker = document.getElementById('progressTracker');
    const progressBar = document.getElementById('progressBar');
    const progressText = document.getElementById('progressText');
    const pwrap = document.getElementById('previewWrap');
    const watchBtn = document.getElementById('watchBtn');

    if (!running) {
        // Don't hide the status bar while a VLM workflow (describe/boost/
        // evolve) is showing its own progress there.
        if (!vlmActiveCount) status.className = 'status';
        if (progressTracker) progressTracker.style.display = 'none';
        if (pwrap) pwrap.style.display = 'none';
        if (watchBtn) watchBtn.style.display = 'none';
        setInterruptButton(interruptBtn, false, false);
        setInterruptButton(watchInterruptBtn, false, false);
        runningJobId = null;
        interruptRequested = false;
        updateWatchOverlay(null);
        return;
    }

    status.className = 'status generating';
    if (watchBtn) watchBtn.style.display = 'inline-block';
    if (running.id !== runningJobId) {
        runningJobId = running.id;
        interruptRequested = false;
    }
    const stopping = interruptRequested || running.cancel_requested;
    setInterruptButton(interruptBtn, true, stopping);
    setInterruptButton(watchInterruptBtn, true, stopping);

    let stepInfo = '';
    if (running.total_steps > 0 && running.step > 0) {
        stepInfo = ` (step ${running.step} of ${running.total_steps})`;
    }
    const runBatch = Math.max(1, running.batch || 1);
    if (runBatch > 1) {
        statusText.textContent = `Generating: ${running.current} / ${runBatch}${stepInfo}...`;
    } else {
        statusText.textContent = running.step > 0
            ? `Generating: step ${running.step} of ${running.total_steps}...`
            : 'Generating...';
    }
    if (interruptRequested || running.cancel_requested) {
        statusText.textContent = 'Stopping…';
    }

    // Overall progress across the whole batch (0-100).
    let pct = 0;
    if (running.total_steps > 0) {
        const totalStepsAll = running.total_steps * runBatch;
        const stepsDone = Math.max(0, (running.current - 1)) * running.total_steps + running.step;
        pct = Math.min(100, Math.round((stepsDone / Math.max(1, totalStepsAll)) * 100));
    }

    if (running.total_steps > 0 && progressTracker && progressBar && progressText) {
        progressTracker.style.display = 'block';
        progressBar.style.width = `${pct}%`;
        progressText.textContent = `${pct}% complete`;
    } else if (progressTracker) {
        progressTracker.style.display = 'none';
    }

    if (running.preview && running.preview_ts && running.preview_ts !== lastPreviewStep) {
        const pimg = document.getElementById('previewImgLive');
        if (pwrap && pimg) {
            pimg.src = `/images/${running.preview}?t=${running.preview_ts}`;
            pwrap.style.display = 'block';
        }
        lastPreviewStep = running.preview_ts;
    }

    updateWatchOverlay(running, pct, runBatch, stepInfo);

    if (running.images && running.images.length > 0) {
        result.className = 'result visible';
        running.images.forEach((img, i) => {
            if (!knownImageFilenames.has(img.filename)) {
                addImageToGrid(img, i + 1);
                knownImageFilenames.add(img.filename);
            }
        });
    }
}

// "view frames" link on the completion line: lazily lists a job's saved
// intermediate frames (/steps/<id>) and toggles a filmstrip of thumbnails,
// each opening the full frame in a new tab.
const stepFrames = document.getElementById('stepFrames');
function attachStepFramesLink(job) {
    if (!stepFrames) return;
    stepFrames.style.display = 'none';
    stepFrames.innerHTML = '';
    if (!job.saved_previews || !generationInfo) return;
    const link = document.createElement('a');
    link.href = '#';
    link.className = 'step-frames-link';
    link.textContent = 'view frames';
    link.addEventListener('click', function(e) {
        e.preventDefault();
        if (stepFrames.style.display !== 'none') {
            stepFrames.style.display = 'none';
            return;
        }
        if (stepFrames.children.length) { stepFrames.style.display = 'flex'; return; }
        fetch('/steps/' + job.id, { headers: getAuthHeaders() })
            .then(function(r) { return r.json(); })
            .then(function(d) {
                if (!d.success) throw new Error(d.error || 'listing failed');
                (d.frames || []).forEach(function(p) {
                    const a = document.createElement('a');
                    a.href = '/images/' + p;
                    a.target = '_blank';
                    const im = document.createElement('img');
                    im.src = '/images/' + p;
                    im.loading = 'lazy';
                    im.title = p.split('/').pop();
                    a.appendChild(im);
                    stepFrames.appendChild(a);
                });
                stepFrames.style.display = 'flex';
            })
            .catch(function(err) { alert('Could not load frames: ' + err.message); });
    });
    generationInfo.appendChild(document.createTextNode(' '));
    generationInfo.appendChild(link);
}

function renderRecentDone(recent) {
    if (!recent || recent.length === 0) return;
    const latest = recent[0];
    // Append images from completed jobs we haven't rendered yet, oldest first
    // so the grid reads chronologically.
    for (let i = recent.length - 1; i >= 0; i--) {
        const job = recent[i];
        if (seenDoneJobIds.has(job.id)) continue;
        seenDoneJobIds.add(job.id);
        if (job.state !== 'done') continue;
        if (job.images) {
            job.images.forEach((img, idx) => {
                if (!knownImageFilenames.has(img.filename)) {
                    addImageToGrid(img, idx + 1);
                    knownImageFilenames.add(img.filename);
                    result.className = 'result visible';
                }
            });
        }
        if (job.composite && !knownImageFilenames.has(job.composite)) {
            addCompositeToGrid(job.composite);
            knownImageFilenames.add(job.composite);
            result.className = 'result visible';
        }
    }

    if (latest.id !== lastCompletedJobId) {
        lastCompletedJobId = latest.id;
        if (latest.state === 'done') {
            let info = latest.composite
                ? `Generated ${(latest.images || []).length} images + 1 composite in ${(latest.generation_time || 0).toFixed(1)}s`
                : `Generated ${(latest.images || []).length} image(s) in ${(latest.generation_time || 0).toFixed(1)}s`;
            if (latest.saved_previews) {
                info += ` — ${latest.saved_previews} preview frames saved`;
            }
            generationInfo.textContent = info;
            attachStepFramesLink(latest);
            loadHistory();
        } else if (latest.state === 'failed') {
            status.className = 'status error';
            statusText.textContent = 'Error: ' + (latest.error || 'Unknown error');
        } else if (latest.state === 'canceled') {
            generationInfo.textContent = 'Job canceled';
        }
    }
}

// Critique-VLM badge in the header: is the edit loop's vision model
// resident in ollama right now? (Value rides on /status; older servers
// don't send it, so the badge stays hidden there.)
function renderVlm(vlm) {
    const badge = document.getElementById('vlmBadge');
    if (!badge) return;
    if (!vlm || !vlm.status) { badge.style.display = 'none'; return; }
    const model = (vlm.model || 'VLM').replace(/:latest$/, '');
    const labels = {
        loaded: `🧠 ${model} ready`,
        loading: `🧠 ${model} loading…`,
        unloaded: `🧠 ${model} idle`,
        unavailable: '🧠 VLM unavailable'
    };
    badge.textContent = labels[vlm.status] || `🧠 ${model} ${vlm.status}`;
    badge.className = 'vlm-badge ' + vlm.status;
    badge.title = 'Edit-loop vision model (' + (vlm.model || '') + '): ' + vlm.status
        + (vlm.status === 'unloaded' ? ' — loads on the first critique' : '');
    badge.style.display = 'inline';
}

// GPU wattage badge in the lower-right corner (value rides on /status).
function renderPower(watts) {
    const badge = document.getElementById('powerBadge');
    const val = document.getElementById('powerWatts');
    if (!badge || !val) return;
    if (typeof watts === 'number') {
        val.textContent = Math.round(watts) + ' W';
        badge.style.display = 'block';
    } else {
        badge.style.display = 'none';
    }
}

async function pollStatus() {
    try {
        const response = await fetch('/status', { headers: getAuthHeaders() });

        if (response.status === 401) {
            pollStopped = true;  // un-latched when the API key input changes
            status.className = 'status error';
            statusText.textContent = 'Error: Unauthorized. Please check your API Key.';
            return;
        }

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        const data = await response.json();
        statusBusy = !!(data.running || (data.queued && data.queued.length));
        renderRunning(data.running);
        renderQueue(data.queued || []);
        renderRecentDone(data.recent_done || []);
        renderPower(data.power_w);
        renderVlm(data.vlm);
    } catch (err) {
        console.error('Polling error:', err);
    }
}

function addImageToGrid(img, index) {
    const t = Date.now();
    const meta = img.guidance != null ? `Guidance: ${img.guidance}${img.strength != null ? ', Strength: ' + img.strength : ''}` : '';
    const card = document.createElement('div');
    card.className = 'image-card';
    // Filename/seed are set via DOM APIs (not string-interpolated into
    // innerHTML/onclick) so unexpected characters can't break out of markup.
    card.innerHTML = `
        <img alt="Generated image ${index}" loading="lazy">
        <div class="actions">
            <a class="download-btn" href="#">Download</a>
            <a href="#" class="seed-btn">Use Seed</a>
            <a href="#" class="ref-btn">Use as Reference</a>
            <a href="#" class="save-hidden-btn">★ Save</a>
        </div>
        <p class="info">${meta ? meta + ' · Seed: ' + img.seed : 'Seed: ' + img.seed}</p>
        <div class="timings">
            <span class="timing-item"><span class="timing-label">Encode:</span> ${img.timings.encoding}s</span>
            <span class="timing-item"><span class="timing-label">Diffuse:</span> ${img.timings.diffusion}s</span>
            <span class="timing-item"><span class="timing-label">Save:</span> ${img.timings.save}s</span>
            <span class="timing-item timing-total"><span class="timing-label">Total:</span> ${img.timings.total}s</span>
        </div>
    `;
    const cardImg = card.querySelector('img');
    cardImg.src = `/images/${encodeURIComponent(img.filename)}?t=${t}`;
    const lbIdx = resultLbItems.length;
    resultLbItems.push({ src: `/images/${encodeURIComponent(img.filename)}`, caption: `Seed ${img.seed}` });
    cardImg.addEventListener('click', () => openLightbox(resultLbItems, lbIdx));
    const dl = card.querySelector('.download-btn');
    dl.href = `/images/${encodeURIComponent(img.filename)}`;
    dl.setAttribute('download', img.filename);
    card.querySelector('.seed-btn').addEventListener('click', (e) => { e.preventDefault(); useSeed(img.seed); });
    card.querySelector('.ref-btn').addEventListener('click', (e) => { e.preventDefault(); useAsReference(img.filename); });
    card.querySelector('.save-hidden-btn').addEventListener('click', function(e) { e.preventDefault(); saveHidden(this, img.filename); });
    imageGrid.appendChild(card);
    updateResultCount();
}

function addCompositeToGrid(filename) {
    const t = Date.now();
    const compositeCard = document.createElement('div');
    compositeCard.className = 'image-card composite-card';
    compositeCard.innerHTML = `
        <p class="composite-label">Matrix composite (guidance → columns, reference following → rows)</p>
        <img src="/images/${filename}?t=${t}" alt="Spectrum grid composite" class="composite-img">
        <div class="actions">
            <a href="/images/${filename}" download="${filename}">Download composite</a>
        </div>
    `;
    compositeCard.querySelector('img').addEventListener('click', () =>
        openLightbox([{ src: `/images/${filename}`, caption: 'Spectrum composite' }], 0));
    imageGrid.insertBefore(compositeCard, imageGrid.firstChild);
    updateResultCount();
}

// Snapshot the whole generation form (including inpaint state) into the JSON
// body /generate expects. Shared by doGenerate and the evolve path. Returns
// { formData, inpaintOn }, or null after reporting the error when inpaint is
// active but no mask has been painted.
function buildGenerateFormData() {
    const seedEl = document.getElementById('seed');
    const seedValue = seedEl ? seedEl.value.trim() : '';
    const promptEl = document.getElementById('prompt');
    const guidanceEl = document.getElementById('guidance');
    const batchEl = document.getElementById('batch');
    const orientationEl = document.getElementById('orientation');
    const sizeEl = document.getElementById('size');
    const stepsEl = document.getElementById('steps');
    const spectrumGridEl = document.getElementById('spectrumGrid');
    const spectrumGrid = spectrumGridEl ? spectrumGridEl.checked : false;
    const spectrumSameSeedEl = document.getElementById('spectrumSameSeed');
    const spectrumSameSeed = spectrumSameSeedEl ? spectrumSameSeedEl.checked : true;
    const showPreviewEl = document.getElementById('showPreview');
    const showPreview = showPreviewEl ? showPreviewEl.checked : false;
    const savePreviewsEl = document.getElementById('savePreviews');
    const savePreviews = showPreview && (savePreviewsEl ? savePreviewsEl.checked : false);

    const negativeEl = document.getElementById('negativePrompt');
    const baseFormData = {
        prompt: promptEl ? promptEl.value : '',
        // SDXL only; the field is hidden (and stays empty) on FLUX servers.
        negative_prompt: negativeEl && negativeEl.value.trim() ? negativeEl.value.trim() : null,
        orientation: orientationEl ? orientationEl.value : 'landscape',
        size: sizeEl ? sizeEl.value : '1mp',
        steps: stepsEl ? parseInt(stepsEl.value, 10) : 25,
        seed: seedValue ? parseInt(seedValue, 10) : null,
        guidance: guidanceEl && guidanceEl.value ? parseFloat(guidanceEl.value) : null,
        batch: batchEl ? parseInt(batchEl.value, 10) : 1,
        spectrum_grid: spectrumGrid,
        spectrum_same_seed: spectrumSameSeed,
        show_preview: showPreview,
        save_previews: savePreviews,
        selected_cells: Array.from(selectedCells)
    };
    if (currentInputImages.length > 0) {
        baseFormData.input_images = currentInputImages.slice(0, MAX_REFERENCE_IMAGES);
        // Legacy single-image field too, so this UI still works against an
        // older server that predates input_images (new servers ignore it).
        baseFormData.input_image = currentInputImages[0];
        if (strengthSlider) baseFormData.strength = parseFloat(strengthSlider.value);
        baseFormData.aspect_mode = aspectModeEl ? aspectModeEl.value : 'keep';
    }

    // Inpaint mode: attach the painted mask and lock to the source image
    // (single orientation, keep aspect, no spectrum sweep).
    let inpaintOn = false;
    if (typeof inpaintActive === 'function' && inpaintActive()) {
        const maskUrl = getMaskDataURL();
        if (!maskUrl) {
            status.className = 'status error';
            statusText.textContent = 'Inpaint: paint a region to regenerate first.';
            return null;
        }
        baseFormData.input_images = [currentInputImage];
        baseFormData.input_image = currentInputImage;  // legacy-server compat
        baseFormData.mask_image = maskUrl;
        baseFormData.aspect_mode = 'keep';
        baseFormData.spectrum_grid = false;
        // FLUX.2's masked diffusion ignores strength; SDXL inpainting uses it
        // as the denoise level for the masked region, so keep the slider value.
        if (window.__fluxVersion === 2) delete baseFormData.strength;
        inpaintOn = true;
    }
    return { formData: baseFormData, inpaintOn: inpaintOn };
}

async function doGenerate() {
    if (!submitBtn || !status || !statusText || !result || !imageGrid || !generationInfo) return;
    if (submitBtn.disabled) return;

    const built = buildGenerateFormData();
    if (!built) return;
    const baseFormData = built.formData;
    const inpaintOn = built.inpaintOn;
    recordPromptHistory(baseFormData.prompt);

    const allOrientationsEl = document.getElementById('allOrientations');
    const allOrientations = allOrientationsEl ? allOrientationsEl.checked : false;
    const orientationsToQueue = (allOrientations && !inpaintOn)
        ? ['square', 'landscape', 'portrait', 'widescreen', 'extra-tall']
        : [baseFormData.orientation];

    // Acknowledge the click before the POST round-trips. Without this the
    // button sits there looking dead until the server answers, and a job that
    // lands behind others in the queue never produces a visible "it worked"
    // moment at all — the two cases people read as "the button did nothing".
    const submitLabel = submitBtn.textContent;
    submitBtn.disabled = true;
    submitBtn.textContent = 'Queuing…';
    status.className = 'status generating';
    statusText.textContent = orientationsToQueue.length > 1
        ? `Submitting ${orientationsToQueue.length} jobs…`
        : 'Submitting…';
    noteActivity();

    let firstPosition = null;
    let submitted = 0;
    let errorMsg = null;
    try {
        for (const orient of orientationsToQueue) {
            const formData = Object.assign({}, baseFormData, { orientation: orient });
            const response = await fetch('/generate', {
                method: 'POST',
                headers: getAuthHeaders({ 'Content-Type': 'application/json' }),
                body: JSON.stringify(formData)
            });
            const data = await response.json().catch(() => ({}));
            if (response.ok && data.success) {
                submitted += 1;
                if (firstPosition === null) firstPosition = data.position;
            } else {
                errorMsg = data.error || `HTTP ${response.status}`;
                break;
            }
        }

        if (submitted > 0 && !errorMsg) {
            // Name the outcome explicitly. "Queued" with a position is the case
            // that most needs saying out loud: the job was accepted, it just
            // isn't the one generating yet.
            const ahead = firstPosition > 1 ? firstPosition - 1 : 0;
            const posMsg = orientationsToQueue.length > 1
                ? `Queued ${submitted} jobs (one per orientation)`
                : (ahead
                    ? `Queued — ${ahead} job${ahead > 1 ? 's' : ''} ahead of it`
                    : 'Queued — starting generation...');
            status.className = 'status generating';
            statusText.textContent = posMsg;
            noteActivity();
            schedulePoll(0);
        } else if (submitted > 0 && errorMsg) {
            status.className = 'status error';
            statusText.textContent = `Queued ${submitted}/${orientationsToQueue.length}; stopped: ${errorMsg}`;
            noteActivity();
            schedulePoll(0);
        } else {
            status.className = 'status error';
            statusText.textContent = 'Error: ' + (errorMsg || 'submission failed');
        }
    } catch (err) {
        status.className = 'status error';
        statusText.textContent = 'Error submitting generation: ' + err.message;
    } finally {
        submitBtn.disabled = false;
        submitBtn.textContent = submitLabel;
    }
}

// ---- Adaptive /status polling ----
// 1.5s while anything is active (running/queued job, edit loop, recent user
// activity); ~10s when idle; fully stopped while the tab is hidden. The old
// unconditional 1.5s interval was a real battery/heat cost on iPad/iPhone.
let pollTimer = null;
let pollStopped = false;   // latched on 401 until the API key changes
let statusBusy = false;    // last /status showed a running or queued job
let lastActivityTs = Date.now();

function pollDelay() {
    const active = statusBusy || loopRun || (Date.now() - lastActivityTs < 30000);
    return active ? 1500 : 10000;
}

function schedulePoll(delayMs) {
    if (pollTimer) clearTimeout(pollTimer);
    pollTimer = null;
    if (pollStopped || document.hidden) return;
    pollTimer = setTimeout(async function() {
        await pollStatus();
        schedulePoll();
    }, delayMs !== undefined ? delayMs : pollDelay());
}

function noteActivity() { lastActivityTs = Date.now(); }

// Polling a hidden tab is wasted work; catch up immediately on return.
document.addEventListener('visibilitychange', function() {
    if (document.hidden) {
        if (pollTimer) { clearTimeout(pollTimer); pollTimer = null; }
    } else {
        noteActivity();
        schedulePoll(0);
    }
});
// Any interaction keeps the fast cadence for 30s (adds no extra requests).
document.addEventListener('pointerdown', noteActivity, { passive: true });
document.addEventListener('keydown', noteActivity, { passive: true });
// Typing a new API key un-latches a 401 stop and retries promptly.
if (apiKeyInput) apiKeyInput.addEventListener('input', function() {
    if (pollStopped) { pollStopped = false; schedulePoll(800); }
});

pollStatus();
// Explicit first delay: pollDelay() reads loopRun, which is declared later
// in the file (TDZ at this point); reschedules run after full script load.
schedulePoll(1500);

if (submitBtn) submitBtn.addEventListener('click', function(e) { e.preventDefault(); doGenerate(); });
if (form) form.addEventListener('submit', function(e) { e.preventDefault(); doGenerate(); });

// ---- Prompt history (◀ ▶ above the prompt box) ----
// Shell-style: each submitted prompt appends to a localStorage-backed list.
// ◀ walks back through prior prompts; ▶ returns toward the in-progress
// draft, which is stashed when navigation leaves it. Editing while on a
// history entry forks that text into a new draft (replacing any stashed
// one); the history entry itself is untouched.
const PROMPT_HISTORY_KEY = 'flux_prompt_history';
const PROMPT_HISTORY_MAX = 100;
let promptHistory = [];
try { promptHistory = JSON.parse(localStorage.getItem(PROMPT_HISTORY_KEY) || '[]') || []; } catch (e) {}
let promptHistoryIdx = promptHistory.length;  // == length means "at the draft"
let promptDraft = '';

function syncPromptNav() {
    const prev = document.getElementById('promptPrevBtn');
    const next = document.getElementById('promptNextBtn');
    const pos = document.getElementById('promptHistoryPos');
    if (prev) prev.disabled = promptHistoryIdx <= 0;
    if (next) next.disabled = promptHistoryIdx >= promptHistory.length;
    if (pos) pos.textContent = promptHistoryIdx < promptHistory.length
        ? (promptHistoryIdx + 1) + '/' + promptHistory.length : '';
    const clear = document.getElementById('promptClearBtn');
    const copy = document.getElementById('promptCopyBtn');
    const ta = document.getElementById('prompt');
    if (clear) clear.disabled = !ta || ta.value.length === 0;
    if (copy) copy.disabled = !ta || ta.value.length === 0;
}

function promptHistoryGo(delta) {
    const ta = document.getElementById('prompt');
    if (!ta) return;
    const target = promptHistoryIdx + delta;
    if (target < 0 || target > promptHistory.length) return;
    if (promptHistoryIdx === promptHistory.length) promptDraft = ta.value;
    promptHistoryIdx = target;
    ta.value = promptHistoryIdx === promptHistory.length ? promptDraft : promptHistory[promptHistoryIdx];
    syncPromptNav();
}

function recordPromptHistory(text) {
    text = (text || '').trim();
    if (!text) return;
    if (promptHistory[promptHistory.length - 1] !== text) {
        promptHistory.push(text);
        if (promptHistory.length > PROMPT_HISTORY_MAX) promptHistory = promptHistory.slice(-PROMPT_HISTORY_MAX);
        try { localStorage.setItem(PROMPT_HISTORY_KEY, JSON.stringify(promptHistory)); } catch (e) {}
    }
    promptHistoryIdx = promptHistory.length;
    promptDraft = '';
    syncPromptNav();
}

(function() {
    const prev = document.getElementById('promptPrevBtn');
    const next = document.getElementById('promptNextBtn');
    const ta = document.getElementById('prompt');
    const clear = document.getElementById('promptClearBtn');
    const copy = document.getElementById('promptCopyBtn');
    if (prev) prev.addEventListener('click', function() { promptHistoryGo(-1); });
    if (next) next.addEventListener('click', function() { promptHistoryGo(1); });
    // Copy the prompt to the clipboard. navigator.clipboard needs a secure
    // context, which plain-http LAN access isn't, so fall back to selecting
    // the textarea and execCommand('copy').
    if (copy) copy.addEventListener('click', function() {
        if (!ta || !ta.value) return;
        function copied() {
            copy.textContent = 'Copied ✓';
            setTimeout(function() { copy.textContent = 'Copy'; }, 1200);
        }
        function fallbackCopy() {
            const start = ta.selectionStart, end = ta.selectionEnd;
            ta.select();
            try { if (document.execCommand('copy')) copied(); } catch (e) {}
            ta.setSelectionRange(start, end);
            ta.blur();
        }
        if (navigator.clipboard && window.isSecureContext) {
            navigator.clipboard.writeText(ta.value).then(copied, fallbackCopy);
        } else {
            fallbackCopy();
        }
    });
    // Clear just the prompt (not the whole form like Reset): empty the box,
    // drop back to an empty draft, and let the 'input' event refresh autogrow.
    if (clear) clear.addEventListener('click', function() {
        if (!ta) return;
        promptDraft = '';
        promptHistoryIdx = promptHistory.length;
        ta.value = '';
        ta.dispatchEvent(new Event('input'));
        // Also clear the negative prompt (SDXL) so Clear resets both boxes.
        const neg = document.getElementById('negativePrompt');
        if (neg) { neg.value = ''; neg.dispatchEvent(new Event('input')); }
        syncPromptNav();
        ta.focus();
    });
    // Typing while viewing a history entry forks it into the draft slot
    // (programmatic .value writes don't fire 'input', so navigation and
    // Boost rewrites don't trip this).
    if (ta) ta.addEventListener('input', function() {
        if (promptHistoryIdx < promptHistory.length) {
            promptHistoryIdx = promptHistory.length;
            promptDraft = ta.value;
        }
        syncPromptNav();
    });
    syncPromptNav();
})();
// Cmd/Ctrl+Return or Shift+Return: start the generation that's available in
// context — when the edit loop is paused between iterations, that's "Continue"
// (with the possibly-edited next instruction); otherwise queue a normal
// generation (works during a running job too — it just queues behind it).
// Shift+Return therefore no longer inserts a newline in the prompt textarea.
document.addEventListener('keydown', function(e) {
    if ((e.metaKey || e.ctrlKey || e.shiftKey) && e.key === 'Enter') {
        e.preventDefault();
        if (loopRun && loopRun.decision) loopRun.decision('continue');
        else doGenerate();
    }
});

// ---- Pixel compare: pick two same-resolution images (⧉ on history cards)
// and render only the pixels they share — per-channel RGB diff within the
// tolerance keeps image A's pixel, anything else goes transparent over the
// stage's checkerboard. The overlay is built here (not in the HTML) so the
// alternate layout, which shares this file, gets it for free. ----
const cmpOverlay = document.createElement('div');
cmpOverlay.className = 'cmp-overlay';
cmpOverlay.innerHTML = `
    <button type="button" class="cmp-close" title="Close (Esc)">✕</button>
    <div class="cmp-stage"><canvas></canvas></div>
    <div class="cmp-hud">
        <span class="cmp-stat"></span>
        <label>Tolerance <input type="range" min="0" max="48" step="1" value="8"></label>
        <span class="cmp-tol-val">8</span>
    </div>
    <p class="cmp-help">
        Only the pixels the two images agree on are drawn, taken from the
        first-picked image. A pixel is kept when its red, green and blue all
        differ by <b>&le; <span class="cmp-tol-echo">8</span> of 255</b> from
        the other image's — each channel is judged on its own, so one channel
        drifting too far drops the pixel. Everything else is transparent and
        shows the checkerboard. <b>0</b> keeps only exact matches; raising it
        forgives the sub-level drift between two runs of the same seed, and past
        ~24 it starts merging genuinely different content. Alpha is ignored, and
        the match percentage above moves with this slider.
    </p>`;
document.body.appendChild(cmpOverlay);
const cmpCanvas = cmpOverlay.querySelector('canvas');
const cmpStat = cmpOverlay.querySelector('.cmp-stat');
const cmpTol = cmpOverlay.querySelector('input[type="range"]');
const cmpTolVal = cmpOverlay.querySelector('.cmp-tol-val');
const cmpTolEcho = cmpOverlay.querySelector('.cmp-tol-echo');
let cmpArmed = null;   // {filename, btn} — the first of the two picks
let cmpData = null;    // {a, b: ImageData, w, h} while the overlay is open

function cmpReset() {
    if (cmpArmed) cmpArmed.btn.classList.remove('cmp-armed');
    cmpArmed = null;
}

function cmpClose() {
    cmpOverlay.classList.remove('visible');
    cmpData = null;
    cmpCanvas.width = cmpCanvas.height = 0;  // frees the decoded bitmap
    cmpReset();
}
cmpOverlay.querySelector('.cmp-close').addEventListener('click', cmpClose);
cmpOverlay.addEventListener('click', (e) => { if (e.target === cmpOverlay) cmpClose(); });
document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape' && cmpOverlay.classList.contains('visible')) cmpClose();
});

function cmpRender() {
    if (!cmpData) return;
    const tol = parseInt(cmpTol.value, 10);
    cmpTolVal.textContent = tol;
    cmpTolEcho.textContent = tol;
    const { a, b, w, h } = cmpData;
    const out = new ImageData(w, h);
    const pa = a.data, pb = b.data, po = out.data;
    let same = 0;
    for (let i = 0; i < pa.length; i += 4) {
        if (Math.abs(pa[i] - pb[i]) <= tol &&
            Math.abs(pa[i + 1] - pb[i + 1]) <= tol &&
            Math.abs(pa[i + 2] - pb[i + 2]) <= tol) {
            po[i] = pa[i]; po[i + 1] = pa[i + 1]; po[i + 2] = pa[i + 2]; po[i + 3] = 255;
            same++;
        }
    }
    cmpCanvas.getContext('2d').putImageData(out, 0, 0);
    const pct = (100 * same / (w * h)).toFixed(1);
    cmpStat.textContent = `${w}×${h} — ${pct}% of pixels match`;
}
cmpTol.addEventListener('input', () => {
    if (cmpRender._raf) cancelAnimationFrame(cmpRender._raf);
    cmpRender._raf = requestAnimationFrame(cmpRender);
});

function cmpLoadPixels(filename) {
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
        img.onerror = () => reject(new Error('failed to load ' + filename));
        img.src = `/images/${encodeURIComponent(filename)}`;
    });
}

async function cmpOpen(fileA, fileB) {
    let a, b;
    try {
        [a, b] = await Promise.all([cmpLoadPixels(fileA), cmpLoadPixels(fileB)]);
    } catch (err) {
        alert('Compare failed: ' + err.message);
        return;
    }
    if (a.width !== b.width || a.height !== b.height) {
        alert(`Compare needs two images of the same resolution — got ${a.width}×${a.height} and ${b.width}×${b.height}.`);
        return;
    }
    cmpData = { a, b, w: a.width, h: a.height };
    cmpCanvas.width = a.width;
    cmpCanvas.height = a.height;
    cmpOverlay.classList.add('visible');
    cmpRender();
}

function cmpPick(filename, btn) {
    if (cmpArmed && cmpArmed.btn === btn) { cmpReset(); return; }  // tap again to cancel
    if (!cmpArmed) {
        cmpArmed = { filename, btn };
        btn.classList.add('cmp-armed');
        return;
    }
    const first = cmpArmed.filename;
    cmpReset();
    cmpOpen(first, filename);
}

const historyGrid = document.getElementById('historyGrid');
const deleteAllBtn = document.getElementById('deleteAllBtn');
const latestThumb = document.getElementById('latestThumb');
const latestThumbImg = document.getElementById('latestThumbImg');
async function loadHistory() {
    try {
        const response = await fetch('/history', { headers: getAuthHeaders() });
        const data = await response.json();
        historyGrid.innerHTML = '';
        const hasImages = data.images.length > 0;
        const historyCount = document.getElementById('historyCount');
        if (historyCount) historyCount.textContent = hasImages ? `(${data.images.length})` : '';
        if (archiveBtn) archiveBtn.style.display = hasImages ? 'block' : 'none';
        if (deleteAllBtn) deleteAllBtn.style.display = hasImages ? 'block' : 'none';
        if (!hasImages) {
            historyGrid.innerHTML = '<p class="history-empty">No images generated today</p>';
            if (latestThumb) latestThumb.style.display = 'none';
            return;
        }
        const lbList = data.images.map(im => ({
            src: `/images/${encodeURIComponent(im.filename)}`,
            caption: (im.time ? im.time + ' — ' : '') + (im.prompt || im.filename)
        }));
        if (latestThumb && latestThumbImg) {
            latestThumbImg.src = lbList[0].src;
            latestThumb.title = lbList[0].caption;
            latestThumb.onclick = () => openLightbox(lbList, 0);
            latestThumb.style.display = 'block';
        }
        data.images.forEach((img, idx) => {
            const item = document.createElement('div');
            item.className = 'history-item';
            item.innerHTML = `
                <img loading="lazy">
                <button type="button" class="item-cmp" title="Compare: pick this and one more image">⧉</button>
                <button type="button" class="item-ref" title="Use as reference">↪</button>
                <button type="button" class="item-save" title="Save (survives housekeeping)">★</button>
                <button type="button" class="item-delete" title="Delete">X</button>
                <div class="overlay">
                    <span class="time"></span>
                </div>
            `;
            // Prompt and filename are set via DOM APIs so user-typed prompt
            // text can't break out of the markup.
            const thumb = item.querySelector('img');
            thumb.src = `/images/${encodeURIComponent(img.filename)}`;
            thumb.alt = img.prompt || 'Generated image';
            item.querySelector('.time').textContent = img.time;
            item.title = img.prompt || img.filename;
            item.addEventListener('click', (e) => {
                if (e.target.classList.contains('item-delete')) return;
                if (e.target.classList.contains('item-save')) return;
                if (e.target.classList.contains('item-ref')) return;
                if (e.target.classList.contains('item-cmp')) return;
                openLightbox(lbList, idx);
            });
            const refBtn = item.querySelector('.item-ref');
            if (refBtn) {
                refBtn.addEventListener('click', (e) => {
                    e.stopPropagation();
                    useAsReference(img.filename);
                });
            }
            const cmpBtn = item.querySelector('.item-cmp');
            if (cmpBtn) {
                cmpBtn.addEventListener('click', (e) => {
                    e.stopPropagation();
                    cmpPick(img.filename, cmpBtn);
                });
                // The grid rebuilds on every /history poll; carry an armed
                // pick over to the freshly created button.
                if (cmpArmed && cmpArmed.filename === img.filename) {
                    cmpArmed.btn = cmpBtn;
                    cmpBtn.classList.add('cmp-armed');
                }
            }
            const saveBtn = item.querySelector('.item-save');
            if (saveBtn) {
                saveBtn.addEventListener('click', (e) => {
                    e.stopPropagation();
                    saveHidden(saveBtn, img.filename);
                });
            }
            const delBtn = item.querySelector('.item-delete');
            if (delBtn) {
                delBtn.addEventListener('click', async (e) => {
                    e.stopPropagation();
                    if (!confirm(`Permanently delete this image?\n\n${img.filename}`)) return;
                    try {
                        const res = await fetch('/delete', {
                            method: 'POST',
                            headers: getAuthHeaders({ 'Content-Type': 'application/json' }),
                            body: JSON.stringify({ filename: img.filename })
                        });
                        const body = await res.json();
                        if (body.success) { loadHistory(); }
                        else { alert('Delete failed: ' + (body.error || 'unknown error')); }
                    } catch (err) { alert('Delete failed: ' + err.message); }
                });
            }
            historyGrid.appendChild(item);
        });
    } catch (err) {
        console.error('Failed to load history:', err);
        historyGrid.innerHTML = '<p class="history-empty">Failed to load history</p>';
    }
}

const archiveBtn = document.getElementById('archiveBtn');

if (archiveBtn) archiveBtn.addEventListener('click', async () => {
    if (!confirm("Move all of today's images to the archive folder?")) return;
    archiveBtn.disabled = true;
    archiveBtn.textContent = 'Archiving...';
    try {
        const response = await fetch('/archive', {
            method: 'POST',
            headers: getAuthHeaders()
        });
        const data = await response.json();
        if (data.success) { loadHistory(); } else { alert('Archive failed: ' + data.error); }
    } catch (err) { alert('Archive failed: ' + err.message); }
    archiveBtn.disabled = false;
    archiveBtn.textContent = 'Archive Today';
});

if (deleteAllBtn) deleteAllBtn.addEventListener('click', async () => {
    if (!confirm("Permanently DELETE all of today's generated images and prompt files? This cannot be undone.")) return;
    deleteAllBtn.disabled = true;
    deleteAllBtn.textContent = 'Deleting...';
    try {
        const response = await fetch('/delete', {
            method: 'POST',
            headers: getAuthHeaders({ 'Content-Type': 'application/json' }),
            body: JSON.stringify({})
        });
        const data = await response.json();
        if (data.success) { loadHistory(); } else { alert('Delete failed: ' + data.error); }
    } catch (err) { alert('Delete failed: ' + err.message); }
    deleteAllBtn.disabled = false;
    deleteAllBtn.textContent = 'Delete Today';
});

// Auto-grow the prompt field as you type (capped at 50vh, then scrolls).
// Especially helpful on phones where a fixed-height field is cramped.
const promptAutoGrow = document.getElementById('prompt');
if (promptAutoGrow) {
    const growPrompt = () => {
        promptAutoGrow.style.height = 'auto';
        const cap = window.innerHeight * 0.5;
        promptAutoGrow.style.height = Math.min(promptAutoGrow.scrollHeight, cap) + 'px';
    };
    promptAutoGrow.addEventListener('input', growPrompt);
    window.addEventListener('resize', growPrompt);
    growPrompt();
}

loadHistory();

// ---- Edit loop ----
// Iteratively: generate an edit from the first reference image, ask the
// server's /critique endpoint (local vision model) whether it landed, revise
// the instruction, repeat. Runs through the normal generation queue, so the
// main status panel shows live progress and outputs land in history as usual.
const loopStartBtn = document.getElementById('loopStartBtn');
const loopStopBtn = document.getElementById('loopStopBtn');
const loopContinueBtn = document.getElementById('loopContinueBtn');
const loopAcceptBtn = document.getElementById('loopAcceptBtn');
const loopStatusEl = document.getElementById('loopStatus');
const loopCards = document.getElementById('loopCards');
const loopControls = document.getElementById('loopControls');
const loopNextPrompt = document.getElementById('loopNextPrompt');

let loopRun = null;  // { stop, decision } — state of the active loop
// Lightbox item list for the loop cards; cleared when a new loop starts.
const loopLbItems = [];

function loopSetStatus(msg, cls) {
    if (!loopStatusEl) return;
    loopStatusEl.style.display = msg ? 'block' : 'none';
    loopStatusEl.textContent = msg || '';
    loopStatusEl.className = 'edit-loop-status' + (cls ? ' ' + cls : '');
}

async function loopSubmitJob(prompt, refDataUrl) {
    const stepsEl = document.getElementById('steps');
    const guidanceEl = document.getElementById('guidance');
    const showPreviewEl = document.getElementById('showPreview');
    const response = await fetch('/generate', {
        method: 'POST',
        headers: getAuthHeaders({ 'Content-Type': 'application/json' }),
        body: JSON.stringify({
            prompt: prompt,
            input_images: [refDataUrl],
            input_image: refDataUrl,   // legacy-server compat
            batch: 1,
            steps: stepsEl ? parseInt(stepsEl.value, 10) : 25,
            guidance: guidanceEl && guidanceEl.value ? parseFloat(guidanceEl.value) : null,
            aspect_mode: 'keep',
            show_preview: showPreviewEl ? showPreviewEl.checked : false
        })
    });
    const data = await response.json().catch(() => ({}));
    if (!response.ok || !data.success) throw new Error(data.error || `HTTP ${response.status}`);
    return data.job_id;
}

async function loopAwaitJob(jobId, onTick) {
    for (;;) {
        await new Promise(r => setTimeout(r, 2000));
        const res = await fetch('/status', { headers: getAuthHeaders() });
        const st = await res.json();
        const done = (st.recent_done || []).find(j => j.id === jobId);
        if (done) {
            if (done.state !== 'done' || done.error) {
                throw new Error(done.error || `job ${done.state}`);
            }
            return done.images[0].filename;
        }
        if (onTick) onTick(st);
        if (loopRun && loopRun.stop) {
            // Cancel if it's still queued; a running job has to finish on its own.
            await fetch('/jobs/' + jobId + '/cancel', {
                method: 'POST', headers: getAuthHeaders()
            }).catch(() => {});
            throw new Error('stopped');
        }
    }
}

// Slow VLM calls (/critique, /describe, /boost) can run for minutes — far past the
// ~60s cap Safari puts on a single fetch — so their POST endpoints return an
// id right away and the result is collected by polling. The optional
// `cancelled` callback aborts the wait; `onTick` fires once per poll cycle
// so callers can show live elapsed-time progress.
async function pollVlmJob(url, cancelled, onTick) {
    for (;;) {
        await new Promise(r => setTimeout(r, 2000));
        if (onTick) onTick();
        const res = await fetch(url, { headers: getAuthHeaders() });
        const data = await res.json().catch(() => ({}));
        if (!res.ok || !data.success) throw new Error(data.error || `HTTP ${res.status}`);
        if (data.done) return data;
        if (cancelled && cancelled()) throw new Error('stopped');
    }
}

function loopAwaitCritique(critiqueId, onTick) {
    return pollVlmJob('/critique/' + critiqueId, () => loopRun && loopRun.stop, onTick);
}

async function loopFetchAsDataUrl(filename) {
    const res = await fetch('/images/' + filename, { headers: getAuthHeaders() });
    if (!res.ok) throw new Error('HTTP ' + res.status);
    const blob = await res.blob();
    return await new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.onload = e => resolve(e.target.result);
        reader.onerror = reject;
        reader.readAsDataURL(blob);
    });
}

function loopAddCard(iter, prompt, filename) {
    const card = document.createElement('div');
    card.className = 'loop-card';
    const img = document.createElement('img');
    img.src = '/images/' + filename;
    img.alt = 'Iteration ' + iter;
    const lbIdx = loopLbItems.length;
    loopLbItems.push({ src: '/images/' + filename, caption: `Iteration ${iter} — ${prompt}` });
    img.addEventListener('click', () => openLightbox(loopLbItems, lbIdx));
    const body = document.createElement('div');
    body.className = 'loop-card-body';
    const title = document.createElement('div');
    title.className = 'loop-card-title';
    title.textContent = `Iteration ${iter}`;
    const promptLine = document.createElement('div');
    promptLine.className = 'loop-card-prompt';
    promptLine.textContent = prompt;
    const verdict = document.createElement('div');
    verdict.className = 'loop-card-verdict';
    verdict.textContent = 'Looking at the result…';
    // Backtrack: make this iteration's output the base for the next one
    // (e.g. go back two versions after a chain went off the rails).
    const baseBtn = document.createElement('button');
    baseBtn.type = 'button';
    baseBtn.className = 'loop-base-btn';
    baseBtn.textContent = '⏪ Base for next iteration';
    baseBtn.addEventListener('click', function() {
        if (!loopRun) return;
        loopRun.nextRefFilename = filename;
        document.querySelectorAll('.loop-card.base-selected')
            .forEach(c => c.classList.remove('base-selected'));
        card.classList.add('base-selected');
        loopSetStatus(`Next iteration will start from iteration ${iter}'s output.`);
    });
    body.appendChild(title);
    body.appendChild(promptLine);
    body.appendChild(verdict);
    body.appendChild(baseBtn);
    card.appendChild(img);
    card.appendChild(body);
    loopCards.appendChild(card);
    card.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    return verdict;
}

// Pause until the user picks: continue (with possibly edited instruction),
// accept, or stop. Resolves to 'continue' | 'accept' | 'stop'.
function loopAwaitDecision() {
    loopControls.style.display = 'block';
    return new Promise(resolve => {
        loopRun.decision = resolve;
    }).finally(() => {
        loopControls.style.display = 'none';
        if (loopRun) loopRun.decision = null;
    });
}

async function runEditLoop() {
    const direction = (document.getElementById('loopDirection').value || '').trim();
    const maxIter = parseInt(document.getElementById('loopIterations').value, 10);
    const auto = document.getElementById('loopAuto').checked;
    const chain = document.getElementById('loopChain').checked;

    if (!direction) { alert('Enter an edit direction first.'); return; }
    if (!currentInputImages.length) {
        alert('Upload a reference image first (the loop edits the first one).');
        return;
    }

    loopRun = { stop: false, decision: null, nextRefFilename: null };
    loopStartBtn.style.display = 'none';
    loopStopBtn.style.display = 'inline-block';
    loopCards.innerHTML = '';
    loopLbItems.length = 0;

    const originalRef = currentInputImages[0];
    let refDataUrl = originalRef;
    let prompt = direction;
    const completed = [];    // {filename, prompt} per finished iteration
    const critHistory = [];  // prompt trajectory sent to the critic each round
    let best = { score: -1, iter: 0 };  // best-graded iteration so far
    let makeStrip = false;   // set on Accept & finish or natural completion
    try {
        for (let i = 1; i <= maxIter && !loopRun.stop; i++) {
            loopSetStatus(`Iteration ${i}/${maxIter}: generating…`);
            const jobId = await loopSubmitJob(prompt, refDataUrl);
            const genT0 = Date.now();
            const filename = await loopAwaitJob(jobId, function(st) {
                const r = st.running;
                let phase = '';
                if (r && r.id === jobId && r.total_steps > 0 && r.step > 0) {
                    phase = ` — step ${r.step} of ${r.total_steps}`;
                } else if ((st.queued || []).some(q => q.id === jobId)) {
                    phase = ' — waiting in queue';
                }
                loopSetStatus(`Iteration ${i}/${maxIter}: generating${phase} — ${fmtElapsed(genT0)}`);
            });
            completed.push({ filename: filename, prompt: prompt });
            const verdictEl = loopAddCard(i, prompt, filename);

            loopSetStatus(`Iteration ${i}/${maxIter}: comparing input and output…`);
            let critique = null;
            try {
                const res = await fetch('/critique', {
                    method: 'POST',
                    headers: getAuthHeaders({ 'Content-Type': 'application/json' }),
                    body: JSON.stringify({
                        direction: direction,
                        prompt: prompt,
                        ref_image: refDataUrl,
                        output_filename: filename,
                        // Trajectory of earlier rounds so the critic doesn't
                        // re-propose phrasings that already failed.
                        history: critHistory
                    })
                });
                const submitted = await res.json().catch(() => ({}));
                if (!res.ok || !submitted.success) throw new Error(submitted.error || `HTTP ${res.status}`);
                const critT0 = Date.now();
                critique = await loopAwaitCritique(submitted.critique_id, function() {
                    loopSetStatus(`Iteration ${i}/${maxIter}: comparing input and output — ${fmtElapsed(critT0)}`);
                });
            } catch (err) {
                critique = null;
                verdictEl.textContent = 'Critique unavailable: ' + err.message;
            }

            let nextPrompt = prompt;
            if (critique) {
                const appliedTxt = critique.applied === null ? ''
                    : (critique.applied ? '✔ edit applied — ' : '✘ edit NOT applied — ');
                const scoreTxt = (typeof critique.score === 'number') ? ` (score ${critique.score}/10)` : '';
                verdictEl.textContent = appliedTxt + (critique.critique || '') + scoreTxt +
                    ' [' + (critique.metrics_text || '') + ']';
                verdictEl.classList.add(critique.applied === false ? 'not-applied' : 'applied');
                nextPrompt = critique.revised_prompt || prompt;
            }
            loopNextPrompt.value = nextPrompt;

            critHistory.push({
                prompt: prompt,
                applied: critique ? critique.applied : null,
                score: critique ? critique.score : null,
                critique: critique ? (critique.critique || '') : ''
            });
            if (critique && typeof critique.score === 'number' && critique.score > best.score) {
                best = { score: critique.score, iter: i };
            }

            if (loopRun.stop) break;
            // Early stop in auto mode: the critic says the goal landed.
            if (auto && critique && critique.applied &&
                typeof critique.score === 'number' && critique.score >= 8) {
                loopSetStatus(`Goal achieved at iteration ${i} (score ${critique.score}/10).`);
                makeStrip = true;
                break;
            }
            if (i === maxIter) { makeStrip = true; break; }

            if (auto) {
                prompt = nextPrompt;
            } else {
                loopSetStatus(`Iteration ${i}/${maxIter} done — adjust the next instruction or continue.`);
                const decision = await loopAwaitDecision();
                if (decision === 'accept') { makeStrip = true; break; }
                if (decision === 'stop') break;
                prompt = (loopNextPrompt.value || '').trim() || nextPrompt;
            }
            // Base for the next round: an explicit backtrack pick wins, then
            // chain mode follows the newest output, else stay on the original.
            if (loopRun.nextRefFilename) {
                refDataUrl = loopRun.nextRefFilename === '__original__'
                    ? originalRef
                    : await loopFetchAsDataUrl(loopRun.nextRefFilename);
                loopRun.nextRefFilename = null;
            } else if (chain) {
                refDataUrl = await loopFetchAsDataUrl(filename);
            }
        }

        if (makeStrip && completed.length) {
            loopSetStatus('Saving iterations and creating film strip…');
            try {
                const res = await fetch('/loop-strip', {
                    method: 'POST',
                    headers: getAuthHeaders({ 'Content-Type': 'application/json' }),
                    body: JSON.stringify({
                        direction: direction,
                        ref_image: originalRef,
                        filenames: completed.map(c => c.filename),
                        prompts: completed.map(c => c.prompt)
                    })
                });
                const body = await res.json();
                if (!res.ok || !body.success) throw new Error(body.error || `HTTP ${res.status}`);
                const stripCard = document.createElement('div');
                stripCard.className = 'loop-strip';
                const title = document.createElement('div');
                title.className = 'loop-card-title';
                title.textContent = 'Film strip — input plus each edit in sequence (iterations preserved in .saved)';
                const img = document.createElement('img');
                img.src = '/images/' + body.filename;
                img.alt = 'Edit loop film strip';
                img.addEventListener('click', () => openLightbox(
                    [{ src: '/images/' + body.filename, caption: 'Film strip — ' + direction }], 0));
                stripCard.appendChild(title);
                stripCard.appendChild(img);
                loopCards.appendChild(stripCard);
                stripCard.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
            } catch (err) {
                loopSetStatus('Film strip failed: ' + err.message, 'error');
            }
        }
        const bestTxt = best.score >= 0 ? ` Best result: iteration ${best.iter} (score ${best.score}/10).` : '';
        loopSetStatus((loopRun.stop ? 'Loop stopped.' : 'Loop finished — outputs are in the history below.') + bestTxt, 'done');
    } catch (err) {
        loopSetStatus(err.message === 'stopped' ? 'Loop stopped.' : 'Loop error: ' + err.message,
                      err.message === 'stopped' ? 'done' : 'error');
    } finally {
        loopControls.style.display = 'none';
        loopStopBtn.style.display = 'none';
        loopStartBtn.style.display = 'inline-block';
        loopRun = null;
        loadHistory();
    }
}

if (loopStartBtn) loopStartBtn.addEventListener('click', runEditLoop);
if (loopStopBtn) loopStopBtn.addEventListener('click', function() {
    if (!loopRun) return;
    loopRun.stop = true;
    if (loopRun.decision) loopRun.decision('stop');
    loopSetStatus('Stopping after the current step…');
});
if (loopContinueBtn) loopContinueBtn.addEventListener('click', function() {
    if (loopRun && loopRun.decision) loopRun.decision('continue');
});
const loopBaseOriginalBtn = document.getElementById('loopBaseOriginalBtn');
if (loopBaseOriginalBtn) loopBaseOriginalBtn.addEventListener('click', function() {
    if (!loopRun) return;
    loopRun.nextRefFilename = '__original__';
    document.querySelectorAll('.loop-card.base-selected')
        .forEach(c => c.classList.remove('base-selected'));
    loopSetStatus('Next iteration will start from the original image.');
});
if (loopAcceptBtn) loopAcceptBtn.addEventListener('click', function() {
    if (loopRun && loopRun.decision) loopRun.decision('accept');
});

// ---- Restore form state after a model switch ----
// Counterpart of saveSwitchState(): the model-switch reload lands here, and
// the snapshot (prompt, reference images, parameters) is put back so the
// working context survives the switch. One-shot: the key is cleared on read.
(function() {
    let saved = null;
    try {
        const raw = sessionStorage.getItem(SWITCH_STATE_KEY);
        if (!raw) return;
        sessionStorage.removeItem(SWITCH_STATE_KEY);
        saved = JSON.parse(raw);
    } catch (err) { return; }
    if (!saved) return;

    const setVal = function(id, v) {
        const el = document.getElementById(id);
        if (el && v !== null && v !== undefined) el.value = v;
    };
    const setChk = function(id, v) {
        const el = document.getElementById(id);
        if (el && typeof v === 'boolean') el.checked = v;
    };
    setVal('prompt', saved.prompt);
    setVal('negativePrompt', saved.negativePrompt);
    setVal('orientation', saved.orientation);
    setVal('size', saved.size);
    setVal('steps', saved.steps);
    setVal('seed', saved.seed);
    setVal('guidance', saved.guidance);
    setVal('batch', saved.batch);
    setVal('evolveCount', saved.evolveCount);
    setChk('allOrientations', saved.allOrientations);
    setChk('spectrumGrid', saved.spectrumGrid);
    setChk('spectrumSameSeed', saved.spectrumSameSeed);
    setChk('showPreview', saved.showPreview);
    setChk('savePreviews', saved.savePreviews);
    setChk('boostThink', saved.boostThink);
    setChk('describeThink', saved.describeThink);
    if (typeof syncSavePreviewsVisibility === 'function') syncSavePreviewsVisibility();
    if (strengthSlider && saved.strength !== null && saved.strength !== undefined) {
        strengthSlider.value = saved.strength;
        if (strengthValue) strengthValue.textContent = saved.strength;
    }
    if (aspectModeEl && saved.aspectMode) aspectModeEl.value = saved.aspectMode;

    // Re-apply the UI side effects the change handlers would have produced.
    if (allOrientationsEl && orientationSelectEl) {
        orientationSelectEl.disabled = allOrientationsEl.checked;
        orientationSelectEl.style.opacity = allOrientationsEl.checked ? '0.5' : '';
    }
    if (gridContainer && spectrumGridEl) {
        gridContainer.style.display = spectrumGridEl.checked ? 'block' : 'none';
    }
    if (Array.isArray(saved.cells) && gridSelector) {
        selectedCells.clear();
        Array.from(gridSelector.children).forEach(function(cell, i) {
            const on = saved.cells.indexOf(i) !== -1;
            cell.classList.toggle('selected', on);
            if (on) selectedCells.add(i);
        });
    }
    if (Array.isArray(saved.refImages) && saved.refImages.length > 0) {
        currentInputImages = saved.refImages.slice(0, MAX_REFERENCE_IMAGES);
        syncRefUI();
    }
})();


// ---- Multi-model comparison ----
// Run the prompt on a subset of the run_server.sh configs, one after another
// (POST /multi-run). The server orchestrates via a state file that survives
// the restart each model switch requires; this module just submits, polls
// GET /multi-run to render per-model progress/results, and hands off to
// watchServerRestart (overlay + reload) whenever the server goes down to
// load the next model. After the reload, the on-load block below resumes.
(function() {
    const section = document.getElementById('multiRunSection');
    const content = document.getElementById('multiRunContent');
    const configsWrap = document.getElementById('multiRunConfigs');
    const startBtn = document.getElementById('multiRunStartBtn');
    const cancelBtn = document.getElementById('multiRunCancelBtn');
    const statusEl = document.getElementById('multiRunStatus');
    const rowsEl = document.getElementById('multiRunRows');
    if (!section || !configsWrap || !startBtn || !cancelBtn || !statusEl || !rowsEl) return;

    let labels = {};          // config id (number) -> label
    let selectAllCb = null;   // the "Select all" checkbox, once configs render
    let mrPollTimer = null;
    let mrFailCount = 0;      // consecutive /multi-run fetch failures
    let mrNextLabel = null;   // label of the config the server switches to next
    let mrNextProgress = null; // "Model N of M" line for the restart overlay
    let mrActive = false;

    // Position of config `cid` within the run, for the switching overlay:
    // "Model 3 of 5 in this comparison run".
    function runProgress(run, cid) {
        if (!run || !run.configs) return null;
        const idx = run.configs.indexOf(cid);
        if (idx < 0) return null;
        return 'Model ' + (idx + 1) + ' of ' + run.configs.length + ' in this comparison run';
    }

    fetch('/configs', { headers: getAuthHeaders() })
        .then(function(r) { if (!r.ok) throw new Error(r.status); return r.json(); })
        .then(function(data) {
            if (!data.switchable || !data.configs) { section.style.display = 'none'; return; }
            const allLab = document.createElement('label');
            allLab.className = 'checkbox-label multi-run-select-all';
            selectAllCb = document.createElement('input');
            selectAllCb.type = 'checkbox';
            allLab.appendChild(selectAllCb);
            allLab.appendChild(document.createTextNode(' Select all'));
            configsWrap.appendChild(allLab);
            data.configs.forEach(function(c) {
                labels[c.id] = c.label;
                const lab = document.createElement('label');
                lab.className = 'checkbox-label';
                const cb = document.createElement('input');
                cb.type = 'checkbox';
                cb.className = 'multi-run-cb';
                cb.value = String(c.id);
                lab.appendChild(cb);
                lab.appendChild(document.createTextNode(
                    ' ' + c.id + ' — ' + c.label + (c.id === data.current ? ' (current)' : '')));
                configsWrap.appendChild(lab);
            });
        })
        .catch(function() { section.style.display = 'none'; });

    function configCheckboxes() {
        return Array.from(configsWrap.querySelectorAll('input.multi-run-cb'));
    }
    function selectedIds() {
        return configCheckboxes().filter(function(cb) { return cb.checked; })
            .map(function(cb) { return parseInt(cb.value, 10); });
    }
    configsWrap.addEventListener('change', function(e) {
        const boxes = configCheckboxes();
        if (selectAllCb && e.target === selectAllCb) {
            boxes.forEach(function(cb) { cb.checked = selectAllCb.checked; });
        } else if (selectAllCb) {
            const checked = boxes.filter(function(cb) { return cb.checked; }).length;
            selectAllCb.checked = checked === boxes.length;
            selectAllCb.indeterminate = checked > 0 && checked < boxes.length;
        }
        startBtn.disabled = mrActive || selectedIds().length === 0;
    });

    function setStatus(msg, cls) {
        statusEl.style.display = msg ? 'block' : 'none';
        statusEl.textContent = msg || '';
        statusEl.className = 'edit-loop-status' + (cls ? ' ' + cls : '');
    }

    function stopPolling() {
        if (mrPollTimer) { clearTimeout(mrPollTimer); mrPollTimer = null; }
    }

    function render(data) {
        const run = data.run;
        if (!run) {
            mrActive = false;
            cancelBtn.style.display = 'none';
            rowsEl.innerHTML = '';
            setStatus(null);
            startBtn.disabled = selectedIds().length === 0;
            return;
        }
        mrActive = data.active;
        startBtn.disabled = mrActive || selectedIds().length === 0;
        cancelBtn.style.display = '';
        cancelBtn.textContent = mrActive ? 'Cancel run' : 'Dismiss results';
        // The label the restart watcher shows when the server goes down: the
        // first config still pending other than the one generating right now.
        const pendingOther = run.configs.find(function(c) {
            return c !== data.current_config &&
                !run.results.some(function(r) { return r.config === c; });
        });
        mrNextLabel = pendingOther != null ? (labels[pendingOther] || ('config ' + pendingOther)) : null;
        mrNextProgress = pendingOther != null ? runProgress(run, pendingOther) : null;

        if (mrActive) {
            const doneCount = run.results.length;
            setStatus('Running on ' + run.configs.length + ' models — ' + doneCount + ' done. Seed ' +
                      run.params.seed + '.', '');
        } else if (run.canceled) {
            setStatus('Run canceled.', 'error');
        } else {
            setStatus('Run complete. Seed ' + run.params.seed + '.', 'done');
        }

        // One lightbox across every image of the run, captioned by model.
        const lbItems = [];
        run.results.forEach(function(r) {
            (r.images || []).forEach(function(img) {
                lbItems.push({ src: '/images/' + img.filename,
                               caption: r.label + ' — seed ' + img.seed });
            });
        });

        rowsEl.innerHTML = '';
        let lbIdx = 0;
        run.configs.forEach(function(cid) {
            const res = run.results.find(function(r) { return r.config === cid; });
            const row = document.createElement('div');
            row.className = 'mr-row';
            const name = document.createElement('div');
            name.className = 'mr-row-label';
            name.textContent = (res && res.label) || labels[cid] || ('config ' + cid);
            row.appendChild(name);
            const state = document.createElement('div');
            state.className = 'mr-row-state';
            if (res) {
                if (res.state === 'done') {
                    state.textContent = '✓ ' + (res.generation_time ? res.generation_time.toFixed(1) + 's' : 'done');
                    state.classList.add('done');
                } else {
                    state.textContent = '✗ ' + (res.error || res.state);
                    state.classList.add('failed');
                }
            } else if (!mrActive) {
                state.textContent = 'not run';
            } else if (cid === data.current_config) {
                state.textContent = 'generating…';
                state.classList.add('active');
            } else if (cid === data.next_config) {
                state.textContent = 'next — switching model…';
                state.classList.add('active');
            } else {
                state.textContent = 'waiting';
            }
            row.appendChild(state);
            if (res && res.images && res.images.length) {
                const thumbs = document.createElement('div');
                thumbs.className = 'mr-thumbs';
                res.images.forEach(function(img) {
                    const t = document.createElement('img');
                    t.className = 'mr-thumb';
                    t.src = '/images/' + img.filename;
                    t.alt = name.textContent;
                    const idx = lbIdx++;
                    t.addEventListener('click', function() { openLightbox(lbItems, idx); });
                    thumbs.appendChild(t);
                });
                row.appendChild(thumbs);
            }
            rowsEl.appendChild(row);
        });
    }

    async function poll() {
        try {
            const res = await fetch('/multi-run', { headers: getAuthHeaders(), cache: 'no-store' });
            if (res.status === 401) { stopPolling(); return; }
            const data = await res.json();
            mrFailCount = 0;
            render(data);
            // A pending config differing from the current one means the server
            // is restarting (or about to) into the next model. Detect it from
            // the state rather than waiting to catch the brief down-window —
            // Flask is back up in seconds while the model load takes minutes,
            // so the failure path below usually never fires.
            if (data.active && data.next_config != null &&
                data.next_config !== data.current_config) {
                stopPolling();
                watchServerRestart(labels[data.next_config] || ('config ' + data.next_config),
                                   runProgress(data.run, data.next_config));
                return;
            }
            if (data.active) {
                mrPollTimer = setTimeout(poll, 2000);
            } else {
                stopPolling();
            }
        } catch (err) {
            // The server has likely exited to load the next model. Require two
            // consecutive failures (vs a transient blip) before handing off to
            // the restart watcher, which overlays and reloads when it's back.
            mrFailCount += 1;
            if (mrFailCount >= 2) {
                stopPolling();
                watchServerRestart(mrNextLabel || 'next model', mrNextProgress);
            } else {
                mrPollTimer = setTimeout(poll, 2000);
            }
        }
    }

    startBtn.addEventListener('click', async function() {
        const ids = selectedIds();
        if (!ids.length || startBtn.disabled) return;
        if (currentInputImages.length > 0) {
            setStatus('Multi-model runs are text-to-image only — remove the reference images first.', 'error');
            return;
        }
        const built = buildGenerateFormData();
        if (!built) return;
        const f = built.formData;
        if (!f.prompt || !f.prompt.trim()) {
            setStatus('Enter a prompt first.', 'error');
            return;
        }
        startBtn.disabled = true;
        try {
            const res = await fetch('/multi-run', {
                method: 'POST',
                headers: getAuthHeaders({ 'Content-Type': 'application/json' }),
                body: JSON.stringify({
                    configs: ids,
                    prompt: f.prompt,
                    orientation: f.orientation,
                    size: f.size,
                    steps: f.steps,
                    seed: f.seed,
                    guidance: f.guidance,
                    batch: f.batch,
                    show_preview: f.show_preview
                })
            });
            const data = await res.json().catch(function() { return {}; });
            if (!res.ok || !data.success) throw new Error(data.error || ('HTTP ' + res.status));
            recordPromptHistory(f.prompt);
            noteActivity();
            schedulePoll(0);  // the run's jobs show as normal jobs in the status area
            mrFailCount = 0;
            poll();
        } catch (err) {
            setStatus('Could not start run: ' + err.message, 'error');
            startBtn.disabled = selectedIds().length === 0;
        }
    });

    cancelBtn.addEventListener('click', async function() {
        try {
            await fetch('/multi-run/cancel', { method: 'POST', headers: getAuthHeaders() });
        } catch (err) {}
        stopPolling();
        render({ run: null });
    });

    // On load: resume/display an existing run (e.g. right after the reload a
    // mid-run model switch causes).
    fetch('/multi-run', { headers: getAuthHeaders(), cache: 'no-store' })
        .then(function(r) { if (!r.ok) throw new Error(r.status); return r.json(); })
        .then(function(data) {
            if (!data.run) return;
            if (content) content.style.display = 'block';
            render(data);
            if (data.active) {
                // If the page came up mid-run while a model is still loading,
                // the plain "Loading model…" overlay is showing — give it the
                // run's progress line too.
                const progressEl = document.getElementById('loadingProgress');
                const p = runProgress(data.run, data.current_config);
                if (progressEl && p) {
                    progressEl.textContent = p;
                    progressEl.style.display = '';
                }
                mrFailCount = 0;
                mrPollTimer = setTimeout(poll, 2000);
            }
        })
        .catch(function() {});
})();
