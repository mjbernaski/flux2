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

const form = document.getElementById('generateForm');
const submitBtn = document.getElementById('submitBtn');
const status = document.getElementById('status');
const statusText = document.getElementById('statusText');
const result = document.getElementById('result');
const imageGrid = document.getElementById('imageGrid');
const generationInfo = document.getElementById('generationInfo');

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

let pollInterval = null;
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
    if (e.key === 'Escape' && watchOpen) closeWatch();
});
// If the user leaves browser fullscreen (Esc/F11), drop our overlay too.
document.addEventListener('fullscreenchange', function() {
    if (!document.fullscreenElement && watchOpen) closeWatch();
});

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

// Copy an image into the server-side .hidden subdir (preserves it from
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
                if (labeled) el.textContent = '★ Saved'; else el.title = 'Saved to hidden';
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
    if (uploadPlaceholder) {
        const span = uploadPlaceholder.querySelector('span');
        if (span) span.textContent = n === 0
            ? 'Click or drag up to 3 images here'
            : `+ Add image (${n}/${MAX_REFERENCE_IMAGES})`;
    }
    // Strength only applies to single-image FLUX.1 img2img; multi-reference
    // runs through Kontext/FLUX.2 conditioning, which ignores it.
    if (strengthControl) strengthControl.style.display = n === 1 ? 'flex' : 'none';
    if (aspectModeControl) aspectModeControl.style.display = n > 0 ? 'block' : 'none';
    if (multiRefHint) multiRefHint.style.display = n > 1 ? 'block' : 'none';
    if (inputImage) inputImage.value = '';
    if (typeof refreshInpaintAvailability === 'function') refreshInpaintAvailability();
}

function handleImageFile(file) {
    if (currentInputImages.length >= MAX_REFERENCE_IMAGES) return;
    var reader = new FileReader();
    reader.onload = function(e) {
        if (currentInputImages.length >= MAX_REFERENCE_IMAGES) return;
        currentInputImages.push(e.target.result);
        syncRefUI();
    };
    reader.readAsDataURL(file);
}

function addImageFiles(fileList) {
    Array.from(fileList || [])
        .filter(function(f) { return f && f.type.indexOf('image/') === 0; })
        .slice(0, Math.max(0, MAX_REFERENCE_IMAGES - currentInputImages.length))
        .forEach(handleImageFile);
}

function clearRefs() {
    currentInputImages = [];
    syncRefUI();
    if (typeof resetInpaint === 'function') resetInpaint();
}

if (uploadArea) {
    uploadArea.addEventListener('click', function() { if (inputImage) inputImage.click(); });
    uploadArea.addEventListener('dragover', function(e) { e.preventDefault(); uploadArea.classList.add('dragover'); });
    uploadArea.addEventListener('dragleave', function() { uploadArea.classList.remove('dragover'); });
    uploadArea.addEventListener('drop', function(e) {
        e.preventDefault();
        uploadArea.classList.remove('dragover');
        addImageFiles(e.dataTransfer.files);
    });
}
if (inputImage) inputImage.addEventListener('change', function(e) { addImageFiles(e.target.files); });

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
    if (generationInfo) generationInfo.textContent = '';
    if (result) result.className = 'result';
    if (status) {
        status.className = 'status';
        if (statusText) statusText.textContent = 'Generating...';
    }
    if (knownImageFilenames) knownImageFilenames.clear();
    seenDoneJobIds.clear();
    lastCompletedJobId = null;
    const pt = document.getElementById('progressTracker'); if (pt) pt.style.display = 'none';
    const pb = document.getElementById('progressBar'); if (pb) pb.style.width = '0%';
});

const clearRecentBtn = document.getElementById('clearRecentBtn');
if (clearRecentBtn) clearRecentBtn.addEventListener('click', async function() {
    try {
        await fetch('/reset', { method: 'POST', headers: getAuthHeaders() });
    } catch (e) { console.warn('Clear recent request failed:', e); }
    if (imageGrid) imageGrid.innerHTML = '';
    if (generationInfo) generationInfo.textContent = '';
    if (result) result.className = 'result';
    if (knownImageFilenames) knownImageFilenames.clear();
    seenDoneJobIds.clear();
    lastCompletedJobId = null;
});

let lastCompletedJobId = null;
let seenDoneJobIds = new Set();

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
        pollStatus();
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
        status.className = 'status';
        if (progressTracker) progressTracker.style.display = 'none';
        if (pwrap) pwrap.style.display = 'none';
        if (watchBtn) watchBtn.style.display = 'none';
        updateWatchOverlay(null);
        return;
    }

    status.className = 'status generating';
    if (watchBtn) watchBtn.style.display = 'inline-block';

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
            const info = latest.composite
                ? `Generated ${(latest.images || []).length} images + 1 composite in ${(latest.generation_time || 0).toFixed(1)}s`
                : `Generated ${(latest.images || []).length} image(s) in ${(latest.generation_time || 0).toFixed(1)}s`;
            generationInfo.textContent = info;
            loadHistory();
        } else if (latest.state === 'failed') {
            status.className = 'status error';
            statusText.textContent = 'Error: ' + (latest.error || 'Unknown error');
        } else if (latest.state === 'canceled') {
            generationInfo.textContent = 'Job canceled';
        }
    }
}

async function pollStatus() {
    try {
        const response = await fetch('/status', { headers: getAuthHeaders() });

        if (response.status === 401) {
            if (pollInterval) { clearInterval(pollInterval); pollInterval = null; }
            status.className = 'status error';
            statusText.textContent = 'Error: Unauthorized. Please check your API Key.';
            return;
        }

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        const data = await response.json();
        renderRunning(data.running);
        renderQueue(data.queued || []);
        renderRecentDone(data.recent_done || []);
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
    card.querySelector('img').src = `/images/${encodeURIComponent(img.filename)}?t=${t}`;
    const dl = card.querySelector('.download-btn');
    dl.href = `/images/${encodeURIComponent(img.filename)}`;
    dl.setAttribute('download', img.filename);
    card.querySelector('.seed-btn').addEventListener('click', (e) => { e.preventDefault(); useSeed(img.seed); });
    card.querySelector('.ref-btn').addEventListener('click', (e) => { e.preventDefault(); useAsReference(img.filename); });
    card.querySelector('.save-hidden-btn').addEventListener('click', function(e) { e.preventDefault(); saveHidden(this, img.filename); });
    imageGrid.appendChild(card);
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
    imageGrid.insertBefore(compositeCard, imageGrid.firstChild);
}

async function doGenerate() {
    if (!submitBtn || !status || !statusText || !result || !imageGrid || !generationInfo) return;
    if (submitBtn.disabled) return;

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
    const allOrientationsEl = document.getElementById('allOrientations');
    const allOrientations = allOrientationsEl ? allOrientationsEl.checked : false;

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
            return;
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

    const orientationsToQueue = (allOrientations && !inpaintOn)
        ? ['square', 'landscape', 'portrait', 'widescreen', 'extra-tall']
        : [baseFormData.orientation];

    // Briefly disable to prevent double-submit during the fetch; re-enable on response.
    submitBtn.disabled = true;

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
            const posMsg = orientationsToQueue.length > 1
                ? `Queued ${submitted} jobs (one per orientation)`
                : (firstPosition > 1 ? `Queued at position ${firstPosition}` : 'Starting generation...');
            status.className = 'status generating';
            statusText.textContent = posMsg;
            pollStatus();
        } else if (submitted > 0 && errorMsg) {
            status.className = 'status error';
            statusText.textContent = `Queued ${submitted}/${orientationsToQueue.length}; stopped: ${errorMsg}`;
            pollStatus();
        } else {
            status.className = 'status error';
            statusText.textContent = 'Error: ' + (errorMsg || 'submission failed');
        }
    } catch (err) {
        status.className = 'status error';
        statusText.textContent = 'Error submitting generation: ' + err.message;
    } finally {
        submitBtn.disabled = false;
    }
}

// Handle visibility change for mobile robustness
document.addEventListener('visibilitychange', function() {
    if (document.visibilityState === 'visible') pollStatus();
});

// Always poll so the queue panel and completed jobs update in real time.
pollStatus();
pollInterval = setInterval(pollStatus, 1500);

if (submitBtn) submitBtn.addEventListener('click', function(e) { e.preventDefault(); doGenerate(); });
if (form) form.addEventListener('submit', function(e) { e.preventDefault(); doGenerate(); });
document.addEventListener('keydown', function(e) {
    if ((e.metaKey || e.ctrlKey) && e.key === 'Enter') {
        e.preventDefault();
        doGenerate();
    }
});

const historyGrid = document.getElementById('historyGrid');
const deleteAllBtn = document.getElementById('deleteAllBtn');
async function loadHistory() {
    try {
        const response = await fetch('/history', { headers: getAuthHeaders() });
        const data = await response.json();
        historyGrid.innerHTML = '';
        const hasImages = data.images.length > 0;
        if (archiveBtn) archiveBtn.style.display = hasImages ? 'block' : 'none';
        if (deleteAllBtn) deleteAllBtn.style.display = hasImages ? 'block' : 'none';
        if (!hasImages) {
            historyGrid.innerHTML = '<p class="history-empty">No images generated today</p>';
            return;
        }
        data.images.forEach(img => {
            const item = document.createElement('div');
            item.className = 'history-item';
            item.innerHTML = `
                <img loading="lazy">
                <button type="button" class="item-ref" title="Use as reference">↪</button>
                <button type="button" class="item-save" title="Save to hidden">★</button>
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
                window.open(`/images/${img.filename}`, '_blank');
            });
            const refBtn = item.querySelector('.item-ref');
            if (refBtn) {
                refBtn.addEventListener('click', (e) => {
                    e.stopPropagation();
                    useAsReference(img.filename);
                });
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

async function loopAwaitJob(jobId) {
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
        if (loopRun && loopRun.stop) {
            // Cancel if it's still queued; a running job has to finish on its own.
            await fetch('/jobs/' + jobId + '/cancel', {
                method: 'POST', headers: getAuthHeaders()
            }).catch(() => {});
            throw new Error('stopped');
        }
    }
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
    img.addEventListener('click', () => window.open('/images/' + filename, '_blank'));
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

    const originalRef = currentInputImages[0];
    let refDataUrl = originalRef;
    let prompt = direction;
    const completed = [];   // {filename, prompt} per finished iteration
    let makeStrip = false;  // set on Accept & finish or natural completion
    try {
        for (let i = 1; i <= maxIter && !loopRun.stop; i++) {
            loopSetStatus(`Iteration ${i}/${maxIter}: generating…`);
            const jobId = await loopSubmitJob(prompt, refDataUrl);
            const filename = await loopAwaitJob(jobId);
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
                        output_filename: filename
                    })
                });
                critique = await res.json();
                if (!res.ok || !critique.success) throw new Error(critique.error || `HTTP ${res.status}`);
            } catch (err) {
                critique = null;
                verdictEl.textContent = 'Critique unavailable: ' + err.message;
            }

            let nextPrompt = prompt;
            if (critique) {
                const appliedTxt = critique.applied === null ? ''
                    : (critique.applied ? '✔ edit applied — ' : '✘ edit NOT applied — ');
                verdictEl.textContent = appliedTxt + (critique.critique || '') +
                    ' [' + (critique.metrics_text || '') + ']';
                verdictEl.classList.add(critique.applied === false ? 'not-applied' : 'applied');
                nextPrompt = critique.revised_prompt || prompt;
            }
            loopNextPrompt.value = nextPrompt;

            if (loopRun.stop) break;
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
                title.textContent = 'Film strip — input plus each edit in sequence (iterations preserved in .hidden)';
                const img = document.createElement('img');
                img.src = '/images/' + body.filename;
                img.alt = 'Edit loop film strip';
                img.addEventListener('click', () => window.open('/images/' + body.filename, '_blank'));
                stripCard.appendChild(title);
                stripCard.appendChild(img);
                loopCards.appendChild(stripCard);
                stripCard.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
            } catch (err) {
                loopSetStatus('Film strip failed: ' + err.message, 'error');
            }
        }
        loopSetStatus(loopRun.stop ? 'Loop stopped.' : 'Loop finished — outputs are in the history below.', 'done');
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

