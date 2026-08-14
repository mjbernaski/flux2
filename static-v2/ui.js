/* Compact FLUX UI — core loop.
 *
 * Speaks only /api/v1, same-origin: new_ui.py proxies it to the generator on
 * 2222 and supplies the API key, so nothing here needs a key unless the user
 * sets one (see keyModal), in which case it wins over the server's.
 *
 * Pass 2 will wire the prompt tools (boost / boost-xN / evolve / describe),
 * the inpaint mask, the edit loop and the multi-model compare. Their controls
 * exist in the markup and are disabled by markPending() below rather than
 * hidden, so the layout is the finished one.
 */

const $  = (id) => document.getElementById(id);
const el = (tag, cls, text) => {
    const n = document.createElement(tag);
    if (cls) n.className = cls;
    if (text != null) n.textContent = text;
    return n;
};

const MAX_REFS = 3;
const TERMINAL = ['done', 'failed', 'canceled'];

const state = {
    caps: {},              // GET /model
    refs: [],              // [{dataUrl, name}]
    promptHistory: JSON.parse(localStorage.getItem('flux_prompts') || '[]'),
    historyPos: -1,        // -1 = live draft, 0..n-1 = walking back
    draft: '',
    activeJob: null,       // id of the job we are following
    gallery: [],           // history entries, for lightbox navigation
    lightboxIndex: 0,
    spectrumCells: new Set(),
};

/* ── transport ──────────────────────────────────────────────────────────── */

class ApiError extends Error {
    constructor(code, message, status) {
        super(message || code);
        this.code = code;
        this.status = status;
    }
}

async function api(method, path, { body, params, raw } = {}) {
    const url = new URL('/api/v1' + path, location.origin);
    for (const [k, v] of Object.entries(params || {})) {
        if (v != null && v !== '') url.searchParams.set(k, v);
    }
    const headers = {};
    const key = localStorage.getItem('flux_api_key');
    if (key) headers['X-API-Key'] = key;

    let payload;
    if (body instanceof FormData) {
        payload = body;                       // let fetch set the boundary
    } else if (body !== undefined) {
        payload = JSON.stringify(body);
        headers['Content-Type'] = 'application/json';
    }

    const res = await fetch(url, { method, headers, body: payload });
    if (res.status === 204) return null;
    if (raw && res.ok) return res.blob();

    const data = await res.json().catch(() => ({}));
    if (!res.ok) {
        const e = data.error || {};
        throw new ApiError(e.code || 'unknown', e.message || res.statusText, res.status);
    }
    return data;
}

const GET  = (p, o) => api('GET', p, o);
const POST = (p, body) => api('POST', p, { body });
const PUT  = (p, body) => api('PUT', p, { body });
const DEL  = (p) => api('DELETE', p);

/* ── toasts ─────────────────────────────────────────────────────────────── */

function toast(message, kind = '') {
    const node = el('div', `toast ${kind}`, message);
    $('toasts').append(node);
    setTimeout(() => {
        node.style.opacity = '0';
        setTimeout(() => node.remove(), 250);
    }, kind === 'err' ? 6000 : 3000);
}

const failed = (e) => toast(e instanceof ApiError ? `${e.code}: ${e.message}` : String(e), 'err');

/* ── boot: wait for the model ───────────────────────────────────────────── */

async function boot() {
    const started = Date.now();
    for (;;) {
        let health;
        try {
            health = await GET('/health');
        } catch {
            health = { ready: false, status: 'server unreachable (restarting?)' };
        }
        if (health.error) {
            $('bootTitle').textContent = 'Model failed to load';
            $('bootStatus').textContent = health.error;
            return false;
        }
        if (health.ready) return true;

        $('bootStatus').textContent = health.status || 'loading…';
        $('bootElapsed').textContent = `${Math.round((Date.now() - started) / 1000)}s`;
        await sleep(1500);
    }
}

const sleep = (ms) => new Promise(r => setTimeout(r, ms));

/* ── capabilities ───────────────────────────────────────────────────────── */

async function loadCapabilities() {
    const info = await GET('/model');
    state.caps = info;

    $('chipHost').textContent = `${info.hostname} · v${info.version}`;
    $('negativeField').hidden = !info.negative_prompt;
    // Inpainting needs exactly one reference plus a FLUX.2/SDXL backend.
    $('inpaintToggleWrap').hidden = !info.inpaint;

    // Few-step variants ignore a high step count, so say so rather than
    // letting the number sit there looking authoritative.
    if (info.turbo || info.schnell) {
        $('composerHint').textContent = 'Few-step model — steps are capped by the scheduler.';
    }
}

async function loadModels() {
    let models;
    try {
        models = await GET('/models');
    } catch { return; }

    const select = $('modelSelect');
    select.replaceChildren();
    for (const config of models.configs) {
        const option = el('option', null, `${config.id} · ${config.label}`);
        option.value = config.id;
        if (config.id === models.current) option.selected = true;
        select.append(option);
    }
    select.disabled = !models.switchable;
    select.title = models.switchable
        ? 'Switching restarts the server'
        : 'Started without its supervisor — cannot switch';
}

$('modelSelect').addEventListener('change', async (e) => {
    const config = Number(e.target.value);
    if (!confirm(`Switch to config ${config}? The server restarts and the model reloads.`)) {
        loadModels();
        return;
    }
    try {
        await PUT('/models/current', { config });
        toast('Restarting into the new model…');
        $('boot').hidden = false;
        $('bootTitle').textContent = 'Switching model';
        // The old process has to exit before /health means anything.
        await sleep(3000);
        if (await boot()) location.reload();
    } catch (e) { failed(e); }
});

/* ── prompt history ─────────────────────────────────────────────────────── */

function rememberPrompt(text) {
    if (!text.trim()) return;
    state.promptHistory = [text, ...state.promptHistory.filter(p => p !== text)].slice(0, 100);
    localStorage.setItem('flux_prompts', JSON.stringify(state.promptHistory));
    state.historyPos = -1;
    renderPromptPos();
}

function renderPromptPos() {
    $('promptPos').textContent = state.historyPos < 0
        ? '' : `${state.historyPos + 1}/${state.promptHistory.length}`;
}

$('promptPrev').onclick = () => {
    if (!state.promptHistory.length) return;
    if (state.historyPos < 0) state.draft = $('prompt').value;
    state.historyPos = Math.min(state.historyPos + 1, state.promptHistory.length - 1);
    $('prompt').value = state.promptHistory[state.historyPos];
    renderPromptPos();
};

$('promptNext').onclick = () => {
    if (state.historyPos < 0) return;
    state.historyPos -= 1;
    $('prompt').value = state.historyPos < 0 ? state.draft : state.promptHistory[state.historyPos];
    renderPromptPos();
};

$('promptCopy').onclick = async () => {
    await navigator.clipboard.writeText($('prompt').value);
    toast('Prompt copied');
};
$('promptClear').onclick = () => { $('prompt').value = ''; $('prompt').focus(); };

/* ── reference images ───────────────────────────────────────────────────── */

function renderRefs() {
    const box = $('refThumbs');
    box.replaceChildren();
    state.refs.forEach((ref, i) => {
        const thumb = el('div', 'ref-thumb');
        const img = el('img');
        img.src = ref.dataUrl;
        img.alt = ref.name || `reference ${i + 1}`;
        const remove = el('button', null, '✕');
        remove.title = 'Remove';
        remove.onclick = () => { state.refs.splice(i, 1); renderRefs(); };
        thumb.append(img, remove);
        box.append(thumb);
    });

    const count = state.refs.length;
    $('refAddBtn').hidden = count >= MAX_REFS;
    $('refControls').hidden = count === 0;
    $('describeBtn').disabled = true;          // pass 2
    $('multiRefHint').hidden = count < 2;

    // FLUX.1 img2img takes a single reference; the server rejects more.
    const multiOk = state.caps.kontext || state.caps.flux_version === 2;
    if (count > 1 && !multiOk) {
        toast('This model accepts one reference image only', 'err');
    }
}

function addRef(dataUrl, name) {
    if (state.refs.length >= MAX_REFS) { toast(`At most ${MAX_REFS} references`, 'err'); return; }
    state.refs.push({ dataUrl, name });
    renderRefs();
}

const RAW_EXT = /\.(nef|nrw|dng|cr2|cr3|arw|raf|orf|rw2)$/i;

function fileToDataUrl(file) {
    return new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.onload = () => resolve(reader.result);
        reader.onerror = reject;
        reader.readAsDataURL(file);
    });
}

$('refAddBtn').onclick = () => $('refFile').click();
$('refFile').onchange = async (e) => {
    for (const file of [...e.target.files].slice(0, MAX_REFS - state.refs.length)) {
        try {
            if (RAW_EXT.test(file.name)) {
                // Browsers cannot decode camera RAW; the server does it with LibRaw.
                const form = new FormData();
                form.append('file', file);
                const imported = await api('POST', '/imports/raw', { body: form });
                addRef(imported.image, file.name);
            } else {
                addRef(await fileToDataUrl(file), file.name);
            }
        } catch (err) { failed(err); }
    }
    e.target.value = '';
};

// One box for both: a URL goes to /imports/url, anything else is a server path.
$('refUrlAdd').onclick = async () => {
    const value = $('refUrl').value.trim();
    if (!value) return;
    const isUrl = /^https?:\/\//i.test(value);
    try {
        const imported = await POST(isUrl ? '/imports/url' : '/imports/path',
                                    isUrl ? { url: value } : { path: value });
        addRef(imported.image, value.split(/[\\/]/).pop());
        $('refUrl').value = '';
    } catch (e) { failed(e); }
};
$('refUrl').addEventListener('keydown', (e) => {
    if (e.key === 'Enter') { e.preventDefault(); $('refUrlAdd').click(); }
});

$('strength').oninput = (e) => { $('strengthOut').value = e.target.value; };

/* ── server file browser ────────────────────────────────────────────────── */

let browseDir = null;

async function openBrowser(dir) {
    const modal = $('browseModal');
    if (!modal.open) modal.showModal();
    try {
        const listing = await GET('/files', { params: { dir } });
        browseDir = listing.dir;
        $('browsePath').textContent = listing.dir;
        $('browseUp').disabled = !listing.parent;
        $('browseUp').dataset.parent = listing.parent || '';

        const grid = $('browseGrid');
        grid.replaceChildren();
        for (const name of listing.dirs) {
            const item = el('div', 'browse-item dir', name);
            item.onclick = () => openBrowser(`${listing.dir}/${name}`);
            grid.append(item);
        }
        for (const entry of listing.files) {
            const path = `${listing.dir}/${entry.filename}`;
            const item = el('div', 'browse-item');
            const img = el('img');
            // Entries carry only a filename, so the thumbnail needs the joined path.
            img.src = `/api/v1/files/thumbnail?path=${encodeURIComponent(path)}`;
            img.loading = 'lazy';
            img.alt = '';
            item.append(img, el('span', null, entry.filename));
            item.onclick = async () => {
                try {
                    const imported = await POST('/imports/path', { path });
                    addRef(imported.image, entry.filename);
                    modal.close();
                } catch (e) { failed(e); }
            };
            grid.append(item);
        }
    } catch (e) { failed(e); }
}

$('browseBtn').onclick = () => openBrowser(null);
$('browseClose').onclick = () => $('browseModal').close();
$('browseUp').onclick = (e) => openBrowser(e.target.dataset.parent || null);
$('browseGo').onclick = () => openBrowser($('browseJump').value.trim() || null);
$('browseJump').addEventListener('keydown', (e) => {
    if (e.key === 'Enter') { e.preventDefault(); $('browseGo').click(); }
});

/* ── spectrum selector ──────────────────────────────────────────────────── */

(function buildSpectrum() {
    const box = $('spectrumCells');
    for (let i = 0; i < 16; i++) {
        const cell = el('button');
        cell.type = 'button';
        cell.setAttribute('aria-pressed', 'false');
        cell.title = `Guidance ${[1, 3, 5, 7][i % 4]} · strength ${[0.2, 0.4, 0.6, 0.8][Math.floor(i / 4)]}`;
        cell.onclick = () => {
            const on = cell.getAttribute('aria-pressed') === 'true';
            cell.setAttribute('aria-pressed', String(!on));
            if (on) state.spectrumCells.delete(i); else state.spectrumCells.add(i);
        };
        box.append(cell);
    }
})();

$('spectrumGrid').onchange = (e) => { $('spectrumBox').hidden = !e.target.checked; };

/* ── generate ───────────────────────────────────────────────────────────── */

function buildRequest() {
    const prompt = $('prompt').value.trim();
    if (!prompt) throw new Error('A prompt is required');

    const body = {
        prompt,
        steps: Number($('steps').value),
        batch: Number($('batch').value),
        size: $('size').value,
        orientation: $('orientation').value,
        show_preview: $('showPreview').checked,
        save_previews: $('savePreviews').checked,
        expansion_same_seed: $('expansionSameSeed').checked,
    };

    const seed = $('seed').value.trim();
    if (seed) body.seed = Number(seed);

    const guidance = $('guidance').value;
    if (guidance) body.guidance = Number(guidance);

    if (state.caps.negative_prompt) {
        const negative = $('negativePrompt').value.trim();
        if (negative) body.negative_prompt = negative;
    }

    if (state.refs.length) {
        body.input_images = state.refs.map(r => r.dataUrl);
        body.strength = Number($('strength').value);
        body.aspect_mode = $('aspectMode').value;
    }

    if ($('spectrumGrid').checked) {
        body.spectrum_grid = true;
        body.spectrum_same_seed = $('spectrumSameSeed').checked;
        if (state.spectrumCells.size) body.selected_cells = [...state.spectrumCells];
    }

    return body;
}

async function generate() {
    let body;
    try { body = buildRequest(); } catch (e) { toast(e.message, 'err'); return; }

    const orientations = $('allOrientations').checked
        ? ['square', 'landscape', 'portrait', 'widescreen', 'extra-tall']
        : [body.orientation];

    $('generateBtn').disabled = true;
    try {
        let first = null;
        for (const orientation of orientations) {
            const job = await POST('/jobs', { ...body, orientation });
            first ??= job;
            if (job.expanded) {
                toast(`${job.expanded.length} jobs queued from {a|b} expansion`, 'ok');
            }
        }
        rememberPrompt(body.prompt);
        state.activeJob = first.id;
        toast(orientations.length > 1 ? `${orientations.length} jobs queued` : 'Queued', 'ok');
        pollNow();
    } catch (e) {
        failed(e);
    } finally {
        $('generateBtn').disabled = false;
    }
}

$('generateBtn').onclick = generate;
$('prompt').addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && (e.metaKey || e.ctrlKey)) { e.preventDefault(); generate(); }
});

$('resetBtn').onclick = () => {
    $('prompt').value = '';
    $('seed').value = '';
    state.refs = [];
    state.spectrumCells.clear();
    document.querySelectorAll('.spectrum button').forEach(c => c.setAttribute('aria-pressed', 'false'));
    $('spectrumGrid').checked = false;
    $('spectrumBox').hidden = true;
    $('allOrientations').checked = false;
    renderRefs();
};

/* ── polling: queue, live job, telemetry ────────────────────────────────── */

let pollTimer = null;

function pollNow() {
    clearTimeout(pollTimer);
    poll();
}

async function poll() {
    try {
        const queue = await GET('/queue');
        renderQueue(queue);
        renderLive(queue.running);
        setState(queue.running ? 'busy' : 'ok', queue.running ? 'generating' : 'idle');

        // A finished job means the gallery is stale. Give the worker a beat to
        // finish writing the PNG first: the job flips to done fractionally
        // before the file is closed, and a thumbnail requested inside that
        // window hangs on a half-written file.
        if (state.activeJob && !queue.running && !queue.waiting.length) {
            state.activeJob = null;
            setTimeout(loadGallery, 700);
        }
    } catch (e) {
        setState('offline', e.code === 'upstream_unreachable' ? 'server down' : 'error');
    }
    pollTimer = setTimeout(poll, 1200);
}

function setState(kind, text) {
    $('chipState').querySelector('.dot').className = `dot ${kind}`;
    $('chipStateText').textContent = text;
}

function renderQueue(queue) {
    const rows = [];
    if (queue.running) rows.push({ ...queue.running, position: '▶', running: true });
    rows.push(...queue.waiting);

    $('queueBadge').hidden = rows.length === 0;
    $('queueBadge').textContent = rows.length;
    $('queueEmpty').hidden = rows.length > 0;

    const parts = [`${queue.depth}/${queue.capacity} waiting`];
    if (queue.images_pending) parts.push(`${queue.images_pending} image(s) pending`);
    if (queue.estimated_wait_s) parts.push(`~${Math.round(queue.estimated_wait_s)}s`);
    if (!queue.accepting) parts.push('queue full');
    $('queueSummary').textContent = parts.join(' · ');

    const list = $('queueList');
    list.replaceChildren();
    for (const job of rows) {
        const row = el('div', `qrow ${job.running ? 'running' : ''}`);
        row.append(el('div', 'qpos', String(job.position)));
        row.append(el('div', 'qprompt', job.prompt || '(no prompt)'));

        const meta = [job.size, job.orientation];
        if (job.batch > 1) meta.push(`×${job.batch}`);
        if (job.refs) meta.push(`${job.refs} ref`);
        row.append(el('div', 'qmeta', meta.filter(Boolean).join(' · ')));

        const cancel = el('button', 'mini danger', '✕');
        cancel.title = job.running ? 'Interrupt' : 'Cancel';
        cancel.onclick = async () => {
            try { await DEL(`/jobs/${job.id}`); pollNow(); } catch (e) { failed(e); }
        };
        row.append(cancel);
        list.append(row);
    }
}

function renderLive(running) {
    const live = $('live');
    if (!running) {
        live.hidden = true;
        if (watchOpen) closeWatch();
        return;
    }
    live.hidden = false;

    const done = running.step || 0;
    const total = running.total_steps || 0;
    const pct = total ? (done / total) * 100 : 0;

    const label = running.batch > 1
        ? `image ${running.current}/${running.batch} · step ${done}/${total}`
        : `step ${done}/${total}`;
    $('liveText').textContent = label;
    $('liveFill').style.width = `${pct}%`;
    $('watchFill').style.width = `${pct}%`;
    $('watchText').textContent = label;

    // The live frame is overwritten in place, so its ts is the cache-buster.
    if (running.preview) {
        const url = `/api/v1/images/${running.preview}?t=${running.preview_ts || Date.now()}`;
        $('livePreview').hidden = false;
        $('livePreview').src = url;
        if (watchOpen) {
            $('watchImg').src = url;
            $('watchWaiting').hidden = true;
        }
    } else {
        $('livePreview').hidden = true;
    }
}

async function pollTelemetry() {
    try {
        const t = await GET('/telemetry');
        if (t.power_w != null) {
            $('chipPower').hidden = false;
            $('chipPowerText').textContent = `${Math.round(t.power_w)} W`;
        }
        if (t.vlm && t.vlm.label) {
            $('chipVlm').hidden = false;
            $('chipVlm').textContent = t.vlm.label;
            $('chipVlm').title = t.vlm.detail || '';
        }
    } catch { /* telemetry is decoration; never let it break the poll loop */ }
    setTimeout(pollTelemetry, 10000);
}

/* ── gallery ────────────────────────────────────────────────────────────── */

async function loadGallery() {
    try {
        const data = await GET('/images');
        state.gallery = data.images;
        $('galleryCount').textContent = data.count
            ? `${data.count} image${data.count === 1 ? '' : 's'} today`
            : '';
        $('galleryEmpty').hidden = data.count > 0;

        const grid = $('galleryGrid');
        grid.replaceChildren();
        data.images.forEach((image, index) => {
            const card = el('div', 'card');
            const img = el('img');
            // A day's output is dozens of multi-megabyte PNGs, so the grid asks
            // for the server's 240px JPEG rather than the full image. The
            // lightbox still loads the original.
            const thumb = `/api/v1/files/thumbnail?path=${encodeURIComponent(image.filename)}`;
            img.loading = 'lazy';
            img.alt = image.prompt || image.filename;
            // A thumbnail fetched the instant a job lands can catch the file
            // mid-write. One retry with a cache-buster covers that race.
            img.onerror = () => {
                img.onerror = null;
                setTimeout(() => { img.src = `${thumb}&retry=${Date.now()}`; }, 900);
            };
            img.src = thumb;
            card.append(img);

            const meta = el('div', 'card-meta');
            meta.append(el('span', null, image.time || ''));
            card.append(meta);
            if (image.prompt) card.append(el('div', 'card-prompt', image.prompt));

            card.onclick = () => openLightbox(index);
            grid.append(card);
        });
    } catch (e) { failed(e); }
}

$('clearRecentBtn').onclick = async () => {
    try { await DEL('/jobs/recent'); toast('Recent list cleared'); pollNow(); }
    catch (e) { failed(e); }
};

$('archiveBtn').onclick = async () => {
    if (!confirm("Move today's output into the archive folder?")) return;
    try { await POST('/archive'); toast('Archived', 'ok'); loadGallery(); }
    catch (e) { failed(e); }
};

$('deleteAllBtn').onclick = async () => {
    if (!confirm("Permanently delete ALL of today's images? This cannot be undone.")) return;
    try {
        const r = await DEL('/images');
        toast(`Deleted ${r.deleted ?? ''}`.trim(), 'ok');
        loadGallery();
    } catch (e) { failed(e); }
};

/* ── lightbox ───────────────────────────────────────────────────────────── */

function openLightbox(index) {
    state.lightboxIndex = index;
    const image = state.gallery[index];
    if (!image) return;
    $('lightboxImg').src = `/api/v1/images/${encodeURIComponent(image.filename)}`;
    $('lightboxCaption').textContent = `${image.time || ''} ${image.prompt || image.filename}`.trim();
    $('lightbox').hidden = false;
}

const closeLightbox = () => { $('lightbox').hidden = true; };
const stepLightbox = (delta) => {
    const next = state.lightboxIndex + delta;
    if (next >= 0 && next < state.gallery.length) openLightbox(next);
};

$('lightboxClose').onclick = closeLightbox;
$('lightboxPrev').onclick = () => stepLightbox(-1);
$('lightboxNext').onclick = () => stepLightbox(1);
$('lightboxStage').onclick = (e) => { if (e.target.id === 'lightboxStage') closeLightbox(); };

$('lbUseRef').onclick = async () => {
    const image = state.gallery[state.lightboxIndex];
    try {
        const imported = await POST('/imports/path', { path: image.filename });
        addRef(imported.image, image.filename);
        closeLightbox();
        toast('Attached as reference', 'ok');
    } catch (e) { failed(e); }
};

$('lbSave').onclick = async () => {
    const image = state.gallery[state.lightboxIndex];
    try {
        await POST(`/images/${encodeURIComponent(image.filename)}/save`);
        toast('Saved into .saved/', 'ok');
    } catch (e) { failed(e); }
};

$('lbDelete').onclick = async () => {
    const image = state.gallery[state.lightboxIndex];
    if (!confirm(`Permanently delete ${image.filename}?`)) return;
    try {
        await DEL(`/images/${encodeURIComponent(image.filename)}`);
        closeLightbox();
        loadGallery();
    } catch (e) { failed(e); }
};

/* ── watch overlay ──────────────────────────────────────────────────────── */

let watchOpen = false;

$('watchBtn').onclick = () => {
    watchOpen = true;
    $('watch').hidden = false;
    $('watchWaiting').hidden = false;
};
const closeWatch = () => { watchOpen = false; $('watch').hidden = true; };
$('watchClose').onclick = closeWatch;

const interrupt = async () => {
    try {
        const queue = await GET('/queue');
        if (queue.running) { await DEL(`/jobs/${queue.running.id}`); toast('Interrupted'); pollNow(); }
    } catch (e) { failed(e); }
};
$('interruptBtn').onclick = interrupt;
$('watchInterrupt').onclick = interrupt;

document.addEventListener('keydown', (e) => {
    if (e.key !== 'Escape') return;
    if (watchOpen) closeWatch();
    else if (!$('lightbox').hidden) closeLightbox();
});

/* ── tabs, theme, key ───────────────────────────────────────────────────── */

document.querySelectorAll('.tab').forEach(tab => {
    tab.onclick = () => {
        document.querySelectorAll('.tab').forEach(t => t.classList.toggle('active', t === tab));
        document.querySelectorAll('.pane').forEach(p => {
            p.classList.toggle('active', p.id === `pane-${tab.dataset.tab}`);
        });
    };
});

const savedTheme = localStorage.getItem('flux_theme');
if (savedTheme) document.documentElement.dataset.theme = savedTheme;
$('themeBtn').onclick = () => {
    const next = document.documentElement.dataset.theme === 'light' ? 'dark' : 'light';
    document.documentElement.dataset.theme = next;
    localStorage.setItem('flux_theme', next);
};

$('keyBtn').onclick = () => {
    $('keyInput').value = localStorage.getItem('flux_api_key') || '';
    $('keyModal').showModal();
};
$('keyClose').onclick = () => $('keyModal').close();
$('keySave').onclick = () => {
    const value = $('keyInput').value.trim();
    if (value) localStorage.setItem('flux_api_key', value);
    else localStorage.removeItem('flux_api_key');
    $('keyModal').close();
    toast('Key saved');
};

/* ── edit loop ──────────────────────────────────────────────────────────── */

/* One round is: generate from the current base with the current instruction,
 * ask the vision model whether the edit actually landed, and take its revised
 * instruction into the next round. The loop's whole point is that the critic,
 * not the user, writes the retry — so a critique failure degrades to reusing
 * the previous instruction rather than aborting the run.
 */

let loopRun = null;   // {stop, decision, nextBase} while a run is in flight

function loopStatus(message, kind = '') {
    const node = $('loopStatus');
    node.hidden = !message;
    node.textContent = message || '';
    node.style.color = kind === 'error' ? 'var(--danger)'
                     : kind === 'done'  ? 'var(--ok)' : '';
}

async function loopGenerate(prompt, refDataUrl, onTick) {
    const job = await POST('/jobs', {
        prompt,
        input_images: [refDataUrl],
        batch: 1,
        steps: Number($('steps').value),
        guidance: $('guidance').value ? Number($('guidance').value) : null,
        aspect_mode: 'keep',
        show_preview: $('showPreview').checked,
    });

    for (;;) {
        await sleep(1500);
        const current = await GET(`/jobs/${job.id}`);
        if (TERMINAL.includes(current.state)) {
            if (current.state !== 'done' || current.error) {
                throw new Error(current.error || `job ${current.state}`);
            }
            return current.images[0].filename;
        }
        onTick?.(current);
        if (loopRun?.stop) {
            // A queued job can be dropped; a running one has to finish.
            await DEL(`/jobs/${job.id}`).catch(() => {});
            throw new Error('stopped');
        }
    }
}

async function loopCritique(fields) {
    const started = await POST('/vlm/jobs', { task: 'critique', ...fields });
    for (;;) {
        await sleep(2000);
        const poll = await GET(`/vlm/jobs/${started.id}`);
        if (poll.done) return poll.result;
        if (loopRun?.stop) throw new Error('stopped');
    }
}

// Chaining and backtracking both need a produced PNG back as a data URL; the
// server already resolves a bare filename against the output directory.
const loopBaseFrom = (filename) =>
    POST('/imports/path', { path: filename }).then(r => r.image);

function loopCard(round, prompt, filename) {
    const card = el('div', 'card');
    const img = el('img');
    img.src = `/api/v1/files/thumbnail?path=${encodeURIComponent(filename)}`;
    img.alt = prompt;
    img.style.aspectRatio = 'auto';
    img.onclick = () => {
        state.gallery = [{ filename, prompt, time: `round ${round}` }];
        openLightbox(0);
    };
    card.append(img, el('div', 'card-meta', `round ${round}`), el('div', 'card-prompt', prompt));

    const verdict = el('div', 'card-prompt');
    verdict.style.color = 'var(--text-3)';
    card.append(verdict);

    // Backtrack: make this round's output the base for the next one.
    const useBase = el('button', 'mini', '⏪ Base for next round');
    useBase.onclick = () => {
        if (loopRun) { loopRun.nextBase = filename; toast(`Round ${round} is the next base`); }
    };
    card.append(useBase);

    $('loopCards').prepend(card);
    return verdict;
}

function loopAwaitDecision() {
    $('loopControls').hidden = false;
    return new Promise(resolve => {
        loopRun.decision = (choice) => {
            $('loopControls').hidden = true;
            loopRun.decision = null;
            resolve(choice);
        };
    });
}

async function runEditLoop() {
    const direction = $('loopDirection').value.trim();
    const maxRounds = Number($('loopIterations').value);
    const auto = $('loopAuto').checked;
    const chain = $('loopChain').checked;

    if (!direction) { toast('Enter an edit direction first', 'err'); return; }
    if (!state.refs.length) { toast('Attach a reference image — the loop edits the first one', 'err'); return; }

    loopRun = { stop: false, decision: null, nextBase: null };
    $('loopStart').hidden = true;
    $('loopStop').hidden = false;
    $('loopCards').replaceChildren();

    const originalRef = state.refs[0].dataUrl;
    let base = originalRef;
    let prompt = direction;
    const completed = [];
    const history = [];                 // trajectory, so the critic stops re-proposing
    let best = { score: -1, round: 0 };
    let makeStrip = false;

    try {
        for (let round = 1; round <= maxRounds && !loopRun.stop; round++) {
            loopStatus(`Round ${round}/${maxRounds}: generating…`);
            const filename = await loopGenerate(prompt, base, (job) => {
                const phase = job.state === 'running' && job.total_steps
                    ? ` — step ${job.step}/${job.total_steps}` : ' — queued';
                loopStatus(`Round ${round}/${maxRounds}: generating${phase}`);
            });
            completed.push({ filename, prompt });
            const verdict = loopCard(round, prompt, filename);

            loopStatus(`Round ${round}/${maxRounds}: comparing input and output…`);
            let critique = null;
            try {
                critique = await loopCritique({
                    direction, prompt, ref_image: base,
                    output_filename: filename, history,
                });
            } catch (e) {
                if (e.message === 'stopped') throw e;
                verdict.textContent = `Critique unavailable: ${e.message}`;
            }

            let next = prompt;
            if (critique) {
                const applied = critique.applied === null ? ''
                    : critique.applied ? '✔ applied — ' : '✘ not applied — ';
                const score = typeof critique.score === 'number' ? ` (${critique.score}/10)` : '';
                verdict.textContent = applied + (critique.critique || '') + score;
                verdict.style.color = critique.applied === false ? 'var(--danger)' : 'var(--ok)';
                next = critique.revised_prompt || prompt;
                if (typeof critique.score === 'number' && critique.score > best.score) {
                    best = { score: critique.score, round };
                }
            }
            $('loopNext').value = next;
            history.push({
                prompt,
                applied: critique ? critique.applied : null,
                score: critique ? critique.score : null,
                critique: critique ? (critique.critique || '') : '',
            });

            if (loopRun.stop) break;
            // Auto mode stops early once the critic says the goal landed.
            if (auto && critique?.applied && typeof critique.score === 'number' && critique.score >= 8) {
                loopStatus(`Goal reached at round ${round} (${critique.score}/10).`);
                makeStrip = true;
                break;
            }
            if (round === maxRounds) { makeStrip = true; break; }

            if (auto) {
                prompt = next;
            } else {
                loopStatus(`Round ${round}/${maxRounds} done — adjust the instruction or continue.`);
                const decision = await loopAwaitDecision();
                if (decision === 'accept') { makeStrip = true; break; }
                if (decision === 'stop') break;
                prompt = $('loopNext').value.trim() || next;
            }

            // Base for the next round: an explicit backtrack wins, then chain
            // mode follows the newest output, otherwise stay on the original.
            if (loopRun.nextBase) {
                base = loopRun.nextBase === '__original__'
                    ? originalRef : await loopBaseFrom(loopRun.nextBase);
                loopRun.nextBase = null;
            } else if (chain) {
                base = await loopBaseFrom(filename);
            }
        }

        if (makeStrip && completed.length) {
            loopStatus('Saving iterations and composing the film strip…');
            try {
                const strip = await POST('/filmstrips', {
                    direction,
                    ref_image: originalRef,
                    filenames: completed.map(c => c.filename),
                    prompts: completed.map(c => c.prompt),
                });
                const card = el('div', 'card');
                const img = el('img');
                img.src = `/api/v1/images/${encodeURIComponent(strip.filename)}`;
                img.style.aspectRatio = 'auto';
                card.append(img, el('div', 'card-meta', 'film strip — input plus each edit'));
                $('loopCards').prepend(card);
            } catch (e) {
                loopStatus(`Film strip failed: ${e.message}`, 'error');
            }
        }

        const bestText = best.score >= 0 ? ` Best: round ${best.round} (${best.score}/10).` : '';
        loopStatus((loopRun.stop ? 'Loop stopped.' : 'Loop finished.') + bestText, 'done');
    } catch (e) {
        loopStatus(e.message === 'stopped' ? 'Loop stopped.' : `Loop error: ${e.message}`,
                   e.message === 'stopped' ? 'done' : 'error');
    } finally {
        $('loopControls').hidden = true;
        $('loopStop').hidden = true;
        $('loopStart').hidden = false;
        loopRun = null;
        loadGallery();
    }
}

$('loopStart').onclick = runEditLoop;
$('loopStop').onclick = () => {
    if (!loopRun) return;
    loopRun.stop = true;
    loopRun.decision?.('stop');
    loopStatus('Stopping after this round…');
};
$('loopContinue').onclick = () => loopRun?.decision?.('continue');
$('loopAccept').onclick = () => loopRun?.decision?.('accept');
$('loopBaseOriginal').onclick = () => {
    if (!loopRun) return;
    loopRun.nextBase = '__original__';
    toast('Next round edits the original again');
};

/* Controls whose behavior lands in pass 2. Disabled rather than hidden so the
   layout being reviewed is the finished one. */
function markPending() {
    const pending = ['boostBtn', 'boostNBtn', 'evolveBtn', 'describeBtn',
                     'inpaintMode', 'compareStart'];
    for (const id of pending) {
        const node = $(id);
        node.disabled = true;
        node.title = 'Not wired yet — coming in the next pass';
    }
}

/* ── start ──────────────────────────────────────────────────────────────── */

(async function start() {
    renderPromptPos();
    markPending();
    if (!(await boot())) return;
    $('boot').hidden = true;

    await loadCapabilities();
    await loadModels();
    renderRefs();
    loadGallery();
    poll();
    pollTelemetry();
    $('prompt').focus();
})();
