/* The /simple front end: a prompt, a button, an image.
 *
 * Standalone on purpose. app.js drives every control the server has and is
 * large enough that "simple" would only ever be a CSS illusion over it — a
 * hidden control is still a control that can break. This file instead speaks
 * the versioned REST API (/api/v1) directly and knows about exactly three
 * calls: health, submit, poll. Everything else about a generation — square,
 * 1MP, 25 steps, a fresh seed — is left to the server's defaults, so this page
 * inherits changes to them (like the square default) without being edited.
 */

const API = '/api/v1';
const POLL_MS = 700;

const el = (id) => document.getElementById(id);
const keyRow = el('keyRow');
const keyInput = el('apiKey');
const form = el('form');
const promptInput = el('prompt');
const generateBtn = el('generate');
const clearBtn = el('clear');
const statusLine = el('status');
const bar = el('bar');
const barFill = el('barFill');
const stage = el('stage');
const image = el('image');
const fileInput = el('file');
const thumb = el('thumb');
const thumbImg = el('thumbImg');
const refNote = el('refNote');
const useAsRefBtn = el('useAsRef');

let modelReady = false;
let busy = false;
let lastPreviewTs = null;

/* The attached reference, or null. Two flavours, because they reach the server
   by different routes: a file the user picked travels as a base64 `input_images`
   entry, while an image this server just generated is already sitting in
   web-generated/ and only needs its filename in `input_paths` — no download and
   re-upload of a megabyte we both already have.
     { src: <url for the thumbnail>, dataUrl?: string, path?: string }        */
let reference = null;

// -------------------------------------------------------------- API key --

// The same localStorage entry app.js uses, so a key entered on either page
// works on both.
const getKey = () => localStorage.getItem('flux_api_key') || '';

function setKey(value) {
    if (value) {
        localStorage.setItem('flux_api_key', value);
    } else {
        localStorage.removeItem('flux_api_key');
    }
    keyRow.hidden = !!value;
}

function saveKey() {
    const value = keyInput.value.trim();
    if (!value) return;
    setKey(value);
    keyInput.value = '';   // the row is hidden now; don't leave the secret in the DOM
    if (modelReady) ready();
}

el('saveKey').addEventListener('click', saveKey);

// The key field sits outside the form (submitting it would generate), so Enter
// needs wiring up by hand — typing a key and pressing Enter is the reflex.
keyInput.addEventListener('keydown', (event) => {
    if (event.key === 'Enter') {
        event.preventDefault();
        saveKey();
    }
});

// --------------------------------------------------------------- plumbing --

function say(text, isError) {
    statusLine.textContent = text;
    statusLine.classList.toggle('error', !!isError);
}

function progress(fraction) {
    bar.hidden = fraction === null;
    if (fraction !== null) barFill.style.width = `${Math.round(fraction * 100)}%`;
}

/* One request. Errors always arrive as {error: {code, message}}, so unwrap
   that into a thrown Error carrying the code — callers branch on the code and
   show the message, which is written for humans. */
async function api(path, options = {}) {
    const headers = { ...(options.headers || {}) };
    const key = getKey();
    if (key) headers['X-API-Key'] = key;
    if (options.body) headers['Content-Type'] = 'application/json';

    const response = await fetch(API + path, { ...options, headers });
    const payload = await response.json().catch(() => null);

    if (!response.ok) {
        const info = (payload && payload.error) || {};
        const error = new Error(info.message || `HTTP ${response.status}`);
        error.code = info.code || 'http_error';
        error.status = response.status;
        throw error;
    }
    return payload;
}

// Image bytes need the key too, and an <img> cannot set a header — the query
// form of the key exists for exactly this.
const imageUrl = (filename, cacheBust) =>
    `${API}/images/${filename}?api_key=${encodeURIComponent(getKey())}` +
    (cacheBust ? `&t=${cacheBust}` : '');

/* `filename` is set only for a finished image — a live preview is a scratch
   file the next job overwrites, so it is never offered as a reference. */
function show(src, isPreview, filename) {
    image.classList.toggle('preview', !!isPreview);
    image.src = src;
    stage.hidden = false;
    useAsRefBtn.hidden = !filename;
    if (filename) useAsRefBtn.dataset.filename = filename;
}

// ---------------------------------------------------------- reference image --

/* FLUX conditions on ~2MP inputs, so anything larger is upload wasted twice
   over — base64 inflates it by a third, and the server rejects a body past
   64MB. A raw phone photo clears that on its own, so bound the long edge and
   re-encode as JPEG before it ever leaves the page. */
const MAX_EDGE = 2048;

function shrink(dataUrl) {
    return new Promise((resolve) => {
        const img = new Image();
        img.onload = () => {
            const scale = Math.min(1, MAX_EDGE / Math.max(img.naturalWidth, img.naturalHeight));
            if (scale === 1 && dataUrl.length < 4 * 1024 * 1024) return resolve(dataUrl);
            const canvas = document.createElement('canvas');
            canvas.width = Math.max(1, Math.round(img.naturalWidth * scale));
            canvas.height = Math.max(1, Math.round(img.naturalHeight * scale));
            canvas.getContext('2d').drawImage(img, 0, 0, canvas.width, canvas.height);
            resolve(canvas.toDataURL('image/jpeg', 0.92));
        };
        // Anything the browser can't decode (HEIC outside Safari, camera RAW)
        // goes up untouched — the server's PIL decode is the real gate, and its
        // 400 is a better message than a guess made here.
        img.onerror = () => resolve(dataUrl);
        img.src = dataUrl;
    });
}

const readFile = (file) => new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(reader.result);
    reader.onerror = () => reject(new Error('could not read that file'));
    reader.readAsDataURL(file);
});

function setReference(next) {
    reference = next;
    thumbImg.src = next.src;
    thumb.hidden = false;
    refNote.hidden = false;
    // aspect_mode defaults to "keep", so an attached reference — not the square
    // default — decides the output's shape. Say so, since this page has no
    // orientation control to make that visible.
    if (!busy) say('reference attached');
    syncClear();
}

async function attach(file) {
    if (!file) return;
    try {
        const dataUrl = await shrink(await readFile(file));
        setReference({ src: dataUrl, dataUrl });
    } catch (e) {
        say(e.message, true);
    }
}

function detach() {
    reference = null;
    thumb.hidden = true;
    refNote.hidden = true;
    fileInput.value = '';   // so re-picking the same file still fires change
    syncClear();
}

fileInput.addEventListener('change', () => attach(fileInput.files[0]));
el('removeRef').addEventListener('click', detach);

/* Carry the last result forward as the next generation's reference: iterate on
   an image by editing the prompt rather than starting over. The file is already
   on the server, so this passes the filename and not the pixels. */
useAsRefBtn.addEventListener('click', () => {
    const filename = useAsRefBtn.dataset.filename;
    if (filename) setReference({ src: imageUrl(filename), path: filename });
});

// Drop anywhere on the page, and paste from the clipboard — both are quicker
// than the file dialog and cost a few lines each.
document.addEventListener('dragover', (event) => {
    event.preventDefault();
    document.body.classList.add('dropping');
});
document.addEventListener('dragleave', (event) => {
    if (event.relatedTarget === null) document.body.classList.remove('dropping');
});
document.addEventListener('drop', (event) => {
    event.preventDefault();
    document.body.classList.remove('dropping');
    const file = [...(event.dataTransfer.files || [])].find((f) => f.type.startsWith('image/'));
    attach(file);
});
document.addEventListener('paste', (event) => {
    const item = [...(event.clipboardData.items || [])].find((i) => i.type.startsWith('image/'));
    if (item) attach(item.getAsFile());
});

// ------------------------------------------------------- model readiness --

/* /health is public and answers 503 while the model loads. A cold FLUX.2 load
   runs into minutes, so the button stays disabled and the status line narrates
   what the server says it is doing rather than sitting on a spinner. */
async function waitForModel() {
    for (;;) {
        try {
            const response = await fetch(`${API}/health`);
            const health = await response.json();
            if (health.ready) {
                modelReady = true;
                ready();
                return;
            }
            say(health.error ? `model failed to load: ${health.error}`
                             : `${health.status}…`, !!health.error);
            if (health.error) return;
        } catch (e) {
            say('server unreachable — retrying…', true);
        }
        await new Promise((resolve) => setTimeout(resolve, 2000));
    }
}

function ready() {
    if (!getKey()) {
        keyRow.hidden = false;
        generateBtn.disabled = true;
        say('enter the API key to generate');
        return;
    }
    keyRow.hidden = true;
    generateBtn.disabled = busy;
    if (!busy) say('ready');
}

// ------------------------------------------------------------- generating --

form.addEventListener('submit', async (event) => {
    event.preventDefault();
    const prompt = promptInput.value.trim();
    if (!prompt || busy || !modelReady) return;

    busy = true;
    generateBtn.disabled = true;
    lastPreviewTs = null;
    progress(0);
    say('queued');

    try {
        // show_preview asks for decoded latents while denoising, which is what
        // makes the wait legible; everything else is a server default —
        // including strength and aspect_mode when a reference is attached.
        const body = { prompt, show_preview: true };
        if (reference && reference.path) {
            body.input_paths = [reference.path];
        } else if (reference) {
            body.input_images = [reference.dataUrl];
        }

        const job = await api('/jobs', {
            method: 'POST',
            body: JSON.stringify(body),
        });
        await watch(job.id);
    } catch (e) {
        if (e.status === 401) {
            setKey('');
            ready();
        } else {
            progress(null);
            say(e.message, true);
        }
    } finally {
        busy = false;
        generateBtn.disabled = !modelReady || !getKey();
    }
});

/* Poll one job to its end. A job goes queued → running → done|failed|canceled;
   total_steps is re-read from the live scheduler on the first step (turbo and
   img2img denoise fewer steps than requested), so the bar is only drawn once
   the server has reported both numbers. */
async function watch(jobId) {
    for (;;) {
        await new Promise((resolve) => setTimeout(resolve, POLL_MS));

        const job = await api(`/jobs/${jobId}`);

        if (job.state === 'queued') {
            say('waiting for the queue');
        } else if (job.state === 'running') {
            const { step, total_steps: total } = job;
            say(total ? `generating — step ${step} of ${total}` : 'generating');
            progress(total ? step / total : 0);
            if (job.preview && job.preview_ts && job.preview_ts !== lastPreviewTs) {
                lastPreviewTs = job.preview_ts;
                show(imageUrl(job.preview, job.preview_ts), true);
            }
        } else if (job.state === 'done') {
            progress(null);
            const output = job.images && job.images[0];
            if (output) {
                show(imageUrl(output.filename), false, output.filename);
                const seconds = job.generation_time;
                say(seconds ? `done in ${seconds.toFixed(1)}s — seed ${output.seed}`
                            : `done — seed ${output.seed}`);
            } else {
                say('finished with no image', true);
            }
            return;
        } else {
            progress(null);
            say(job.error || `job ${job.state}`, true);
            return;
        }
    }
}

/* Clear resets both inputs — prompt and reference. Clearing only the text
   would leave an attached image quietly steering the next generation, which is
   the sort of thing you notice three confusing results later. The image from
   the last run stays on screen, though: the usual reason to clear is to write
   something different while still looking at what the last prompt produced. */
clearBtn.addEventListener('click', () => {
    promptInput.value = '';
    detach();
    promptInput.focus();
});

// Enabled whenever there is something to clear.
function syncClear() {
    clearBtn.disabled = promptInput.value === '' && !reference;
}

promptInput.addEventListener('input', syncClear);

// Cmd/Ctrl+Enter submits from the textarea; a bare Enter stays a newline, since
// prompts run to several lines.
promptInput.addEventListener('keydown', (event) => {
    if (event.key === 'Enter' && (event.metaKey || event.ctrlKey)) {
        event.preventDefault();
        form.requestSubmit();
    }
});

keyRow.hidden = !!getKey();
syncClear();
waitForModel();
