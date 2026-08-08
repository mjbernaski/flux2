/**
 * Sample JavaScript client for the FLUX REST API (/api/v1).
 *
 * Runs unmodified in the browser and in Node 18+ (both have fetch). Node:
 *
 *   FLUX_API_KEY=... node examples/flux_client.js "a red fox in snow"
 *
 * Browser: import the class and construct it with your key. Note that the key
 * is visible to anyone with the page, so only do that on a trusted network —
 * which is the same assumption the built-in UI makes.
 *
 * Every error is {error: {code, message}}; branch on `code`, since the message
 * text is free to change.
 */

const TERMINAL_STATES = ['done', 'failed', 'canceled'];

export class FluxError extends Error {
    constructor(code, message, status) {
        super(`${code}: ${message}`);
        this.code = code;
        this.status = status;
    }
}

export class FluxClient {
    constructor({ baseUrl = 'http://localhost:2222', apiKey } = {}) {
        this.base = baseUrl.replace(/\/$/, '') + '/api/v1';
        this.apiKey = apiKey;
    }

    async request(method, path, { body, raw = false, params } = {}) {
        const url = new URL(this.base + path);
        for (const [k, v] of Object.entries(params || {})) {
            if (v != null) url.searchParams.set(k, v);
        }
        const res = await fetch(url, {
            method,
            headers: {
                'X-API-Key': this.apiKey,
                ...(body ? { 'Content-Type': 'application/json' } : {}),
            },
            body: body ? JSON.stringify(body) : undefined,
        });
        if (res.status === 204) return null;
        if (raw && res.ok) return res.blob();
        const data = await res.json().catch(() => ({}));
        if (!res.ok) {
            const err = data.error || {};
            throw new FluxError(err.code || 'unknown', err.message || res.statusText, res.status);
        }
        return data;
    }

    get(p, o) { return this.request('GET', p, o); }
    post(p, body) { return this.request('POST', p, { body }); }
    put(p, body) { return this.request('PUT', p, { body }); }
    del(p) { return this.request('DELETE', p); }

    // -- readiness and model ------------------------------------------------

    /** Public — answers 503 while the model is still loading. */
    async health() {
        const res = await fetch(this.base + '/health');
        return res.json();
    }

    /** Block until the model is loaded. A config switch restarts the process,
     *  so connection failures here are expected and retried. */
    async waitUntilReady({ timeoutMs = 1_800_000, pollMs = 3000, onStatus } = {}) {
        const deadline = Date.now() + timeoutMs;
        let last;
        while (Date.now() < deadline) {
            let state;
            try {
                state = await this.health();
            } catch {
                state = { ready: false, status: 'server unreachable (restarting?)' };
            }
            if (state.ready) return state;
            if (state.error) throw new FluxError('model_load_failed', state.error);
            if (onStatus && state.status !== last) onStatus((last = state.status));
            await new Promise(r => setTimeout(r, pollMs));
        }
        throw new Error('Model was not ready in time');
    }

    model() { return this.get('/model'); }
    models() { return this.get('/models'); }
    telemetry() { return this.get('/telemetry'); }

    /** Restarts the server; 202 means scheduled, not ready. */
    switchModel(config) { return this.put('/models/current', { config }); }

    // -- generation ---------------------------------------------------------

    /** Queue a job without waiting. Accepts every generation parameter —
     *  steps, batch, seed, guidance, strength, orientation, size,
     *  input_images, mask_image, show_preview, ... */
    submit(prompt, params = {}) { return this.post('/jobs', { ...params, prompt }); }

    job(id) { return this.get(`/jobs/${id}`); }
    jobs() { return this.get('/jobs'); }

    /** The current queue, ordered. `waiting[].position` is 1-based;
     *  `accepting` is false once a further submit would hit queue_full. */
    queue() { return this.get('/queue'); }
    cancel(id) { return this.del(`/jobs/${id}`); }
    /** Intermediate images: `live` is the frame being denoised now (needs
     *  show_preview), `frames` are per-step images on disk (needs
     *  save_previews, but they outlive the job). */
    previews(id) { return this.get(`/jobs/${id}/previews`); }

    /** Poll until the job settles. `onProgress` fires on each change of
     *  state/image/step — enough to drive a progress bar. */
    async waitForJob(id, { pollMs = 1500, timeoutMs = 3_600_000, onProgress } = {}) {
        const deadline = Date.now() + timeoutMs;
        let lastKey;
        while (Date.now() < deadline) {
            const job = await this.job(id);
            const key = `${job.state}:${job.current}:${job.step}`;
            if (onProgress && key !== lastKey) { onProgress(job); lastKey = key; }
            if (TERMINAL_STATES.includes(job.state)) return job;
            await new Promise(r => setTimeout(r, pollMs));
        }
        throw new Error(`Job ${id} did not finish in time`);
    }

    /** Submit and wait. Throws if the job failed. */
    async generate(prompt, { onProgress, ...params } = {}) {
        const { id } = await this.submit(prompt, params);
        const job = await this.waitForJob(id, { onProgress });
        if (job.state === 'failed') throw new FluxError('generation_failed', job.error);
        return job;
    }

    // -- images -------------------------------------------------------------

    async history() { return (await this.get('/images')).images; }

    /** A URL usable directly as an <img src>. The query-param form of the key
     *  exists precisely because an <img> cannot send headers. */
    imageUrl(filename) {
        return `${this.base}/images/${encodeURIComponent(filename)}?api_key=${encodeURIComponent(this.apiKey)}`;
    }

    imageBlob(filename) { return this.request('GET', `/images/${filename}`, { raw: true }); }
    saveImage(filename) { return this.post(`/images/${filename}/save`); }
    deleteImage(filename) { return this.del(`/images/${filename}`); }
    deleteToday() { return this.del('/images'); }
    archive() { return this.post('/archive'); }

    // -- reference images ---------------------------------------------------

    /** Turn a File/Blob (drag-drop, file input) into the data URL the API
     *  takes for input_images. */
    static async toDataUrl(file) {
        const buf = new Uint8Array(await file.arrayBuffer());
        let binary = '';
        for (const b of buf) binary += String.fromCharCode(b);
        const mime = file.type || 'image/png';
        return `data:${mime};base64,${btoa(binary)}`;
    }

    importUrl(url) { return this.post('/imports/url', { url }); }
    importPath(path) { return this.post('/imports/path', { path }); }
    browse(dir) { return this.get('/files', { params: { dir } }); }

    /** Camera RAW needs multipart, so it bypasses the JSON helper. */
    async importRaw(file) {
        const form = new FormData();
        form.append('file', file);
        const res = await fetch(this.base + '/imports/raw', {
            method: 'POST', headers: { 'X-API-Key': this.apiKey }, body: form,
        });
        const data = await res.json();
        if (!res.ok) throw new FluxError(data.error?.code, data.error?.message, res.status);
        return data;
    }

    // -- vision-model jobs --------------------------------------------------

    /** Start a describe / boost / critique job; returns the id to poll. */
    async vlmJob(task, fields) {
        return (await this.post('/vlm/jobs', { ...fields, task })).id;
    }

    async awaitVlm(id, { pollMs = 2000, timeoutMs = 900_000 } = {}) {
        const deadline = Date.now() + timeoutMs;
        while (Date.now() < deadline) {
            const state = await this.get(`/vlm/jobs/${id}`);
            if (state.done) return state.result;
            await new Promise(r => setTimeout(r, pollMs));
        }
        throw new Error(`VLM job ${id} did not finish in time`);
    }

    /** Photo(s) -> a prompt that would recreate them. */
    async describe(dataUrls, think = false) {
        const images = Array.isArray(dataUrls) ? dataUrls : [dataUrls];
        return (await this.awaitVlm(await this.vlmJob('describe', { images, think }))).prompt;
    }

    /** Draft prompt -> a stronger one in the loaded model's idiom. */
    boost(prompt, { level = 3, hasImage = false, negativePrompt, think = false } = {}) {
        return this.vlmJob('boost', {
            prompt, level, think, has_image: hasImage, negative_prompt: negativePrompt,
        }).then(id => this.awaitVlm(id));
    }

    critique({ direction, refImage, outputFilename, prompt, history = [] }) {
        return this.vlmJob('critique', {
            direction, ref_image: refImage, output_filename: outputFilename,
            prompt: prompt || direction, history,
        }).then(id => this.awaitVlm(id));
    }

    // -- multi-model runs ---------------------------------------------------

    multiRun(prompt, configs, params = {}) {
        return this.post('/multi-runs', { ...params, prompt, configs });
    }

    async multiRunStatus() {
        try {
            return await this.get('/multi-runs/current');
        } catch (e) {
            if (e.code === 'not_found') return null;
            throw e;
        }
    }

    cancelMultiRun() { return this.del('/multi-runs/current'); }
}

// --------------------------------------------------------------------------
// Browser usage sketch:
//
//   const flux = new FluxClient({ apiKey: localStorage.flux_api_key });
//   await flux.waitUntilReady({ onStatus: s => status.textContent = s });
//
//   const job = await flux.generate('a red fox in snow', {
//       steps: 30,
//       onProgress: j => bar.value = j.step / (j.total_steps || 1),
//   });
//   for (const image of job.images) {
//       const img = new Image();
//       img.src = flux.imageUrl(image.filename);
//       document.body.append(img);
//   }
//
// Editing from a file input:
//
//   const dataUrl = await FluxClient.toDataUrl(fileInput.files[0]);
//   await flux.generate('make it winter', {
//       input_images: [dataUrl], strength: 0.55, aspect_mode: 'keep',
//   });
// --------------------------------------------------------------------------

// Node CLI: node examples/flux_client.js "your prompt"
if (typeof process !== 'undefined' && process.argv?.[1]?.endsWith('flux_client.js')) {
    const prompt = process.argv.slice(2).join(' ') || 'a red fox in falling snow';
    const flux = new FluxClient({
        baseUrl: process.env.FLUX_URL || 'http://localhost:2222',
        apiKey: process.env.FLUX_API_KEY,
    });
    const state = await flux.health();
    if (!state.ready) {
        console.log(`Waiting for the model (${state.status})...`);
        await flux.waitUntilReady({ onStatus: s => console.log('  ' + s) });
    }
    console.log(`Generating: ${prompt}`);
    const job = await flux.generate(prompt, {
        onProgress: j => {
            if (j.state === 'running' && j.total_steps) {
                process.stdout.write(`\r  step ${j.step}/${j.total_steps}`);
            }
        },
    });
    console.log(`\n  ${job.generation_time.toFixed(1)}s`);
    for (const image of job.images) {
        console.log(`  ${image.filename}  (seed ${image.seed})`);
        console.log(`  ${flux.imageUrl(image.filename)}`);
    }
}
