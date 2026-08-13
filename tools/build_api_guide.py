"""Build the FLUX API developer guides (PDF), one per client language.

    python tools/build_api_guide.py

Writes docs/flux-api-guide-{python,rust,js}.pdf, plus the intermediate HTML
next to them for inspection.

The prose is shared; only the code samples differ, so the guides can't drift
apart in what they claim the API does. Rendering is headless Chrome's
print-to-PDF — the project has no PDF library, and Chrome gives us real
pagination and web typography for free.

Add a section by appending to SECTIONS. Every Code entry needs a sample for
each language in LANGUAGES (`py`, `rs`, `js`), so a language never silently
lacks coverage: validate() checks before anything is written. Adding a fourth
language is a row in LANGUAGES, a field on Code, and a sample per block.
"""

import html
import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from datetime import date

from pygments import highlight
from pygments.formatters import HtmlFormatter
from pygments.lexers import get_lexer_by_name

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(ROOT, 'docs')

# `attr` names the field on a Code block that holds this language's sample, so
# adding a language is a row here plus a field on Code — render and validate
# both drive off this table rather than testing for a particular language.
LANGUAGES = {
    'python': {'title': 'Python', 'lexer': 'python', 'file': 'python', 'attr': 'py'},
    'rust': {'title': 'Rust', 'lexer': 'rust', 'file': 'rust', 'attr': 'rs'},
    'javascript': {'title': 'JavaScript', 'lexer': 'javascript', 'file': 'js', 'attr': 'js'},
}

CHROME_CANDIDATES = [
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
    os.path.expandvars(r"%LOCALAPPDATA%\Google\Chrome\Application\chrome.exe"),
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    "/usr/bin/google-chrome",
    "/usr/bin/chromium",
]


# --------------------------------------------------------------- content ----

@dataclass
class Code:
    """A code sample per language. `shared` renders in every guide."""
    py: str = ''
    rs: str = ''
    js: str = ''
    shared: str = ''
    shared_lang: str = 'bash'
    caption: str = ''


@dataclass
class Section:
    title: str
    intro: str = ''
    blocks: list = field(default_factory=list)   # str (html) | Code | Table


@dataclass
class Table:
    headers: list
    rows: list
    caption: str = ''


def p(text):
    """A paragraph. Backtick spans become <code>."""
    escaped = html.escape(text)
    out, parts = [], escaped.split('`')
    for i, part in enumerate(parts):
        out.append(f'<code>{part}</code>' if i % 2 else part)
    return f"<p>{''.join(out)}</p>"


def note(text):
    escaped = html.escape(text)
    parts = escaped.split('`')
    body = ''.join(f'<code>{s}</code>' if i % 2 else s for i, s in enumerate(parts))
    return f'<div class="note">{body}</div>'


SECTIONS = [
    Section(
        'What this API is',
        blocks=[
            p("The FLUX image generator runs FLUX.1 and FLUX.2 diffusion models on your own "
              "hardware and exposes them over HTTP on port 2222. This guide covers the REST "
              "API at /api/v1: generating images, tracking jobs, editing with reference "
              "images, driving the vision model, and managing which checkpoint is loaded."),
            p("Three properties shape every client you will write against it, and each one "
              "causes trouble if you assume otherwise."),
            '<h3>Generation is always asynchronous</h3>',
            p("POST /jobs returns 201 with a job id immediately — even when the queue is "
              "empty. A single worker thread runs every job in turn, because the GPU can "
              "only serve one diffusion pipeline at a time. There is no synchronous "
              "\"generate and return the image\" call, and adding one would only hide the "
              "queue from you. Poll GET /jobs/{id} until `state` leaves `queued` and "
              "`running`."),
            '<h3>Capabilities depend on the loaded model</h3>',
            p("The server runs one checkpoint at a time, chosen from a menu of 14 launcher "
              "configs. Negative prompts require the SDXL backend; inpainting requires "
              "FLUX.2 or SDXL; more than one reference image requires FLUX.2 or the Kontext "
              "editor. Sending an unsupported field is a 400, not a silent no-op, so read "
              "GET /model before you build a request rather than assuming."),
            '<h3>Errors carry a stable code</h3>',
            p("Every failure is the same JSON shape, with the HTTP status carrying the "
              "category and a machine-readable `code` inside. Branch on the code. The "
              "message is written for a human reading a log and is free to change."),
            Code(shared='{"error": {"code": "queue_full",\n'
                        '           "message": "Queue is full (10 max). Cancel a queued job or wait."}}',
                 shared_lang='json'),
        ],
    ),

    Section(
        'Getting started',
        blocks=[
            p("You need the server's API key — the same value it was started with, in "
              "FLUX_API_KEY or its .env file. Everything except the health check and the "
              "self-describing documents requires it."),
            Code(shared='export FLUX_API_KEY=your_key\n'
                        'export FLUX_URL=http://localhost:2222   # optional; this is the default',
                 shared_lang='bash'),
            p("Confirm the server is up and see what it can do. This is the one endpoint "
              "that needs no key, so it is also how you check connectivity before you know "
              "whether your key is right."),
            Code(shared='curl -s localhost:2222/api/v1/health | jq', shared_lang='bash'),
            p("Install the client dependencies:"),
            Code(
                py="# requests is already a server dependency, so the project venv works as-is\n"
                   "pip install requests\n\n"
                   "# The sample client lives in examples/flux_client.py\n"
                   "python examples/flux_client.py health",
                rs="# examples/rust/Cargo.toml\n"
                   "[dependencies]\n"
                   'tokio = { version = "1", features = ["rt-multi-thread", "macros", "time", "fs"] }\n'
                   'reqwest = { version = "0.12", features = ["json", "multipart", "stream"] }\n'
                   'serde = { version = "1", features = ["derive"] }\n'
                   'serde_json = "1"\n'
                   'base64 = "0.22"',
                js="// No dependencies: the client is one ES module using fetch, which\n"
                   "// Node 18+ and every current browser already have.\n"
                   "//\n"
                   "// The sample client lives in examples/flux_client.js\n"
                   "FLUX_API_KEY=your_key node examples/flux_client.js health",
                caption='Dependencies'),
            p("The complete client used throughout this guide ships with the server. "
              "Everything below is an excerpt from it, so you can read the whole thing "
              "for context."),
            Code(
                py="from flux_client import FluxClient\n\n"
                   "flux = FluxClient()                       # reads FLUX_URL / FLUX_API_KEY\n"
                   "flux.wait_until_ready()\n\n"
                   'job = flux.generate("a red fox in falling snow", steps=30)\n'
                   'print(flux.download(job["images"][0]["filename"]))',
                rs="use flux_client::{FluxClient, GenerateRequest};\n\n"
                   "#[tokio::main]\n"
                   "async fn main() -> Result<(), Box<dyn std::error::Error>> {\n"
                   "    let flux = FluxClient::from_env()?;    // FLUX_URL / FLUX_API_KEY\n"
                   "    flux.wait_until_ready(None).await?;\n\n"
                   "    let job = flux\n"
                   '        .generate(&GenerateRequest::new("a red fox in falling snow").steps(30), None)\n'
                   "        .await?;\n\n"
                   "    let path = flux.download(&job.images[0].filename, \".\").await?;\n"
                   '    println!("{}", path.display());\n'
                   "    Ok(())\n"
                   "}",
                js="import { FluxClient } from './flux_client.js';\n\n"
                   "const flux = new FluxClient({\n"
                   "    baseUrl: process.env.FLUX_URL ?? 'http://localhost:2222',\n"
                   "    apiKey: process.env.FLUX_API_KEY,\n"
                   "});\n"
                   "await flux.waitUntilReady();\n\n"
                   "const job = await flux.generate('a red fox in falling snow', { steps: 30 });\n"
                   "console.log(job.images[0].filename, 'seed', job.images[0].seed);",
                caption='The whole thing, end to end'),
        ],
    ),

    Section(
        'Authentication',
        blocks=[
            p("Send the key as an X-API-Key header. An api_key query parameter works too, "
              "for consumers that cannot set headers — an <img> tag pointed at an image "
              "endpoint being the case it exists for. Prefer the header everywhere else: "
              "query strings end up in logs and browser history."),
            Code(
                py="import os, requests\n\n"
                   "session = requests.Session()\n"
                   'session.headers["X-API-Key"] = os.environ["FLUX_API_KEY"]\n\n'
                   'r = session.get("http://localhost:2222/api/v1/model")\n'
                   'print(r.json()["description"])',
                rs="let http = reqwest::Client::new();\n"
                   "let key = std::env::var(\"FLUX_API_KEY\")?;\n\n"
                   "let info: serde_json::Value = http\n"
                   '    .get("http://localhost:2222/api/v1/model")\n'
                   '    .header("X-API-Key", &key)\n'
                   "    .send()\n"
                   "    .await?\n"
                   "    .json()\n"
                   "    .await?;\n"
                   'println!("{}", info["description"]);',
                js="const res = await fetch('http://localhost:2222/api/v1/model', {\n"
                   "    headers: { 'X-API-Key': process.env.FLUX_API_KEY },\n"
                   "});\n"
                   "const info = await res.json();\n"
                   "console.log(info.description);\n\n"
                   "// The query-param form is what an <img> needs, since a tag sends no headers\n"
                   "img.src = `${base}/images/${filename}?api_key=${encodeURIComponent(key)}`;"),
            note("A missing or wrong key is 401 with code `unauthorized`. The key gates "
                 "endpoints that read arbitrary server paths and fetch arbitrary URLs, so "
                 "treat it as a host credential and do not expose the server to untrusted "
                 "networks."),
        ],
    ),

    Section(
        'Readiness and capabilities',
        blocks=[
            p("Loading a checkpoint takes anywhere from seconds to several minutes — a "
              "32B FLUX.2 from cold is the slow end. Until it finishes, generation calls "
              "return 503 with code model_loading. GET /health reports progress and "
              "narrates what it is doing, which is worth surfacing rather than showing a "
              "spinner."),
            Code(
                py='state = flux.health()\n'
                   '# {"ready": false, "status": "loading FLUX.2 model", "elapsed_s": 47.2}\n\n'
                   "if not state['ready']:\n"
                   "    flux.wait_until_ready()      # polls until ready, printing each stage",
                rs="let state = flux.health().await?;\n"
                   "// Health { ready: false, status: \"loading FLUX.2 model\", elapsed_s: 47.2 }\n\n"
                   "if !state.ready {\n"
                   "    flux.wait_until_ready(Some(&|status| println!(\"  {status}...\")))\n"
                   "        .await?;\n"
                   "}",
                js="const state = await flux.health();\n"
                   "// { ready: false, status: 'loading FLUX.2 model', elapsed_s: 47.2 }\n\n"
                   "if (!state.ready) {\n"
                   "    await flux.waitUntilReady({ onStatus: s => console.log(`  ${s}...`) });\n"
                   "}"),
            p("Once ready, ask what the loaded backend can actually do. Build your request "
              "from these flags rather than assuming — the same server binary serves very "
              "different capability sets depending on which config was launched."),
            Code(
                py="info = flux.model()\n"
                   "if info['inpaint']:\n"
                   "    ...        # mask_image is accepted\n"
                   "if info['negative_prompt']:\n"
                   "    ...        # SDXL backend; negative_prompt is accepted\n"
                   "max_refs = 3 if (info['kontext'] or info['flux_version'] == 2) else 1",
                rs="let info = flux.model().await?;\n"
                   "if info.inpaint {\n"
                   "    // mask_image is accepted\n"
                   "}\n"
                   "if info.negative_prompt {\n"
                   "    // SDXL backend; negative_prompt is accepted\n"
                   "}\n"
                   "let max_refs = if info.kontext || info.flux_version == 2 { 3 } else { 1 };",
                js="const info = await flux.model();\n"
                   "if (info.inpaint) {\n"
                   "    // mask_image is accepted\n"
                   "}\n"
                   "if (info.negative_prompt) {\n"
                   "    // SDXL backend; negative_prompt is accepted\n"
                   "}\n"
                   "const maxRefs = info.kontext || info.flux_version === 2 ? 3 : 1;"),
            Table(
                headers=['Field', 'Meaning'],
                rows=[
                    ['model, description', 'Human-readable name of the loaded checkpoint'],
                    ['flux_version', '1 or 2 — FLUX.2 accepts multiple references and masks'],
                    ['negative_prompt', 'SDXL backend; negative_prompt is accepted'],
                    ['inpaint', 'mask_image is accepted'],
                    ['kontext', 'Instruction-editing backend'],
                    ['turbo, schnell', 'Few-step variants — expect fewer steps than requested'],
                    ['vae_tiling', "How the final decode is tiled: auto (the default — "
                                   "tile only above vae_tiling_threshold_mp), always, or off"],
                    ['vae_tiling_threshold_mp', 'Megapixel cutoff auto compares against '
                                                '(1.9 by default)'],
                ],
                caption='GET /model'),
        ],
    ),

    Section(
        'Generating an image',
        blocks=[
            p("Only prompt is required. The call returns as soon as the job is queued; the "
              "helper below then polls until it settles and raises if it failed."),
            Code(
                py='job = flux.generate(\n'
                   '    "a red fox in falling snow, golden hour",\n'
                   '    steps=30,\n'
                   '    size="1.5mp",\n'
                   '    orientation="widescreen",\n'
                   ')\n\n'
                   'for image in job["images"]:\n'
                   '    print(image["filename"], "seed", image["seed"])\n'
                   '    flux.download(image["filename"], "./out")',
                rs='let request = GenerateRequest::new("a red fox in falling snow, golden hour")\n'
                   "    .steps(30)\n"
                   '    .size("1.5mp")\n'
                   '    .orientation("widescreen");\n\n'
                   "let job = flux.generate(&request, None).await?;\n\n"
                   "for image in &job.images {\n"
                   '    println!("{} seed {}", image.filename, image.seed);\n'
                   '    flux.download(&image.filename, "./out").await?;\n'
                   "}",
                js="const job = await flux.generate('a red fox in falling snow, golden hour', {\n"
                   "    steps: 30,\n"
                   "    size: '1.5mp',\n"
                   "    orientation: 'widescreen',\n"
                   "});\n\n"
                   "for (const image of job.images) {\n"
                   "    console.log(image.filename, 'seed', image.seed);\n"
                   "    document.body.append(\n"
                   "        Object.assign(new Image(), { src: flux.imageUrl(image.filename) }));\n"
                   "}"),
            Table(
                headers=['Field', 'Type', 'Default', 'Notes'],
                rows=[
                    ['prompt', 'string', '—', 'Required'],
                    ['steps', 'int', '25', '1–200'],
                    ['batch', 'int', '1', '1–128; prompt encoded once for the batch'],
                    ['seed', 'int', 'random', 'Omit for a fresh seed per image'],
                    ['guidance', 'float', 'model', 'Non-negative'],
                    ['strength', 'float', '0.5', '0–1; img2img only'],
                    ['orientation', 'string', 'landscape',
                     'square, portrait, landscape, widescreen, extra-tall'],
                    ['size', 'string', '1mp', '0.25mp … 2mp'],
                    ['negative_prompt', 'string', '—', 'SDXL only'],
                    ['input_images', 'string[]', '[]', 'Base64 / data URLs, up to 3'],
                    ['input_paths', 'string[]', '[]', 'Paths the server can read'],
                    ['mask_image', 'string', '—', 'Inpainting; needs exactly one reference'],
                    ['aspect_mode', 'string', 'keep', 'keep derives dims from the reference'],
                    ['show_preview', 'bool', 'false', 'Decode latent previews while running'],
                    ['save_previews', 'bool', 'false', 'Also write frames to steps/'],
                ],
                caption='POST /jobs — request body'),
            p("Asking for several images in one job is markedly cheaper than several jobs: "
              "the server encodes the prompt once and reuses the embedding for the whole "
              "batch, which matters because FLUX.2's text encoders are LLM-sized."),
            Code(
                py='job = flux.generate("a lighthouse in a storm", batch=8, steps=28)\n'
                   'print(len(job["images"]), "images in", job["generation_time"], "s")',
                rs='let request = GenerateRequest::new("a lighthouse in a storm")\n'
                   "    .batch(8)\n"
                   "    .steps(28);\n"
                   "let job = flux.generate(&request, None).await?;\n"
                   'println!("{} images in {:.1}s", job.images.len(), job.generation_time);',
                js="const job = await flux.generate('a lighthouse in a storm',\n"
                   "                                { batch: 8, steps: 28 });\n"
                   "console.log(job.images.length, 'images in', job.generation_time, 's');"),
        ],
    ),

    Section(
        'Tracking progress',
        blocks=[
            p("Submit and poll yourself when you want to render progress. A job moves "
              "queued → running → done | failed | canceled; while running, current/batch "
              "track which image is being produced and step/total_steps the diffusion "
              "progress."),
            Code(
                py='queued = flux.submit("a mountain range at dawn", steps=40, show_preview=True)\n'
                   'print("queued at position", queued["position"])\n\n'
                   "while True:\n"
                   "    job = flux.job(queued['id'])\n"
                   "    if job['state'] == 'running':\n"
                   "        print(f\"image {job['current']}/{job['batch']} \"\n"
                   "              f\"step {job['step']}/{job['total_steps']}\")\n"
                   "    if job['state'] in ('done', 'failed', 'canceled'):\n"
                   "        break\n"
                   "    time.sleep(1.5)",
                rs='let queued = flux\n'
                   '    .submit(&GenerateRequest::new("a mountain range at dawn")\n'
                   "        .steps(40)\n"
                   "        .with_previews())\n"
                   "    .await?;\n"
                   'println!("queued at position {:?}", queued.position);\n\n'
                   "let job = flux\n"
                   "    .wait_for_job(&queued.id, Some(&|job| {\n"
                   '        if job.state == "running" {\n'
                   '            println!("image {}/{} step {}/{}",\n'
                   "                job.current, job.batch, job.step, job.total_steps);\n"
                   "        }\n"
                   "    }))\n"
                   "    .await?;",
                js="const queued = await flux.submit('a mountain range at dawn',\n"
                   "                                 { steps: 40, show_preview: true });\n"
                   "console.log('queued at position', queued.position);\n\n"
                   "const job = await flux.waitForJob(queued.id, {\n"
                   "    onProgress: j => {\n"
                   "        if (j.state === 'running') {\n"
                   "            console.log(`image ${j.current}/${j.batch} `\n"
                   "                      + `step ${j.step}/${j.total_steps}`);\n"
                   "        }\n"
                   "    },\n"
                   "});"),
            note("total_steps is not always what you asked for. It is re-read from the live "
                 "scheduler on the first step, because img2img, turbo and schnell variants "
                 "denoise fewer steps than requested. Compute progress from the reported "
                 "total, never from your own step count."),
            '<h3>Intermediate images</h3>',
            p("A diffusion run passes through every intermediate state on its way to the "
              "final image, and you can watch or keep them. GET /jobs/{id}/previews returns "
              "both kinds at once."),
            p("live is the latent being denoised right now, decoded at reduced resolution "
              "and overwritten in place. It exists only while the job runs with "
              "show_preview, and because the file is reused, its ts changes on every frame "
              "— use the supplied url rather than caching the path. frames are per-step "
              "images written to disk, which additionally needs save_previews, but they "
              "persist after the job ends so a finished run can be replayed."),
            Code(
                py='job = flux.submit(prompt, steps=30, show_preview=True, save_previews=True)\n\n'
                   "while True:\n"
                   "    previews = flux.previews(job['id'])\n"
                   "    if previews['live']:\n"
                   "        live = previews['live']\n"
                   "        print(f\"step {live['step']}/{live['total_steps']}  {live['url']}\")\n"
                   "    if previews['state'] in ('done', 'failed', 'canceled'):\n"
                   "        break\n"
                   "    time.sleep(1.0)\n\n"
                   "# Kept frames outlive the job\n"
                   "for frame in flux.previews(job['id'])['frames']:\n"
                   "    print(frame['image'], frame['step'], frame['url'])",
                rs="let queued = flux\n"
                   "    .submit(&GenerateRequest::new(prompt).steps(30).with_previews())\n"
                   "    .await?;\n\n"
                   "loop {\n"
                   "    let previews = flux.previews(&queued.id).await?;\n"
                   "    if let Some(live) = &previews.live {\n"
                   '        println!("step {}/{}  {}", live.step, live.total_steps, live.url);\n'
                   "    }\n"
                   "    if previews.state.as_deref().is_some_and(|s| {\n"
                   '        matches!(s, "done" | "failed" | "canceled")\n'
                   "    }) {\n"
                   "        break;\n"
                   "    }\n"
                   "    tokio::time::sleep(Duration::from_secs(1)).await;\n"
                   "}\n\n"
                   "// Kept frames outlive the job\n"
                   "for frame in flux.previews(&queued.id).await?.frames {\n"
                   '    println!("{:?} {:?} {}", frame.image, frame.step, frame.url);\n'
                   "}",
                js="const queued = await flux.submit(prompt,\n"
                   "    { steps: 30, show_preview: true, save_previews: true });\n\n"
                   "for (;;) {\n"
                   "    const previews = await flux.previews(queued.id);\n"
                   "    if (previews.live) {\n"
                   "        const { step, total_steps, url } = previews.live;\n"
                   "        console.log(`step ${step}/${total_steps}  ${url}`);\n"
                   "        // `url` carries a cache-busting ts — the file is overwritten in place\n"
                   "        preview.src = url;\n"
                   "    }\n"
                   "    if (['done', 'failed', 'canceled'].includes(previews.state)) break;\n"
                   "    await new Promise(r => setTimeout(r, 1000));\n"
                   "}\n\n"
                   "// Kept frames outlive the job\n"
                   "for (const frame of (await flux.previews(queued.id)).frames) {\n"
                   "    console.log(frame.image, frame.step, frame.url);\n"
                   "}"),
            note("Previews cost real time: each one decodes a latent through the VAE. The "
                 "server throttles live previews to at most one every 0.75s for that reason, "
                 "but save_previews bypasses the throttle to capture every step — expect a "
                 "measurably slower generation when you turn it on."),
            p("When you would rather watch than integrate, two endpoints render the same "
              "data as self-refreshing pages: /api/v1/queue.html and "
              "/api/v1/jobs/{id}/previews.html. A browser navigation cannot send a header, "
              "so pass the key in the query string."),
            Code(shared='http://localhost:2222/api/v1/queue.html?api_key=YOUR_KEY\n'
                        'http://localhost:2222/api/v1/jobs/a1b2c3d4e5f6/previews.html?api_key=YOUR_KEY',
                 shared_lang='text', caption='Watch in a browser'),
            p("Cancelling a queued job drops it; cancelling the running one interrupts the "
              "diffusion loop at the next step. Batch images that already finished are kept."),
            Code(
                py="flux.cancel(job['id'])",
                rs="flux.cancel(&job.id).await?;",
                js="await flux.cancel(job.id);"),
            '<h3>Seeing the whole queue</h3>',
            p("One worker serves everyone, so your job may sit behind others. GET /queue "
              "answers where it is in line and roughly when it will run: each waiting entry "
              "carries a 1-based position, and estimated_wait_s extrapolates from recently "
              "completed jobs (null until at least one has finished)."),
            Code(
                py="q = flux.queue()\n"
                   "print(f\"{q['depth']}/{q['capacity']} waiting, \"\n"
                   "      f\"{q['images_pending']} image(s) pending\")\n\n"
                   "for job in q['waiting']:\n"
                   "    print(job['position'], job['id'], job['prompt'][:50])\n\n"
                   "if not q['accepting']:\n"
                   "    ...        # a further submit would fail with queue_full",
                rs="let q = flux.queue().await?;\n"
                   'println!("{}/{} waiting, {} image(s) pending",\n'
                   "    q.depth, q.capacity, q.images_pending);\n\n"
                   "for job in &q.waiting {\n"
                   '    println!("{} {} {}", job.position, job.id,\n'
                   "        job.prompt.chars().take(50).collect::<String>());\n"
                   "}\n\n"
                   "if !q.accepting {\n"
                   "    // a further submit would fail with queue_full\n"
                   "}",
                js="const q = await flux.queue();\n"
                   "console.log(`${q.depth}/${q.capacity} waiting, `\n"
                   "          + `${q.images_pending} image(s) pending`);\n\n"
                   "for (const job of q.waiting) {\n"
                   "    console.log(job.position, job.id, job.prompt.slice(0, 50));\n"
                   "}\n\n"
                   "if (!q.accepting) {\n"
                   "    // a further submit would fail with queue_full\n"
                   "}"),
            note("`accepting` is the honest answer to \"can I submit right now\". It goes "
                 "false once ten jobs are pending; the running job holds no pending slot, so "
                 "a full queue can still have one generating."),
        ],
    ),

    Section(
        'Editing with reference images',
        blocks=[
            p("Reference images travel as base64 data URLs in input_images. One reference "
              "works on every backend; two or three need FLUX.2 or Kontext. With "
              "aspect_mode \"keep\", the output dimensions are derived from the reference's "
              "aspect ratio at your requested megapixel count."),
            Code(
                py='from flux_client import encode_image\n\n'
                   'job = flux.generate(\n'
                   '    "make it winter, heavy snow on the roof",\n'
                   '    input_images=[encode_image("house.jpg")],\n'
                   "    strength=0.55,\n"
                   '    aspect_mode="keep",\n'
                   ")",
                rs="use flux_client::encode_image;\n\n"
                   'let request = GenerateRequest::new("make it winter, heavy snow on the roof")\n'
                   '    .reference(encode_image("house.jpg")?)\n'
                   "    .strength(0.55)\n"
                   "    .keep_aspect();\n\n"
                   "let job = flux.generate(&request, None).await?;",
                js="// toDataUrl takes anything File-like — a drop event, a file input, a Blob\n"
                   "const reference = await FluxClient.toDataUrl(input.files[0]);\n\n"
                   "const job = await flux.generate('make it winter, heavy snow on the roof', {\n"
                   "    input_images: [reference],\n"
                   "    strength: 0.55,\n"
                   "    aspect_mode: 'keep',\n"
                   "});"),
            p("strength controls how far the result may drift from the reference: low "
              "values preserve the original closely, high values treat it as loose "
              "inspiration. It applies to FLUX.1 img2img; Kontext and FLUX.2 read the "
              "prompt as an edit instruction instead."),
            p("When the file already lives on the server, skip the upload entirely. "
              "input_paths accepts absolute paths, ~-prefixed paths, or paths relative to "
              "the output directory; the server loads them and folds them into "
              "input_images for you."),
            Code(
                py='job = flux.generate(\n'
                   '    "make it winter",\n'
                   '    input_paths=["archive/house.jpg"],\n'
                   ")",
                rs='let request = GenerateRequest::new("make it winter")\n'
                   '    .server_path("archive/house.jpg");\n'
                   "let job = flux.generate(&request, None).await?;",
                js="const job = await flux.generate('make it winter', {\n"
                   "    input_paths: ['archive/house.jpg'],\n"
                   "});"),
            '<h3>Inpainting</h3>',
            p("Supply exactly one reference plus a mask, on a FLUX.2 or SDXL backend. "
              "White in the mask marks the region to regenerate."),
            Code(
                py="if flux.model()['inpaint']:\n"
                   "    job = flux.generate(\n"
                   '        "a brass telescope on the table",\n'
                   '        input_images=[encode_image("room.png")],\n'
                   '        mask_image=encode_image("mask.png"),\n'
                   "    )",
                rs="if flux.model().await?.inpaint {\n"
                   '    let request = GenerateRequest::new("a brass telescope on the table")\n'
                   '        .reference(encode_image("room.png")?)\n'
                   '        .mask(encode_image("mask.png")?);\n'
                   "    let job = flux.generate(&request, None).await?;\n"
                   "}",
                js="if ((await flux.model()).inpaint) {\n"
                   "    const job = await flux.generate('a brass telescope on the table', {\n"
                   "        input_images: [await FluxClient.toDataUrl(roomFile)],\n"
                   "        mask_image: await FluxClient.toDataUrl(maskFile),\n"
                   "    });\n"
                   "}"),
        ],
    ),

    Section(
        'Importing references',
        blocks=[
            p("Three endpoints turn something that is not yet a usable reference into one. "
              "All return the same shape — a JPEG data URL bounded to 2048px, plus its "
              "dimensions — so the result drops straight into input_images."),
            Table(
                headers=['Endpoint', 'Takes', 'For'],
                rows=[
                    ['POST /imports/raw', 'multipart file',
                     'Camera RAW (NEF/DNG/CR3) a browser cannot decode'],
                    ['POST /imports/url', '{"url"}',
                     'Remote images, fetched server-side with no CORS limits'],
                    ['POST /imports/path', '{"path"}', 'Files on the server\'s own disk'],
                ]),
            Code(
                py='ref = flux.import_url("https://example.com/photo.jpg")\n'
                   'print(ref["width"], ref["height"])\n\n'
                   'job = flux.generate("in the style of a woodcut", input_images=[ref["image"]])\n\n'
                   '# Camera RAW needs the rawpy package on the server (501 if absent)\n'
                   'raw = flux.import_raw("DSC_0001.NEF")',
                rs='let reference = flux.import_url("https://example.com/photo.jpg").await?;\n'
                   'println!("{}x{}", reference.width, reference.height);\n\n'
                   'let request = GenerateRequest::new("in the style of a woodcut")\n'
                   "    .reference(reference.image);\n"
                   "let job = flux.generate(&request, None).await?;\n\n"
                   "// Camera RAW needs the rawpy package on the server (501 if absent)\n"
                   'let raw = flux.import_raw("DSC_0001.NEF").await?;',
                js="const reference = await flux.importUrl('https://example.com/photo.jpg');\n"
                   "console.log(reference.width, reference.height);\n\n"
                   "const job = await flux.generate('in the style of a woodcut',\n"
                   "                                { input_images: [reference.image] });\n\n"
                   "// Camera RAW needs the rawpy package on the server (501 if absent).\n"
                   "// importRaw posts multipart, so it takes the File itself, not a data URL.\n"
                   "const raw = await flux.importRaw(rawInput.files[0]);"),
            p("To find server-side paths, browse the filesystem. Relative directories "
              "resolve against the output folder; the response always carries an absolute "
              "dir and its parent so you can navigate without doing path arithmetic."),
            Code(
                py='listing = flux.browse("archive")\n'
                   'for name in listing["dirs"]:\n'
                   '    print("[dir]", name)\n'
                   'for entry in listing["files"]:\n'
                   '    print(entry["filename"])',
                rs='let listing = flux.browse(Some("archive")).await?;\n'
                   "for name in &listing.dirs {\n"
                   '    println!("[dir] {name}");\n'
                   "}\n"
                   "for file in &listing.files {\n"
                   '    println!("{}", file.filename);\n'
                   "}",
                js="const listing = await flux.browse('archive');\n"
                   "for (const name of listing.dirs) console.log('[dir]', name);\n"
                   "for (const entry of listing.files) console.log(entry.filename);\n\n"
                   "// Entries carry only a filename; join them onto the absolute\n"
                   "// listing.dir to get a path the thumbnail endpoint can resolve.\n"
                   "grid.append(...listing.files.map(entry => Object.assign(new Image(),\n"
                   "    { src: flux.thumbnailUrl(`${listing.dir}/${entry.filename}`) })));"),
        ],
    ),

    Section(
        'Images and housekeeping',
        blocks=[
            p("Generated images are written to the server's output directory and named "
              "flux{1|2}_{date}_{time}_{hex}.png, each with a .prompt sidecar recording how "
              "it was made. The listing covers today's output, newest first."),
            Code(
                py="for image in flux.history():\n"
                   "    print(image['time'], image['filename'], image['prompt'])\n\n"
                   "data = flux.image_bytes(filename)      # raw bytes\n"
                   "path = flux.download(filename, './out')",
                rs="for image in flux.history().await? {\n"
                   '    println!("{} {} {:?}", image.time, image.filename, image.prompt);\n'
                   "}\n\n"
                   "let data = flux.image_bytes(&filename).await?;      // raw bytes\n"
                   'let path = flux.download(&filename, "./out").await?;',
                js="for (const image of await flux.history()) {\n"
                   "    console.log(image.time, image.filename, image.prompt);\n"
                   "}\n\n"
                   "const blob = await flux.imageBlob(filename);   // a Blob, for canvas or download\n"
                   "img.src = flux.imageUrl(filename);             // or let the tag fetch it"),
            p("Housekeeping has three levels. Saving copies an image somewhere the other "
              "two cannot reach; archiving moves the day's work aside non-destructively; "
              "deleting is permanent."),
            Code(
                py="flux.save_image(filename)     # copy into .saved/, beyond archive and delete\n"
                   "flux.archive()                # move today's output into archive/\n"
                   "flux.delete_image(filename)   # permanent\n"
                   "flux.delete_today()           # permanent, everything from today",
                rs="flux.save_image(&filename).await?;   // copy into .saved/\n"
                   "flux.archive().await?;               // move today's output into archive/\n"
                   "flux.delete_image(&filename).await?; // permanent\n"
                   "flux.delete_today().await?;          // permanent, everything from today",
                js="await flux.saveImage(filename);    // copy into .saved/, beyond archive and delete\n"
                   "await flux.archive();              // move today's output into archive/\n"
                   "await flux.deleteImage(filename);  // permanent\n"
                   "await flux.deleteToday();          // permanent, everything from today"),
            note("Both delete calls are irreversible and the delete-today form takes no "
                 "confirmation. Archive first if there is any doubt."),
        ],
    ),

    Section(
        'Vision-model jobs',
        blocks=[
            p("A local vision model (via ollama) backs three features. They share one "
              "resource, discriminated by task, because a single call can run for minutes "
              "— well past any sensible request timeout. POST returns 202 with an id; poll "
              "GET /vlm/jobs/{id}."),
            Table(
                headers=['Task', 'Takes', 'Returns'],
                rows=[
                    ['describe', 'images[], think',
                     'A prompt that would recreate the photo'],
                    ['boost', 'prompt, level 1–5, has_image',
                     'The prompt rewritten in the model\'s idiom'],
                    ['critique', 'direction, ref_image, output_filename',
                     'A judgment plus a revised instruction'],
                ]),
            '<h3>Photo to prompt</h3>',
            p("Hand it one to three images and get back a prompt describing them — with "
              "several, it composes a single scene combining them."),
            Code(
                py='prompt = flux.describe("photo.jpg")\n'
                   "print(prompt)\n\n"
                   "# then generate a fresh image from that description alone\n"
                   "job = flux.generate(prompt)",
                rs='let prompt = flux.describe(&[encode_image("photo.jpg")?], false).await?;\n'
                   'println!("{prompt}");\n\n'
                   "// then generate a fresh image from that description alone\n"
                   "let job = flux.generate(&GenerateRequest::new(&prompt), None).await?;",
                js="const prompt = await flux.describe(\n"
                   "    await FluxClient.toDataUrl(photoInput.files[0]));\n"
                   "console.log(prompt);\n\n"
                   "// then generate a fresh image from that description alone\n"
                   "const job = await flux.generate(prompt);"),
            '<h3>Improving a prompt</h3>',
            p("Boost rewrites a draft into the prompting idiom of whichever model is "
              "loaded — descriptive prose for FLUX, an imperative instruction for Kontext, "
              "tag phrases for SDXL. level runs 1 to 5, from polishing the wording to "
              "reimagining the idea. Pass has_image when references will be attached: the "
              "idiom shifts to edit instructions."),
            Code(
                py='result = flux.boost("a castle", level=4)\n'
                   'job = flux.generate(result["prompt"])',
                rs='let prompt = flux.boost("a castle", 4, false).await?;\n'
                   "let job = flux.generate(&GenerateRequest::new(&prompt), None).await?;",
                js="const result = await flux.boost('a castle', { level: 4 });\n"
                   "const job = await flux.generate(result.prompt);"),
            note("A vision-model failure polls as 502 with code `vlm_failed` — the request "
                 "was fine, the model was not. In practice this means ollama is not running "
                 "or the model has not been pulled. GET /telemetry reports its residency."),
        ],
    ),

    Section(
        'Switching models',
        blocks=[
            p("The checkpoints are far too large to hot-swap in-process, so switching is a "
              "supervised restart: the server records the target config, exits with a "
              "status its supervisor recognises, and comes back on the new one. PUT "
              "/models/current answers 202 — the restart is scheduled, not finished."),
            Code(
                py="for config in flux.models()['configs']:\n"
                   "    print(config['id'], config['label'])\n\n"
                   "flux.switch_model(9)     # waits through the restart by default",
                rs="for config in &flux.models().await?.configs {\n"
                   '    println!("{} {}", config.id, config.label);\n'
                   "}\n\n"
                   "flux.switch_model(9).await?;\n"
                   "tokio::time::sleep(Duration::from_secs(3)).await;   // let the old process exit\n"
                   "flux.wait_until_ready(None).await?;",
                js="for (const config of (await flux.models()).configs) {\n"
                   "    console.log(config.id, config.label);\n"
                   "}\n\n"
                   "await flux.switchModel(9);\n"
                   "await new Promise(r => setTimeout(r, 3000));   // let the old process exit\n"
                   "await flux.waitUntilReady();"),
            note("Expect connection failures while polling across a restart — the process "
                 "genuinely is gone for a moment. Both sample clients treat transport "
                 "errors during a readiness wait as \"still restarting\" rather than fatal. "
                 "If GET /models reports switchable: false the server was started without "
                 "its supervisor and cannot restart itself; you will get 400 no_supervisor."),
            '<h3>Comparing models on one prompt</h3>',
            p("A multi-model run generates the same prompt on several configs in turn, "
              "sharing one seed so the outputs are genuinely comparable. Because each "
              "config change restarts the process, the run is file-backed and survives "
              "them. Text-to-image only, and at most one run at a time."),
            Code(
                py='run = flux.multi_run("a red fox in snow", configs=[9, 6, 1], steps=28)\n'
                   'print("seed", run["seed"])\n'
                   "flux.wait_for_multi_run()     # follows it across the restarts",
                rs='let run = flux.multi_run("a red fox in snow", &[9, 6, 1], Some(28)).await?;\n'
                   'println!("seed {}", run.seed);\n\n'
                   "while let Some(state) = flux.multi_run_status().await? {\n"
                   "    if !state.active { break; }\n"
                   "    tokio::time::sleep(Duration::from_secs(5)).await;\n"
                   "}",
                js="const run = await flux.multiRun('a red fox in snow', [9, 6, 1], { steps: 28 });\n"
                   "console.log('seed', run.seed);\n\n"
                   "// multiRunStatus resolves to null once the run is gone; the server is\n"
                   "// unreachable across each restart, so treat a transport error as 'still going'\n"
                   "for (;;) {\n"
                   "    const state = await flux.multiRunStatus().catch(() => ({ active: true }));\n"
                   "    if (!state?.active) break;\n"
                   "    await new Promise(r => setTimeout(r, 5000));\n"
                   "}"),
        ],
    ),

    Section(
        'Handling errors',
        blocks=[
            p("Match on the code. These are the ones worth handling distinctly: "
              "model_loading and queue_full are both temporary and worth retrying, busy "
              "means something else must finish first, and invalid_request means the "
              "request will never succeed as written."),
            Table(
                headers=['Code', 'Status', 'Meaning'],
                rows=[
                    ['unauthorized', '401', 'Missing or wrong API key'],
                    ['invalid_request', '400', 'A parameter failed validation'],
                    ['not_found', '404', 'No such job, image, run, or path'],
                    ['busy', '409', 'Jobs in flight block this operation'],
                    ['queue_full', '429', 'Ten jobs already pending'],
                    ['model_loading', '503', 'Model not up yet — poll /health'],
                    ['no_supervisor', '400', 'Server cannot restart itself'],
                    ['undecodable_image', '400', 'A reference or mask could not be decoded'],
                    ['fetch_failed', '400', 'A remote URL could not be retrieved'],
                    ['too_large', '400/413', 'Over the 64MB cap'],
                    ['vlm_failed', '502', 'The vision model failed'],
                    ['io_error', '500', 'A filesystem operation failed'],
                ]),
            Code(
                py="from flux_client import FluxError\n\n"
                   "try:\n"
                   "    job = flux.generate(prompt)\n"
                   "except FluxError as e:\n"
                   "    if e.code == 'queue_full':\n"
                   "        time.sleep(30)          # transient — retry\n"
                   "    elif e.code == 'model_loading':\n"
                   "        flux.wait_until_ready()\n"
                   "    elif e.code == 'invalid_request':\n"
                   "        raise                   # the request itself is wrong\n"
                   "    else:\n"
                   "        raise",
                rs="use flux_client::Error;\n\n"
                   "match flux.generate(&request, None).await {\n"
                   "    Ok(job) => { /* ... */ }\n"
                   '    Err(e) if e.code() == Some("queue_full") => {\n'
                   "        tokio::time::sleep(Duration::from_secs(30)).await;  // transient\n"
                   "    }\n"
                   '    Err(e) if e.code() == Some("model_loading") => {\n'
                   "        flux.wait_until_ready(None).await?;\n"
                   "    }\n"
                   "    Err(e) => return Err(e),    // including invalid_request\n"
                   "}",
                js="import { FluxError } from './flux_client.js';\n\n"
                   "try {\n"
                   "    const job = await flux.generate(prompt);\n"
                   "} catch (e) {\n"
                   "    if (!(e instanceof FluxError)) throw e;   // transport, not a rejection\n"
                   "    if (e.code === 'queue_full') {\n"
                   "        await new Promise(r => setTimeout(r, 30_000));   // transient — retry\n"
                   "    } else if (e.code === 'model_loading') {\n"
                   "        await flux.waitUntilReady();\n"
                   "    } else {\n"
                   "        throw e;                              // including invalid_request\n"
                   "    }\n"
                   "}"),
            p("Transport failures are a separate category from rejections, and both clients "
              "keep them distinct. During a model switch the server is legitimately absent "
              "for a while, so a connection error there means \"wait\", not \"fail\"."),
        ],
    ),

    Section(
        'Endpoint reference',
        blocks=[
            p("The complete surface. A running server serves this as a browsable page at "
              "/api/v1/docs and as an OpenAPI 3.1 document at /api/v1/openapi.json, either "
              "of which can generate a client for a language not covered here."),
            Table(
                headers=['Method', 'Path', 'Purpose'],
                rows=[
                    ['GET', '/api/v1', 'Service index (public)'],
                    ['GET', '/api/v1/health', 'Readiness (public)'],
                    ['GET', '/api/v1/openapi.json', 'OpenAPI document (public)'],
                    ['GET', '/api/v1/docs', 'Browsable reference (public)'],
                    ['GET', '/api/v1/model', 'Loaded backend capabilities'],
                    ['GET', '/api/v1/models', 'Available configs'],
                    ['GET', '/api/v1/models/current', 'Running config'],
                    ['PUT', '/api/v1/models/current', 'Switch config (restarts)'],
                    ['GET', '/api/v1/telemetry', 'GPU power, vision-model residency'],
                    ['POST', '/api/v1/jobs', 'Queue a generation job'],
                    ['GET', '/api/v1/queue', 'Ordered queue with wait estimate'],
                    ['GET', '/api/v1/queue.html', 'Live HTML view of the queue'],
                    ['GET', '/api/v1/jobs', 'Running, queued, recent jobs'],
                    ['GET', '/api/v1/jobs/{id}', 'One job'],
                    ['DELETE', '/api/v1/jobs/{id}', 'Cancel or interrupt'],
                    ['GET', '/api/v1/jobs/{id}/previews', 'Intermediate images'],
                    ['GET', '/api/v1/jobs/{id}/previews.html', 'Live HTML view of those'],
                    ['DELETE', '/api/v1/jobs/recent', 'Clear finished list'],
                    ['GET', '/api/v1/images', "Today's images"],
                    ['GET', '/api/v1/images/{name}', 'Image bytes'],
                    ['DELETE', '/api/v1/images/{name}', 'Delete one'],
                    ['DELETE', '/api/v1/images', "Delete all of today's"],
                    ['POST', '/api/v1/images/{name}/save', 'Preserve in .saved/'],
                    ['POST', '/api/v1/archive', "Archive today's output"],
                    ['POST', '/api/v1/filmstrips', 'Compose an edit-loop strip'],
                    ['POST', '/api/v1/imports/raw', 'Camera RAW to reference'],
                    ['POST', '/api/v1/imports/url', 'Remote image to reference'],
                    ['POST', '/api/v1/imports/path', 'Server file to reference'],
                    ['GET', '/api/v1/files', 'Browse server directories'],
                    ['GET', '/api/v1/files/thumbnail', 'Thumbnail of a server image'],
                    ['POST', '/api/v1/vlm/jobs', 'Start describe/boost/critique'],
                    ['GET', '/api/v1/vlm/jobs/{id}', 'Poll a vision job'],
                    ['POST', '/api/v1/multi-runs', 'Run one prompt on many configs'],
                    ['GET', '/api/v1/multi-runs/current', 'Active run'],
                    ['DELETE', '/api/v1/multi-runs/current', 'Cancel or dismiss'],
                ]),
        ],
    ),
]


# ---------------------------------------------------------------- render ----

def render_code(source, lexer_name):
    lexer = get_lexer_by_name(lexer_name)
    formatter = HtmlFormatter(noclasses=True, nowrap=False, style='friendly')
    return highlight(source, lexer, formatter)


def render_table(table):
    head = ''.join(f'<th>{html.escape(h)}</th>' for h in table.headers)
    body = ''.join(
        '<tr>' + ''.join(f'<td>{html.escape(str(c))}</td>' for c in row) + '</tr>'
        for row in table.rows)
    caption = f'<div class="caption">{html.escape(table.caption)}</div>' if table.caption else ''
    return f'{caption}<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>'


def render_block(block, language):
    if isinstance(block, str):
        return block
    if isinstance(block, Table):
        return render_table(block)

    if block.shared:
        source, lexer = block.shared, block.shared_lang
    else:
        source = getattr(block, LANGUAGES[language]['attr'])
        lexer = LANGUAGES[language]['lexer']
    caption = f'<div class="caption">{html.escape(block.caption)}</div>' if block.caption else ''
    return f'<div class="code">{caption}{render_code(source.rstrip(), lexer)}</div>'


def build_html(language):
    meta = LANGUAGES[language]
    parts = [_COVER.format(lang=meta['title'], built=date.today().isoformat())]

    toc = ''.join(
        f'<li><span class="n">{i}</span>{html.escape(s.title)}</li>'
        for i, s in enumerate(SECTIONS, 1))
    parts.append(f'<section class="toc"><h1>Contents</h1><ol>{toc}</ol></section>')

    for i, section in enumerate(SECTIONS, 1):
        body = ''.join(render_block(b, language) for b in section.blocks)
        parts.append(
            f'<section><h1><span class="num">{i}</span>{html.escape(section.title)}</h1>'
            f'{section.intro}{body}</section>')

    return _PAGE.format(lang=meta['title'], body=''.join(parts), css=_CSS)


_COVER = """
<section class="cover">
  <div class="mark">FLUX</div>
  <h1>Image Generator</h1>
  <h2>REST API Developer Guide</h2>
  <div class="lang">{lang}</div>
  <div class="meta">API v1 &middot; built {built}</div>
</section>
"""

_CSS = """
@page { size: A4; margin: 20mm 18mm; }
* { box-sizing: border-box; }
body {
  margin: 0; color: #1a1a1a; background: #fff;
  font: 10.5pt/1.6 "Segoe UI", -apple-system, system-ui, sans-serif;
  -webkit-print-color-adjust: exact; print-color-adjust: exact;
}
section { break-before: page; }
section.cover, section.cover + section { break-before: auto; }
section.cover { break-after: page; }

.cover { padding-top: 55mm; text-align: left; }
.cover .mark {
  font-size: 13pt; letter-spacing: .42em; font-weight: 700;
  color: #d9480f; margin-bottom: 26mm;
}
.cover h1 { font-size: 33pt; line-height: 1.1; margin: 0; border: 0; padding: 0; }
.cover h2 {
  font-size: 16pt; font-weight: 400; color: #555;
  margin: 5mm 0 0; border: 0; padding: 0;
}
.cover .lang {
  margin-top: 22mm; font-size: 20pt; font-weight: 600; color: #d9480f;
}
.cover .meta { margin-top: 3mm; color: #888; font-size: 9.5pt; }

h1 {
  font-size: 18pt; margin: 0 0 6mm; padding-bottom: 2.5mm;
  border-bottom: 2px solid #d9480f;
}
h1 .num {
  display: inline-block; min-width: 11mm; color: #d9480f; font-weight: 700;
}
h3 {
  font-size: 12pt; margin: 7mm 0 2mm; color: #111;
  break-after: avoid;
}
p { margin: 0 0 3.6mm; text-align: left; orphans: 2; widows: 2; }

code {
  font: 9.2pt "Cascadia Mono", "SF Mono", Consolas, monospace;
  background: #f2f2f4; padding: .4mm 1.1mm; border-radius: 1mm; color: #b03000;
}

.code {
  margin: 0 0 5mm; break-inside: avoid;
  border: 1px solid #e3e3e6; border-radius: 1.6mm; overflow: hidden;
}
.code .caption {
  background: #f7f7f9; border-bottom: 1px solid #e3e3e6;
  padding: 1.4mm 3mm; font-size: 8.6pt; font-weight: 600;
  color: #555; text-transform: uppercase; letter-spacing: .05em;
}
.code pre {
  margin: 0; padding: 3mm; background: #fbfbfc; overflow-x: auto;
  font: 8.9pt/1.5 "Cascadia Mono", "SF Mono", Consolas, monospace;
  white-space: pre-wrap; word-break: break-word;
}

table {
  width: 100%; border-collapse: collapse; margin: 0 0 5mm;
  font-size: 9.2pt; break-inside: avoid;
}
th {
  text-align: left; background: #f7f7f9; border-bottom: 1.4px solid #d0d0d4;
  padding: 1.7mm 2.2mm; font-weight: 600;
}
td { border-bottom: 1px solid #ececef; padding: 1.5mm 2.2mm; vertical-align: top; }
td:first-child { font-family: "Cascadia Mono", Consolas, monospace; font-size: 8.7pt; }
.caption {
  font-size: 8.6pt; font-weight: 600; color: #555;
  text-transform: uppercase; letter-spacing: .05em; margin-bottom: 1.4mm;
}

.note {
  margin: 0 0 5mm; padding: 2.6mm 3.4mm; break-inside: avoid;
  background: #fff8f3; border-left: 3px solid #d9480f; border-radius: 0 1.4mm 1.4mm 0;
  font-size: 9.8pt;
}

.toc ol { list-style: none; padding: 0; margin: 0; font-size: 11.5pt; }
.toc li { padding: 2.2mm 0; border-bottom: 1px solid #ececef; }
.toc .n {
  display: inline-block; min-width: 10mm; color: #d9480f; font-weight: 700;
}
"""

_PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<title>FLUX Image Generator — REST API Developer Guide ({lang})</title>
<style>{css}</style>
</head><body>{body}</body></html>
"""


# ------------------------------------------------------------------ build ----

def find_chrome():
    for candidate in CHROME_CANDIDATES:
        if candidate and os.path.isfile(candidate):
            return candidate
    found = shutil.which('chrome') or shutil.which('google-chrome') or shutil.which('chromium')
    if found:
        return found
    raise SystemExit(
        "Chrome not found — it renders the PDF. Install it, or add its path to "
        "CHROME_CANDIDATES.")


def html_to_pdf(chrome, html_path, pdf_path):
    profile = tempfile.mkdtemp(prefix='flux-guide-')
    try:
        result = subprocess.run(
            [chrome, '--headless=new', '--disable-gpu', '--no-sandbox',
             f'--user-data-dir={profile}',
             '--no-pdf-header-footer',
             '--virtual-time-budget=10000',
             f'--print-to-pdf={pdf_path}',
             'file:///' + html_path.replace('\\', '/')],
            capture_output=True, text=True, timeout=180)
        if not os.path.isfile(pdf_path):
            raise SystemExit(
                f"Chrome produced no PDF.\nstdout: {result.stdout}\nstderr: {result.stderr}")
    finally:
        shutil.rmtree(profile, ignore_errors=True)


def validate():
    """Every code block must cover every language, or a guide loses a sample."""
    problems = []
    for section in SECTIONS:
        for block in section.blocks:
            if isinstance(block, Code) and not block.shared:
                for language, meta in LANGUAGES.items():
                    if not getattr(block, meta['attr']).strip():
                        problems.append(f"{section.title}: missing {meta['title']} sample")
    if problems:
        raise SystemExit("Incomplete samples:\n  " + "\n  ".join(problems))


def main():
    validate()
    os.makedirs(OUT_DIR, exist_ok=True)
    chrome = find_chrome()

    for language in LANGUAGES:
        stem = f"flux-api-guide-{LANGUAGES[language]['file']}"
        html_path = os.path.join(OUT_DIR, stem + '.html')
        pdf_path = os.path.join(OUT_DIR, stem + '.pdf')

        with open(html_path, 'w', encoding='utf-8') as f:
            f.write(build_html(language))
        html_to_pdf(chrome, html_path, pdf_path)

        size = os.path.getsize(pdf_path)
        print(f"  {os.path.relpath(pdf_path, ROOT)}  ({size / 1024:.0f} KB)")

    print(f"\n{len(SECTIONS)} sections, {len(LANGUAGES)} guides.")
    return 0


if __name__ == '__main__':
    sys.exit(main())
