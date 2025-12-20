import os
import argparse
import threading
import time
from datetime import datetime
import uuid
from flask import Flask, request, jsonify, send_from_directory

# Import model components from fl24bit
from fl24bit import load_model, generate_image, device, save_prompt_file

app = Flask(__name__)

# Will be set by command-line args
_local_encoder = False
_full_model = False

# Configuration
OUTPUT_DIR = "web-generated"
PORT = 2222

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Lock for sequential processing
_generation_lock = threading.Lock()
_current_status = {"generating": False, "prompt": None}

# Orientation presets (width, height) at 1K base
ORIENTATIONS_1K = {
    'square': (1024, 1024),
    'portrait': (768, 1344),
    'landscape': (1344, 768),
}

SIZES = {
    '1mp': 1.0,
    '2mp': 2.0,
    '4mp': 4.0,
}

HTML_PAGE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>FLUX.2 Image Generator</title>
    <style>
        * { box-sizing: border-box; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            max-width: 900px;
            margin: 0 auto;
            padding: 20px;
            background: #1a1a2e;
            color: #eee;
        }
        h1 { color: #00d4ff; margin-bottom: 5px; }
        .subtitle { color: #888; margin-bottom: 20px; }
        .form-group { margin-bottom: 15px; }
        label { display: block; margin-bottom: 5px; color: #aaa; }
        input[type="text"], select {
            width: 100%;
            padding: 12px;
            border: 1px solid #333;
            border-radius: 6px;
            background: #16213e;
            color: #fff;
            font-size: 16px;
        }
        input[type="text"]:focus, select:focus {
            outline: none;
            border-color: #00d4ff;
        }
        .row { display: flex; gap: 15px; }
        .row .form-group { flex: 1; }
        button {
            width: 100%;
            padding: 15px;
            background: #00d4ff;
            color: #000;
            border: none;
            border-radius: 6px;
            font-size: 18px;
            font-weight: bold;
            cursor: pointer;
            transition: background 0.2s;
        }
        button:hover:not(:disabled) { background: #00b8e6; }
        button:disabled {
            background: #444;
            color: #888;
            cursor: not-allowed;
        }
        .status {
            text-align: center;
            padding: 15px;
            margin: 20px 0;
            border-radius: 6px;
            display: none;
        }
        .status.generating {
            display: block;
            background: #2d2d44;
            color: #00d4ff;
        }
        .status.error {
            display: block;
            background: #442d2d;
            color: #ff6b6b;
        }
        .result {
            margin-top: 20px;
            text-align: center;
            display: none;
        }
        .result.visible { display: block; }
        .image-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
        }
        .image-card {
            background: #16213e;
            border-radius: 8px;
            padding: 15px;
            text-align: center;
        }
        .image-card img {
            max-width: 100%;
            border-radius: 6px;
            box-shadow: 0 4px 20px rgba(0,0,0,0.5);
        }
        .image-card .actions {
            margin-top: 12px;
            display: flex;
            gap: 10px;
            justify-content: center;
        }
        .image-card a {
            display: inline-block;
            padding: 8px 16px;
            background: #28a745;
            color: #fff;
            text-decoration: none;
            border-radius: 6px;
            font-weight: bold;
            font-size: 14px;
        }
        .image-card a:hover { background: #218838; }
        .image-card .seed-btn {
            background: #6c757d;
            cursor: pointer;
        }
        .image-card .seed-btn:hover { background: #5a6268; }
        .image-card .info {
            color: #888;
            font-size: 13px;
            margin-top: 8px;
        }
        .timings {
            display: flex;
            flex-wrap: wrap;
            justify-content: center;
            gap: 8px;
            margin-top: 10px;
            padding: 8px;
            background: #1a1a2e;
            border-radius: 4px;
            font-size: 12px;
        }
        .timing-item {
            color: #aaa;
        }
        .timing-label {
            color: #00d4ff;
        }
        .timing-total {
            color: #fff;
            font-weight: bold;
        }
        .timing-total .timing-label {
            color: #28a745;
        }
        .generation-info {
            color: #888;
            font-size: 14px;
            margin-top: 15px;
            text-align: center;
        }
        .spinner {
            display: inline-block;
            width: 20px;
            height: 20px;
            border: 3px solid #00d4ff;
            border-top-color: transparent;
            border-radius: 50%;
            animation: spin 1s linear infinite;
            margin-right: 10px;
            vertical-align: middle;
        }
        @keyframes spin { to { transform: rotate(360deg); } }
    </style>
</head>
<body>
    <h1>FLUX.2 Image Generator</h1>
    <p class="subtitle" id="modelInfo">Loading model info...</p>

    <form id="generateForm">
        <div class="form-group">
            <label for="prompt">Prompt</label>
            <input type="text" id="prompt" name="prompt" placeholder="A majestic mountain landscape at sunset..." required>
        </div>

        <div class="row">
            <div class="form-group">
                <label for="orientation">Orientation</label>
                <select id="orientation" name="orientation">
                    <option value="landscape" selected>Landscape</option>
                    <option value="portrait">Portrait</option>
                    <option value="square">Square</option>
                </select>
            </div>
            <div class="form-group">
                <label for="size">Size</label>
                <select id="size" name="size">
                    <option value="1mp" selected>1 MP</option>
                    <option value="2mp">2 MP</option>
                    <option value="4mp">4 MP</option>
                </select>
            </div>
            <div class="form-group">
                <label for="steps">Steps</label>
                <select id="steps" name="steps">
                    <option value="10">10</option>
                    <option value="15">15</option>
                    <option value="20">20</option>
                    <option value="25" selected>25</option>
                    <option value="30">30</option>
                    <option value="40">40</option>
                    <option value="50">50</option>
                </select>
            </div>
        </div>

        <div class="row">
            <div class="form-group">
                <label for="seed">Seed (optional)</label>
                <input type="text" id="seed" name="seed" placeholder="Random if empty">
            </div>
            <div class="form-group">
                <label for="batch">Batch Size</label>
                <select id="batch" name="batch">
                    <option value="1" selected>1 image</option>
                    <option value="2">2 images</option>
                    <option value="3">3 images</option>
                    <option value="4">4 images</option>
                </select>
            </div>
        </div>

        <button type="submit" id="submitBtn">Generate Image</button>
    </form>

    <div class="status" id="status">
        <span class="spinner"></span>
        <span id="statusText">Generating...</span>
    </div>

    <div class="result" id="result">
        <div class="image-grid" id="imageGrid"></div>
        <p class="generation-info" id="generationInfo"></p>
    </div>

    <script>
        const form = document.getElementById('generateForm');
        const submitBtn = document.getElementById('submitBtn');
        const status = document.getElementById('status');
        const statusText = document.getElementById('statusText');
        const result = document.getElementById('result');
        const imageGrid = document.getElementById('imageGrid');
        const generationInfo = document.getElementById('generationInfo');

        function useSeed(seed) {
            document.getElementById('seed').value = seed;
        }

        // Fetch and display model info on page load
        fetch('/model-info')
            .then(r => r.json())
            .then(data => {
                document.getElementById('modelInfo').textContent = data.description;
            })
            .catch(() => {
                document.getElementById('modelInfo').textContent = 'FLUX.2 Image Generator';
            });

        form.addEventListener('submit', async (e) => {
            e.preventDefault();

            const seedValue = document.getElementById('seed').value.trim();
            const batch = parseInt(document.getElementById('batch').value);

            const formData = {
                prompt: document.getElementById('prompt').value,
                orientation: document.getElementById('orientation').value,
                size: document.getElementById('size').value,
                steps: parseInt(document.getElementById('steps').value),
                seed: seedValue ? parseInt(seedValue) : null,
                batch: batch
            };

            submitBtn.disabled = true;
            status.className = 'status generating';
            statusText.textContent = batch > 1 ? `Generating ${batch} images...` : 'Generating image...';
            result.className = 'result';
            imageGrid.innerHTML = '';

            try {
                const response = await fetch('/generate', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(formData)
                });

                const data = await response.json();

                if (data.success) {
                    status.className = 'status';
                    result.className = 'result visible';

                    const t = Date.now();
                    data.images.forEach((img, i) => {
                        const card = document.createElement('div');
                        card.className = 'image-card';
                        const timings = img.timings;
                        card.innerHTML = `
                            <img src="/images/${img.filename}?t=${t}" alt="Generated image ${i+1}">
                            <div class="actions">
                                <a href="/images/${img.filename}" download="${img.filename}">Download</a>
                                <a href="#" class="seed-btn" onclick="useSeed(${img.seed}); return false;">Use Seed</a>
                            </div>
                            <p class="info">Seed: ${img.seed}</p>
                            <div class="timings">
                                <span class="timing-item"><span class="timing-label">Encode:</span> ${timings.encoding}s</span>
                                <span class="timing-item"><span class="timing-label">Diffuse:</span> ${timings.diffusion}s</span>
                                <span class="timing-item"><span class="timing-label">Save:</span> ${timings.save}s</span>
                                <span class="timing-item timing-total"><span class="timing-label">Total:</span> ${timings.total}s</span>
                            </div>
                        `;
                        imageGrid.appendChild(card);
                    });

                    generationInfo.textContent = `Generated ${data.images.length} image(s) in ${data.generation_time.toFixed(1)}s`;
                } else {
                    status.className = 'status error';
                    statusText.textContent = 'Error: ' + data.error;
                }
            } catch (err) {
                status.className = 'status error';
                statusText.textContent = 'Error: ' + err.message;
            }

            submitBtn.disabled = false;
        });
    </script>
</body>
</html>
"""


@app.route('/')
def index():
    return HTML_PAGE


@app.route('/generate', methods=['POST'])
def generate():
    global _current_status

    # Check if already generating
    if not _generation_lock.acquire(blocking=False):
        return jsonify({
            'success': False,
            'error': 'Generation in progress. Please wait.'
        }), 503

    try:
        data = request.json
        prompt = data.get('prompt', '').strip()
        if not prompt:
            return jsonify({'success': False, 'error': 'Prompt is required'}), 400

        orientation = data.get('orientation', 'landscape')
        size = data.get('size', '1mp')
        steps = int(data.get('steps', 25))
        seed = data.get('seed')  # None if not provided
        batch = min(max(int(data.get('batch', 1)), 1), 4)  # Clamp to 1-4

        # Calculate dimensions
        base_w, base_h = ORIENTATIONS_1K.get(orientation, ORIENTATIONS_1K['landscape'])
        scale = SIZES.get(size, 1.0)
        width, height = int(base_w * scale), int(base_h * scale)

        _current_status = {"generating": True, "prompt": prompt, "batch": batch, "current": 0}

        print(f"Generating: '{prompt}' ({batch}x, {steps} steps, {size} {orientation} {width}x{height})")

        start_time = time.perf_counter()

        images_data = []
        for i in range(batch):
            _current_status["current"] = i + 1
            print(f"  Image {i+1}/{batch}...")

            # If seed specified, use seed+i for each image in batch (so they're different)
            current_seed = (seed + i) if seed is not None else None

            # Generate the image
            image, used_seed, timings = generate_image(prompt, seed=current_seed, steps=steps, width=width, height=height, local_encoder=_local_encoder)

            # Save to web-generated folder
            t_save = time.perf_counter()
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            unique_id = uuid.uuid4().hex[:8]
            output_filename = f"flux2_{timestamp}_{unique_id}.png"
            output_path = os.path.join(OUTPUT_DIR, output_filename)
            image.save(output_path)
            timings['save'] = time.perf_counter() - t_save

            # Save prompt file alongside image
            prompt_file = save_prompt_file(output_path, prompt, prompt, width, height, used_seed, steps, timings)

            print(f"  Saved: {output_path} (seed: {used_seed})")
            print(f"  Prompt: {prompt_file}")
            print(f"    Timings: encoding={timings['encoding']:.2f}s, diffusion={timings['diffusion']:.2f}s, save={timings['save']:.2f}s")

            images_data.append({
                'filename': output_filename,
                'seed': used_seed,
                'timings': {
                    'encoding': round(timings['encoding'], 2),
                    'diffusion': round(timings['diffusion'], 2),
                    'save': round(timings['save'], 2),
                    'total': round(timings['encoding'] + timings['diffusion'] + timings['save'], 2)
                }
            })

        generation_time = time.perf_counter() - start_time

        _current_status = {"generating": False, "prompt": None}

        return jsonify({
            'success': True,
            'images': images_data,
            'generation_time': generation_time
        })

    except Exception as e:
        _current_status = {"generating": False, "prompt": None}
        print(f"Error: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

    finally:
        _generation_lock.release()


@app.route('/images/<filename>')
def serve_image(filename):
    return send_from_directory(OUTPUT_DIR, filename)


@app.route('/status')
def status():
    return jsonify(_current_status)


@app.route('/model-info')
def model_info():
    model_type = "FLUX.2-dev (full)" if _full_model else "FLUX.2-dev-bnb-4bit"
    encoder_type = "local encoder" if _local_encoder else "remote encoder"
    return jsonify({
        'model': model_type,
        'encoder': encoder_type,
        'description': f"{model_type} with {encoder_type}"
    })


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="FLUX.2 Web Server")
    parser.add_argument("--local-encoder", action="store_true", help="Use local text encoder instead of remote API (requires more VRAM)")
    parser.add_argument("--full-model", action="store_true", help="Use full FLUX.2-dev model instead of 4-bit quantized (requires more VRAM)")
    parser.add_argument("--port", type=int, default=PORT, help=f"Port to run server on (default: {PORT})")
    args = parser.parse_args()

    _full_model = args.full_model
    # Full model always uses local encoder
    _local_encoder = args.local_encoder or args.full_model

    model_mode = "full model" if _full_model else "4-bit quantized"
    encoder_mode = "local encoder" if _local_encoder else "remote encoder"

    print(f"Loading FLUX.2 ({model_mode}, {encoder_mode})...")
    load_model(local_encoder=_local_encoder, full_model=_full_model)
    print(f"\nStarting web server on http://0.0.0.0:{args.port}")
    print(f"Access from other devices: http://<your-ip>:{args.port}")
    app.run(host='0.0.0.0', port=args.port, threaded=True)
