import os
import argparse
import threading
import time
import base64
import io
from datetime import datetime
import uuid
from flask import Flask, request, jsonify, send_from_directory
from PIL import Image

# Import model components from fl24bit
import fl24bit
from fl24bit import load_model, generate_image, device, save_prompt_file, load_turbo_lora, load_uncensored_lora

app = Flask(__name__)

# Will be set by command-line args
_local_encoder = False
_full_model = False
_gguf_quant = None
_flux2 = False
_schnell = False
_turbo = False
_uncensored = False

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
}

HTML_PAGE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>FLUX.1 Image Generator</title>
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
        input[type="text"], select, textarea {
            width: 100%;
            padding: 12px;
            border: 1px solid #333;
            border-radius: 6px;
            background: #16213e;
            color: #fff;
            font-size: 16px;
            font-family: inherit;
        }
        textarea {
            resize: vertical;
            min-height: 60px;
        }
        input[type="text"]:focus, select:focus, textarea:focus {
            outline: none;
            border-color: #00d4ff;
        }
        .row { display: flex; gap: 15px; }
        .row .form-group { flex: 1; }
        .button-row {
            display: flex;
            gap: 10px;
        }
        button {
            flex: 1;
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
        .reset-btn {
            flex: 0 0 auto;
            width: auto;
            padding: 15px 25px;
            background: #6c757d;
            color: #fff;
        }
        .reset-btn:hover:not(:disabled) { background: #5a6268; }
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
        .image-input-container {
            display: flex;
            gap: 20px;
            align-items: flex-start;
        }
        .image-upload-area {
            flex: 0 0 200px;
            height: 150px;
            border: 2px dashed #333;
            border-radius: 8px;
            cursor: pointer;
            position: relative;
            overflow: hidden;
            transition: border-color 0.2s;
        }
        .image-upload-area:hover, .image-upload-area.dragover {
            border-color: #00d4ff;
        }
        .upload-placeholder {
            display: flex;
            align-items: center;
            justify-content: center;
            height: 100%;
            color: #666;
            text-align: center;
            padding: 10px;
        }
        .image-preview {
            width: 100%;
            height: 100%;
            position: relative;
        }
        .image-preview img {
            width: 100%;
            height: 100%;
            object-fit: contain;
            background: #0a0a15;
        }
        .clear-btn {
            position: absolute;
            top: 5px;
            right: 5px;
            width: 24px;
            height: 24px;
            padding: 0;
            background: rgba(255, 0, 0, 0.8);
            color: #fff;
            border: none;
            border-radius: 4px;
            cursor: pointer;
            font-size: 14px;
            font-weight: bold;
            line-height: 1;
        }
        .clear-btn:hover {
            background: rgba(255, 0, 0, 1);
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
        .history-section {
            margin-top: 40px;
            padding-top: 30px;
            border-top: 1px solid #333;
        }
        .history-section h2 {
            color: #888;
            font-size: 18px;
            margin-bottom: 20px;
        }
        .history-grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(150px, 1fr));
            gap: 12px;
        }
        .history-item {
            position: relative;
            aspect-ratio: 1;
            border-radius: 6px;
            overflow: hidden;
            cursor: pointer;
            transition: transform 0.2s;
        }
        .history-item:hover {
            transform: scale(1.05);
        }
        .history-item img {
            width: 100%;
            height: 100%;
            object-fit: cover;
        }
        .history-item .overlay {
            position: absolute;
            bottom: 0;
            left: 0;
            right: 0;
            background: linear-gradient(transparent, rgba(0,0,0,0.8));
            padding: 8px 6px 6px;
            opacity: 0;
            transition: opacity 0.2s;
        }
        .history-item:hover .overlay {
            opacity: 1;
        }
        .history-item .time {
            color: #fff;
            font-size: 11px;
        }
        .history-empty {
            color: #666;
            text-align: center;
            padding: 20px;
        }
    </style>
</head>
<body>
    <h1>FLUX.1 Image Generator</h1>
    <p class="subtitle" id="modelInfo">Loading model info...</p>

    <form id="generateForm">
        <div class="form-group">
            <label for="prompt">Prompt</label>
            <textarea id="prompt" name="prompt" rows="3" placeholder="A majestic mountain landscape at sunset..." required></textarea>
        </div>

        <div class="form-group">
            <label>Reference Image (optional - guides generation)</label>
            <div class="image-input-container">
                <div class="image-upload-area" id="uploadArea">
                    <input type="file" id="inputImage" accept="image/*" style="display: none;">
                    <div class="upload-placeholder" id="uploadPlaceholder">
                        <span>Click or drag image here</span>
                    </div>
                    <div class="image-preview" id="imagePreview" style="display: none;">
                        <img id="previewImg" src="">
                        <button type="button" class="clear-btn" id="clearImage">X</button>
                    </div>
                </div>
            </div>
        </div>

        <div class="row">
            <div class="form-group">
                <label for="orientation">Orientation</label>
                <select id="orientation" name="orientation">
                    <option value="square" selected>Square</option>
                    <option value="landscape">Landscape</option>
                    <option value="portrait">Portrait</option>
                </select>
            </div>
            <div class="form-group">
                <label for="size">Size</label>
                <select id="size" name="size">
                    <option value="1mp" selected>1 MP</option>
                    <option value="2mp">2 MP</option>
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
                <label for="guidance">Guidance Scale</label>
                <select id="guidance" name="guidance">
                    <option value="">Auto</option>
                    <option value="1">1 (high variety)</option>
                    <option value="2">2</option>
                    <option value="3">3</option>
                    <option value="4">4 (default)</option>
                    <option value="5">5</option>
                    <option value="6">6</option>
                    <option value="7">7 (strict)</option>
                </select>
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

        <div class="button-row">
            <button type="submit" id="submitBtn">Generate Image</button>
            <button type="button" class="reset-btn" id="resetBtn">Reset</button>
        </div>
    </form>

    <div class="status" id="status">
        <span class="spinner"></span>
        <span id="statusText">Generating...</span>
    </div>

    <div class="result" id="result">
        <div class="image-grid" id="imageGrid"></div>
        <p class="generation-info" id="generationInfo"></p>
    </div>

    <div class="history-section" id="historySection">
        <h2>Today's Generations</h2>
        <div class="history-grid" id="historyGrid"></div>
    </div>

    <script>
        const form = document.getElementById('generateForm');
        const submitBtn = document.getElementById('submitBtn');
        const status = document.getElementById('status');
        const statusText = document.getElementById('statusText');
        const result = document.getElementById('result');
        const imageGrid = document.getElementById('imageGrid');
        const generationInfo = document.getElementById('generationInfo');

        // Image upload elements
        const uploadArea = document.getElementById('uploadArea');
        const inputImage = document.getElementById('inputImage');
        const uploadPlaceholder = document.getElementById('uploadPlaceholder');
        const imagePreview = document.getElementById('imagePreview');
        const previewImg = document.getElementById('previewImg');
        const clearImage = document.getElementById('clearImage');

        let currentInputImage = null;

        function useSeed(seed) {
            document.getElementById('seed').value = seed;
        }

        // Image upload handling
        uploadArea.addEventListener('click', () => inputImage.click());

        uploadArea.addEventListener('dragover', (e) => {
            e.preventDefault();
            uploadArea.classList.add('dragover');
        });

        uploadArea.addEventListener('dragleave', () => {
            uploadArea.classList.remove('dragover');
        });

        uploadArea.addEventListener('drop', (e) => {
            e.preventDefault();
            uploadArea.classList.remove('dragover');
            const file = e.dataTransfer.files[0];
            if (file && file.type.startsWith('image/')) {
                handleImageFile(file);
            }
        });

        inputImage.addEventListener('change', (e) => {
            const file = e.target.files[0];
            if (file) {
                handleImageFile(file);
            }
        });

        function handleImageFile(file) {
            const reader = new FileReader();
            reader.onload = (e) => {
                currentInputImage = e.target.result;
                previewImg.src = currentInputImage;
                uploadPlaceholder.style.display = 'none';
                imagePreview.style.display = 'block';
            };
            reader.readAsDataURL(file);
        }

        clearImage.addEventListener('click', (e) => {
            e.stopPropagation();
            currentInputImage = null;
            previewImg.src = '';
            inputImage.value = '';
            uploadPlaceholder.style.display = 'flex';
            imagePreview.style.display = 'none';
        });

        // Reset button - clears all fields to defaults
        document.getElementById('resetBtn').addEventListener('click', () => {
            document.getElementById('prompt').value = '';
            document.getElementById('orientation').value = 'square';
            document.getElementById('size').value = '1mp';
            document.getElementById('steps').value = '25';
            document.getElementById('seed').value = '';
            document.getElementById('guidance').value = '';
            document.getElementById('batch').value = '1';
            // Clear input image
            currentInputImage = null;
            previewImg.src = '';
            inputImage.value = '';
            uploadPlaceholder.style.display = 'flex';
            imagePreview.style.display = 'none';
        });

        // Cmd+Return (Mac) or Ctrl+Return (Windows/Linux) to submit form
        document.addEventListener('keydown', (e) => {
            if ((e.metaKey || e.ctrlKey) && e.key === 'Enter') {
                e.preventDefault();
                if (!submitBtn.disabled) {
                    form.dispatchEvent(new Event('submit', { cancelable: true }));
                }
            }
        });

        // Fetch and display model info on page load
        fetch('/model-info')
            .then(r => r.json())
            .then(data => {
                document.getElementById('modelInfo').textContent = data.description;
                // Disable steps and guidance for schnell mode (fixed at 4 steps, guidance=0)
                if (data.schnell) {
                    const stepsSelect = document.getElementById('steps');
                    const guidanceSelect = document.getElementById('guidance');
                    stepsSelect.disabled = true;
                    stepsSelect.title = 'Schnell uses fixed 4 steps (8 for img2img)';
                    guidanceSelect.disabled = true;
                    guidanceSelect.title = 'Schnell requires guidance_scale=0';
                }
            })
            .catch(() => {
                document.getElementById('modelInfo').textContent = 'FLUX.1 Image Generator';
            });

        form.addEventListener('submit', async (e) => {
            e.preventDefault();

            const seedValue = document.getElementById('seed').value.trim();
            const guidanceValue = document.getElementById('guidance').value;
            const batch = parseInt(document.getElementById('batch').value);

            const formData = {
                prompt: document.getElementById('prompt').value,
                orientation: document.getElementById('orientation').value,
                size: document.getElementById('size').value,
                steps: parseInt(document.getElementById('steps').value),
                seed: seedValue ? parseInt(seedValue) : null,
                guidance: guidanceValue ? parseFloat(guidanceValue) : null,
                batch: batch
            };

            // Add reference image if present
            if (currentInputImage) {
                formData.input_image = currentInputImage;
            }

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
                    // Refresh history after successful generation
                    loadHistory();
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

        // History section
        const historyGrid = document.getElementById('historyGrid');

        async function loadHistory() {
            try {
                const response = await fetch('/history');
                const data = await response.json();

                historyGrid.innerHTML = '';

                if (data.images.length === 0) {
                    historyGrid.innerHTML = '<p class="history-empty">No images generated today</p>';
                    return;
                }

                data.images.forEach(img => {
                    const item = document.createElement('div');
                    item.className = 'history-item';
                    item.innerHTML = `
                        <img src="/images/${img.filename}" alt="${img.prompt || 'Generated image'}" loading="lazy">
                        <div class="overlay">
                            <span class="time">${img.time}</span>
                        </div>
                    `;
                    item.title = img.prompt || img.filename;
                    item.addEventListener('click', () => {
                        window.open(`/images/${img.filename}`, '_blank');
                    });
                    historyGrid.appendChild(item);
                });
            } catch (err) {
                console.error('Failed to load history:', err);
                historyGrid.innerHTML = '<p class="history-empty">Failed to load history</p>';
            }
        }

        // Load history on page load
        loadHistory();
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
        guidance_scale = data.get('guidance')  # None if not provided (uses default)
        batch = min(max(int(data.get('batch', 1)), 1), 4)  # Clamp to 1-4

        # Handle optional input image for img2img
        input_image = None
        strength = float(data.get('strength', 0.75))
        input_image_b64 = data.get('input_image')
        if input_image_b64:
            # Decode base64 image
            # Handle data URL format (e.g., "data:image/png;base64,...")
            if ',' in input_image_b64:
                input_image_b64 = input_image_b64.split(',', 1)[1]
            image_data = base64.b64decode(input_image_b64)
            input_image = Image.open(io.BytesIO(image_data)).convert('RGB')
            print(f"Received input image: {input_image.size}, strength={strength}")

        # Calculate dimensions
        scale = SIZES.get(size, 1.0)
        if input_image is not None:
            # For img2img: use input image's aspect ratio, scale to target resolution
            # Base target is ~1MP, scale multiplies each dimension
            in_w, in_h = input_image.size
            target_pixels = 1_000_000 * (scale ** 2)
            current_pixels = in_w * in_h
            factor = (target_pixels / current_pixels) ** 0.5
            width = int(round(in_w * factor / 8) * 8)  # Round to multiple of 8
            height = int(round(in_h * factor / 8) * 8)
            print(f"Img2img: scaling {in_w}x{in_h} -> {width}x{height} (preserving aspect ratio)")
        else:
            # For txt2img: use orientation preset
            base_w, base_h = ORIENTATIONS_1K.get(orientation, ORIENTATIONS_1K['landscape'])
            width, height = int(base_w * scale), int(base_h * scale)

        _current_status = {"generating": True, "prompt": prompt, "batch": batch, "current": 0}

        img2img_str = f", img2img strength={strength}" if input_image else ""
        guidance_str = f", guidance={guidance_scale}" if guidance_scale else ""
        orientation_str = "" if input_image else f" {orientation}"
        print(f"Generating: '{prompt}' ({batch}x, {steps} steps{guidance_str}, {size}{orientation_str} {width}x{height}{img2img_str})")

        start_time = time.perf_counter()

        images_data = []
        for i in range(batch):
            _current_status["current"] = i + 1
            print(f"  Image {i+1}/{batch}...")

            # If seed specified, use seed+i for each image in batch (so they're different)
            current_seed = (seed + i) if seed is not None else None

            # Generate the image
            image, used_seed, timings = generate_image(
                prompt, seed=current_seed, steps=steps, width=width, height=height,
                local_encoder=_local_encoder, input_image=input_image, strength=strength,
                guidance_scale=guidance_scale
            )

            # Save to web-generated folder
            t_save = time.perf_counter()
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            unique_id = uuid.uuid4().hex[:8]
            output_filename = f"flux{fl24bit._flux_version}_{timestamp}_{unique_id}.png"
            output_path = os.path.join(OUTPUT_DIR, output_filename)
            image.save(output_path)
            timings['save'] = time.perf_counter() - t_save

            # Save prompt file alongside image
            prompt_file = save_prompt_file(output_path, prompt, prompt, width, height, used_seed, steps, timings, guidance_scale)

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
    flux_name = f"FLUX.{fl24bit._flux_version}"
    if _schnell:
        model_type = f"{flux_name}-schnell (4-step)"
    elif _gguf_quant:
        model_type = f"{flux_name}-dev GGUF {_gguf_quant.upper()}"
    elif _full_model:
        model_type = f"{flux_name}-dev (full)"
    else:
        model_type = f"{flux_name}-dev-bnb-4bit"
    encoder_type = "local encoder" if _local_encoder else "remote encoder"
    turbo_str = " + Turbo" if fl24bit._turbo_enabled else ""
    uncensored_str = " + Uncensored" if fl24bit._uncensored_enabled else ""
    return jsonify({
        'model': model_type,
        'encoder': encoder_type,
        'turbo': fl24bit._turbo_enabled,
        'schnell': _schnell,
        'uncensored': fl24bit._uncensored_enabled,
        'description': f"{model_type}{turbo_str}{uncensored_str} with {encoder_type}"
    })


@app.route('/history')
def history():
    """Return today's generated images, newest first."""
    today = datetime.now().strftime("%Y%m%d")
    images = []

    try:
        for filename in os.listdir(OUTPUT_DIR):
            if not filename.endswith('.png'):
                continue
            # Filename format: flux{version}_{YYYYMMDD}_{HHMMSS}_{uuid}.png
            parts = filename.split('_')
            if len(parts) >= 3 and parts[1] == today:
                # Extract time from filename
                time_str = parts[2]
                if len(time_str) == 6:
                    display_time = f"{time_str[:2]}:{time_str[2:4]}:{time_str[4:6]}"
                else:
                    display_time = time_str

                # Try to read prompt from .prompt file
                prompt = None
                prompt_file = os.path.join(OUTPUT_DIR, filename.rsplit('.', 1)[0] + '.prompt')
                if os.path.exists(prompt_file):
                    try:
                        with open(prompt_file, 'r') as f:
                            for line in f:
                                if line.startswith('# Prompt: '):
                                    prompt = line[10:].strip()
                                    break
                    except Exception:
                        pass

                images.append({
                    'filename': filename,
                    'time': display_time,
                    'prompt': prompt,
                    'sort_key': parts[2] if len(parts) >= 3 else '000000'
                })

        # Sort by time, newest first
        images.sort(key=lambda x: x['sort_key'], reverse=True)

        # Remove sort_key from response
        for img in images:
            del img['sort_key']

    except Exception as e:
        print(f"Error reading history: {e}")

    return jsonify({'images': images})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="FLUX Web Server")
    parser.add_argument("--local-encoder", action="store_true", help="Use local text encoder instead of remote API (requires more VRAM)")
    parser.add_argument("--full-model", action="store_true", help="Use full FLUX model instead of 4-bit quantized (requires more VRAM)")
    parser.add_argument("--gguf", type=str, choices=["bf16", "q8", "q4"], default=None,
                        help="Use GGUF model (FLUX.1 only, recommended for DGX Spark). Options: bf16 (full quality), q8 (8-bit), q4 (4-bit smallest)")
    parser.add_argument("--flux2", action="store_true", help="Use FLUX.2 model instead of FLUX.1 (requires more VRAM)")
    parser.add_argument("--schnell", action="store_true", help="Use FLUX.1-schnell (fast 4-step model, Apache 2.0 license)")
    parser.add_argument("--turbo", action="store_true", default=None, help="Enable turbo LoRA for faster 8-step inference (FLUX.2 only, default: on for FLUX.2)")
    parser.add_argument("--no-turbo", action="store_true", help="Disable turbo LoRA (use standard inference)")
    parser.add_argument("--uncensored", action="store_true", help="Load Flux-Uncensored-V2 LoRA (FLUX.1 only)")
    parser.add_argument("--port", type=int, default=PORT, help=f"Port to run server on (default: {PORT})")
    args = parser.parse_args()

    _full_model = args.full_model
    _gguf_quant = args.gguf
    _flux2 = args.flux2
    _schnell = args.schnell
    _uncensored = args.uncensored
    # Full model and schnell always use local encoder; uncensored needs full model
    _local_encoder = args.local_encoder or args.full_model or args.schnell or args.uncensored
    if args.uncensored and not args.full_model:
        _full_model = True  # Uncensored LoRA requires full model
    # Turbo defaults to on for FLUX.2, can be disabled with --no-turbo
    _turbo = (args.turbo or args.flux2) and not args.no_turbo

    flux_name = "FLUX.2" if _flux2 else "FLUX.1"
    if _schnell:
        model_mode = "schnell (4-step)"
    elif _gguf_quant:
        model_mode = f"GGUF {_gguf_quant.upper()}"
    elif _full_model:
        model_mode = "full model"
    else:
        model_mode = "4-bit BNB"
    encoder_mode = "local encoder" if _local_encoder else "remote encoder"
    turbo_mode = " + Turbo LoRA" if _turbo else ""
    uncensored_mode = " + Uncensored LoRA" if _uncensored else ""

    print(f"Loading {flux_name} ({model_mode}, {encoder_mode}{turbo_mode}{uncensored_mode})...")
    load_model(local_encoder=_local_encoder, full_model=_full_model, gguf_quant=_gguf_quant, flux2=_flux2, schnell=_schnell, for_lora=_uncensored)

    if _turbo:
        load_turbo_lora()

    if _uncensored:
        load_uncensored_lora()

    print(f"\nStarting web server on http://0.0.0.0:{args.port}")
    print(f"Access from other devices: http://<your-ip>:{args.port}")
    app.run(host='0.0.0.0', port=args.port, threaded=True)
