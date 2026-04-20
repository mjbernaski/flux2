import os
import argparse
import json
import threading
import time
import base64
import io
import socket
import shutil
from datetime import datetime
import random
import uuid
from flask import Flask, request, jsonify, send_from_directory, Response, stream_with_context
from PIL import Image, ImageDraw, ImageFont
from dotenv import load_dotenv

# Load .env file if it exists
load_dotenv()

# Version number - update this when releasing new versions
VERSION = "1.1.1"

# Import model components from fl24bit
import fl24bit
from fl24bit import load_model, generate_image, device, save_prompt_file, load_turbo_lora, load_uncensored_lora

app = Flask(__name__)

# Security: Load API key from environment
API_KEY = os.environ.get("FLUX_API_KEY")
if API_KEY:
    API_KEY = API_KEY.strip()

if not API_KEY:
    print("\n" + "="*60)
    print("CRITICAL ERROR: FLUX_API_KEY environment variable is not set.")
    print("="*60)
    print("For security, this server now requires an API key to be set.")
    print("\nPlease set it before running the server:")
    print("  export FLUX_API_KEY=your_secret_key")
    print("\nOr create a .env file with:")
    print("  FLUX_API_KEY=your_secret_key")
    print("="*60 + "\n")
    import sys
    sys.exit(1)

def check_auth():
    # Check header or query param
    provided_key = request.headers.get("X-API-Key") or request.args.get("api_key")
    return provided_key == API_KEY

@app.before_request
def require_auth():
    # Allow the main page and images to load without authentication
    # Images are served with random filenames which provides basic security
    if request.endpoint in ['index', 'static', 'serve_image']:
        return
        
    if not check_auth():
        return jsonify({"success": False, "error": "Unauthorized. Please provide a valid X-API-Key header or api_key parameter."}), 401

# Will be set by command-line args
_local_encoder = False
_full_model = False
_gguf_quant = None
_flux2 = False
_schnell = False
_turbo = False
_uncensored = False
_klein = False

# Configuration
OUTPUT_DIR = "web-generated"
PORT = 2222

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Global state for background generation
_generation_lock = threading.Lock()
_current_status = {
    "generating": False,
    "prompt": None,
    "current": 0,
    "batch": 0,
    "step": 0,
    "total_steps": 0,
    "images": [],
    "composite": None,
    "done": False,
    "error": None,
    "generation_time": 0,
    "preview": None,
    "preview_step": 0,
    "preview_ts": 0
}

PREVIEW_FILENAME = "_preview_current.png"
PREVIEW_MIN_INTERVAL_S = 0.75  # throttle: skip decode if last preview was this recent

# Orientation presets (width, height) at 1K base
ORIENTATIONS_1K = {
    'square': (1024, 1024),
    'portrait': (768, 1344),
    'landscape': (1344, 768),
    'widescreen': (1568, 672),  # ~21:9 extra-wide, ~1 MP
}

SIZES = {
    '0.75mp': 0.75,
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
        .header-info { display: flex; justify-content: space-between; align-items: center; margin-bottom: 5px; }
        .hostname { color: #666; font-size: 12px; text-transform: uppercase; letter-spacing: 1px; }
        .version { color: #666; font-size: 12px; }
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
        .strength-control {
            flex: 1;
            display: flex;
            flex-direction: column;
            gap: 8px;
        }
        .strength-control label {
            color: #888;
            font-size: 14px;
        }
        .strength-control input[type="range"] {
            width: 100%;
            accent-color: #00d4ff;
        }
        .strength-hint {
            color: #666;
            font-size: 12px;
        }
        .spectrum-option .checkbox-label { display: flex; align-items: center; gap: 8px; cursor: pointer; }
        .spectrum-option input[type="checkbox"] { width: auto; }
        .spectrum-hint { color: #666; font-size: 12px; margin-top: 4px; }
        .spectrum-grid-selector {
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 4px;
            margin-top: 10px;
            max-width: 200px;
        }
        .grid-cell {
            aspect-ratio: 1;
            background: #222;
            border: 1px solid #444;
            cursor: pointer;
            border-radius: 2px;
            transition: all 0.2s;
        }
        .grid-cell:hover {
            border-color: #00d4ff;
        }
        .grid-cell.selected {
            background: #00d4ff;
            box-shadow: 0 0 8px rgba(0, 212, 255, 0.5);
        }
        .grid-labels {
            display: flex;
            justify-content: space-between;
            font-size: 10px;
            color: #666;
            margin-top: 4px;
            max-width: 200px;
        }
        .composite-card { grid-column: 1 / -1; }
        .composite-label { font-size: 12px; color: #00d4ff; margin-bottom: 8px; }
        .composite-img { max-width: 100%; height: auto; }
        .progress-tracker { margin-top: 10px; }
        .progress-bar-wrap { height: 8px; background: #333; border-radius: 4px; overflow: hidden; margin-bottom: 6px; }
        .progress-bar { height: 100%; background: #00d4ff; border-radius: 4px; transition: width 0.2s; }
        .progress-text { font-size: 13px; color: #888; }
        #strengthValue {
            color: #00d4ff;
            font-weight: bold;
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
        .history-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 20px;
        }
        .history-section h2 {
            color: #888;
            font-size: 18px;
            margin: 0;
        }
        .history-actions { display: flex; gap: 8px; }
        .archive-btn {
            flex: 0 0 auto;
            padding: 8px 16px;
            background: #e67e22;
            color: #fff;
            font-size: 13px;
            font-weight: bold;
            border-radius: 6px;
        }
        .archive-btn:hover:not(:disabled) { background: #d35400; }
        .delete-btn {
            flex: 0 0 auto;
            padding: 8px 16px;
            background: #c0392b;
            color: #fff;
            font-size: 13px;
            font-weight: bold;
            border-radius: 6px;
        }
        .delete-btn:hover:not(:disabled) { background: #992d22; }
        .history-item .item-delete {
            position: absolute;
            top: 4px;
            right: 4px;
            width: 22px;
            height: 22px;
            padding: 0;
            background: rgba(192, 57, 43, 0.85);
            color: #fff;
            border: none;
            border-radius: 4px;
            font-size: 13px;
            font-weight: bold;
            line-height: 1;
            cursor: pointer;
            opacity: 0;
            transition: opacity 0.2s;
            z-index: 2;
        }
        .history-item:hover .item-delete { opacity: 1; }
        .history-item .item-delete:hover { background: rgba(153, 45, 34, 1); }
        .history-grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(150px, 1fr));
            gap: 12px;
        }
        .history-item {
            position: relative;
            background: #0a0a15;
            border-radius: 6px;
            overflow: hidden;
            cursor: pointer;
            transition: transform 0.2s;
            display: flex;
            align-items: center;
            justify-content: center;
            min-height: 150px;
        }
        .history-item:hover {
            transform: scale(1.05);
        }
        .history-item img {
            max-width: 100%;
            max-height: 200px;
            width: auto;
            height: auto;
            object-fit: contain;
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

        /* Security CSS */
        .security-section {
            margin-bottom: 20px;
            padding: 15px;
            background: #2d2d44;
            border-radius: 8px;
            border: 1px solid #3d3d5c;
        }
        .security-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            cursor: pointer;
        }
        .security-title {
            color: #00d4ff;
            font-size: 14px;
            font-weight: bold;
            display: flex;
            align-items: center;
            gap: 8px;
        }
        #apiKeyInput {
            margin-top: 10px;
            padding: 8px;
            font-size: 14px;
            background: #16213e;
            border: 1px solid #444;
            border-radius: 4px;
            color: #fff;
            width: 100%;
        }
    </style>
</head>
<body>
    <div class="header-info">
        <span class="hostname" id="hostname"></span>
        <span class="version" id="version"></span>
    </div>
    <h1>FLUX Image Generator</h1>
    <p class="subtitle" id="modelInfo">Loading model info...</p>

    <!-- Security UI -->
    <div class="security-section">
        <div class="security-header" onclick="document.getElementById('securityContent').style.display = document.getElementById('securityContent').style.display === 'none' ? 'block' : 'none'">
            <div class="security-title">
                <span>🔒 Security Settings</span>
            </div>
            <span style="font-size: 12px; color: #888;">Click to toggle</span>
        </div>
        <div id="securityContent" style="display: none; padding-top: 10px;">
            <label for="apiKey">API Access Key:</label>
            <input type="password" id="apiKeyInput" placeholder="Enter API Key (REQUIRED)..." oninput="localStorage.setItem('flux_api_key', this.value)">
            <p style="font-size: 11px; color: #666; margin-top: 5px;">Stored in browser local storage for convenience. Always required to use this server.</p>
            <button type="button" id="resetLockBtn" style="margin-top: 10px; background: #944; border: none; padding: 4px 8px; font-size: 11px;">Emergency Reset Server Lock</button>
        </div>
    </div>

    <form id="generateForm" action="#">
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
                <div class="strength-control" id="strengthControl" style="display: none;">
                    <label for="strength">Reference following (strength): <span id="strengthValue">0.5</span></label>
                    <input type="range" id="strength" name="strength" min="0" max="1" step="0.1" value="0.5">
                    <div class="strength-hint">0 = closest to original, 0.5 = default, 1 = most change</div>
                </div>
                <div class="aspect-mode-control" id="aspectModeControl" style="display: none; margin-top: 10px;">
                    <label for="aspectMode">Image aspect ratio mode:</label>
                    <select id="aspectMode" name="aspectMode">
                        <option value="keep" selected>Keep original aspect ratio (auto-scale)</option>
                        <option value="stretch">Stretch/Squash to match Orientation selection</option>
                    </select>
                </div>
            </div>
        </div>

        <div class="row">
            <div class="form-group">
                <label for="orientation">Orientation</label>
                <select id="orientation" name="orientation">
                    <option value="square">Square</option>
                    <option value="landscape" selected>Landscape</option>
                    <option value="portrait">Portrait</option>
                    <option value="widescreen">Extra-wide (21:9)</option>
                </select>
            </div>
            <div class="form-group">
                <label for="size">Size</label>
                <select id="size" name="size">
                    <option value="0.75mp">0.75 MP</option>
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
                    <option value="1">1.0 (high variety)</option>
                    <option value="1.5">1.5</option>
                    <option value="2">2.0</option>
                    <option value="2.5">2.5</option>
                    <option value="3">3.0</option>
                    <option value="3.5">3.5</option>
                    <option value="4" selected>4.0 (default)</option>
                    <option value="4.5">4.5</option>
                    <option value="5">5.0</option>
                    <option value="5.5">5.5</option>
                    <option value="6">6.0</option>
                    <option value="6.5">6.5</option>
                    <option value="7">7.0 (strict)</option>
                </select>
            </div>
            <div class="form-group">
                <label for="batch">Batch Size</label>
                <select id="batch" name="batch">
                    <option value="1" selected>1 image</option>
                    <option value="2">2 images</option>
                    <option value="4">4 images</option>
                    <option value="8">8 images</option>
                    <option value="16">16 images</option>
                    <option value="32">32 images</option>
                    <option value="64">64 images</option>
                    <option value="128">128 images</option>
                </select>
            </div>
        </div>

        <div class="row">
            <div class="form-group spectrum-option">
                <label class="checkbox-label">
                    <input type="checkbox" id="spectrumGrid" name="spectrum_grid" value="1">
                    Generate spectrum grid (interactive 4×4 matrix)
                </label>
                <div id="gridContainer" style="display: none; margin-top: 10px;">
                    <div style="font-size: 11px; color: #888; margin-bottom: 4px;">Select combinations (Guidance →, Strength ↓)</div>
                    <div class="spectrum-grid-selector" id="gridSelector">
                        <!-- 16 cells added via JS -->
                    </div>
                    <div class="grid-labels">
                        <span>Min G/S</span>
                        <span>Max G/S</span>
                    </div>
                </div>
                <div class="spectrum-hint" id="spectrumHint">Guidance: 1, 3, 5, 7. Strength (img2img): 0.2, 0.4, 0.6, 0.8.</div>
                <label class="checkbox-label" style="margin-top: 6px;">
                    <input type="checkbox" id="spectrumSameSeed" checked>
                    Use same seed for all grid images
                </label>
            </div>
        </div>

        <div class="row">
            <div class="form-group">
                <label class="checkbox-label">
                    <input type="checkbox" id="showPreview">
                    Show live preview (slower — decodes each step)
                </label>
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
        <div class="progress-tracker" id="progressTracker" style="display: none;">
            <div class="progress-bar-wrap">
                <div class="progress-bar" id="progressBar" style="width: 0%;"></div>
            </div>
            <span class="progress-text" id="progressText">0 / 0</span>
        </div>
        <div class="preview-wrap" id="previewWrap" style="display: none; margin-top: 12px;">
            <img id="previewImgLive" alt="Live preview" style="max-width: 512px; width: 100%; border-radius: 6px; display: block;">
        </div>
    </div>

    <div class="result" id="result">
        <div class="image-grid" id="imageGrid"></div>
        <p class="generation-info" id="generationInfo"></p>
    </div>

    <div class="history-section" id="historySection">
        <div class="history-header">
            <h2>Today's Generations</h2>
            <div class="history-actions">
                <button type="button" class="archive-btn" id="archiveBtn" style="display: none;">Archive Today</button>
                <button type="button" class="delete-btn" id="deleteAllBtn" style="display: none;">Delete Today</button>
            </div>
        </div>
        <div class="history-grid" id="historyGrid"></div>
    </div>

    <script>
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
                    el.textContent = data.description || 'FLUX Image Generator';
                    var h = document.getElementById('hostname'); if (h) h.textContent = data.hostname || '';
                    var v = document.getElementById('version'); if (v) v.textContent = 'v' + (data.version || '');
                    if (data.schnell) {
                        var s = document.getElementById('steps'); if (s) { s.disabled = true; s.title = 'Schnell uses fixed 4 steps'; }
                        var g = document.getElementById('guidance'); if (g) { g.disabled = true; g.title = 'Schnell requires guidance_scale=0'; }
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
        const imagePreview = document.getElementById('imagePreview');
        const previewImg = document.getElementById('previewImg');
        const clearImage = document.getElementById('clearImage');

        let currentInputImage = null;
        const strengthControl = document.getElementById('strengthControl');
        const aspectModeControl = document.getElementById('aspectModeControl');
        const aspectModeEl = document.getElementById('aspectMode');
        const strengthSlider = document.getElementById('strength');
        const strengthValue = document.getElementById('strengthValue');
        
        let pollInterval = null;
        let knownImageFilenames = new Set();
        let lastPreviewStep = -1;

        if (strengthSlider) strengthSlider.addEventListener('input', function() { if (strengthValue) strengthValue.textContent = strengthSlider.value; });

        function useSeed(seed) {
            var el = document.getElementById('seed'); if (el) el.value = seed;
        }

        function handleImageFile(file) {
            var reader = new FileReader();
            reader.onload = function(e) {
                currentInputImage = e.target.result;
                if (previewImg) previewImg.src = currentInputImage;
                if (uploadPlaceholder) uploadPlaceholder.style.display = 'none';
                if (imagePreview) imagePreview.style.display = 'block';
                if (strengthControl) strengthControl.style.display = 'flex';
                if (aspectModeControl) aspectModeControl.style.display = 'block';
                };
                reader.readAsDataURL(file);        }
        if (uploadArea) {
            uploadArea.addEventListener('click', function() { if (inputImage) inputImage.click(); });
            uploadArea.addEventListener('dragover', function(e) { e.preventDefault(); uploadArea.classList.add('dragover'); });
            uploadArea.addEventListener('dragleave', function() { uploadArea.classList.remove('dragover'); });
            uploadArea.addEventListener('drop', function(e) {
                e.preventDefault();
                uploadArea.classList.remove('dragover');
                var file = e.dataTransfer.files[0];
                if (file && file.type.indexOf('image/') === 0) handleImageFile(file);
            });
        }
        if (inputImage) inputImage.addEventListener('change', function(e) { var f = e.target.files[0]; if (f) handleImageFile(f); });
        if (clearImage) clearImage.addEventListener('click', function(e) {
            e.stopPropagation();
            currentInputImage = null;
            if (previewImg) previewImg.src = '';
            if (inputImage) inputImage.value = '';
            if (uploadPlaceholder) uploadPlaceholder.style.display = 'flex';
            if (imagePreview) imagePreview.style.display = 'none';
            if (strengthControl) strengthControl.style.display = 'none';
            if (aspectModeControl) aspectModeControl.style.display = 'none';
        });
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

        if (resetBtn) resetBtn.addEventListener('click', function() {
            var p = document.getElementById('prompt'); if (p) p.value = '';
            var o = document.getElementById('orientation'); if (o) o.value = 'landscape';
            var s = document.getElementById('size'); if (s) s.value = '1mp';
            var st = document.getElementById('steps'); if (st) st.value = '25';
            var sd = document.getElementById('seed'); if (sd) sd.value = '';
            var gu = document.getElementById('guidance'); if (gu) gu.value = '4';
            var b = document.getElementById('batch'); if (b) b.value = '1';
            currentInputImage = null;
            if (previewImg) previewImg.src = '';
            if (inputImage) inputImage.value = '';
            if (uploadPlaceholder) uploadPlaceholder.style.display = 'flex';
            if (imagePreview) imagePreview.style.display = 'none';
            if (strengthSlider) strengthSlider.value = '0.5';
            if (strengthValue) strengthValue.textContent = '0.5';
            if (strengthControl) strengthControl.style.display = 'none';
            if (aspectModeControl) aspectModeControl.style.display = 'none';
            if (aspectModeEl) aspectModeEl.value = 'keep';
            var sg = document.getElementById('spectrumGrid'); if (sg) sg.checked = false;
            var sss = document.getElementById('spectrumSameSeed'); if (sss) sss.checked = true;
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
            const pt = document.getElementById('progressTracker'); if (pt) pt.style.display = 'none';
            const pb = document.getElementById('progressBar'); if (pb) pb.style.width = '0%';
        });

        async function pollStatus() {
            try {
                const response = await fetch('/status', { headers: getAuthHeaders() });
                
                if (response.status === 401) {
                    clearInterval(pollInterval);
                    pollInterval = null;
                    submitBtn.disabled = false;
                    status.className = 'status error';
                    statusText.textContent = 'Error: Unauthorized. Please check your API Key.';
                    return;
                }
                
                if (!response.ok) {
                    throw new Error(`HTTP error! status: ${response.status}`);
                }
                
                const data = await response.json();
                
                if (data.generating) {
                    submitBtn.disabled = true;
                    status.className = 'status generating';

                    // Build status text with step progress
                    let stepInfo = '';
                    if (data.total_steps > 0 && data.step > 0) {
                        stepInfo = ` (step ${data.step} of ${data.total_steps})`;
                    }
                    if (data.batch > 1) {
                        statusText.textContent = `Generating: ${data.current} / ${data.batch}${stepInfo}...`;
                    } else {
                        statusText.textContent = data.step > 0
                            ? `Generating: step ${data.step} of ${data.total_steps}...`
                            : 'Generating...';
                    }

                    const progressTracker = document.getElementById('progressTracker');
                    const progressBar = document.getElementById('progressBar');
                    const progressText = document.getElementById('progressText');

                    // Show progress bar based on steps (always) or batch progress
                    if (data.total_steps > 0) {
                        progressTracker.style.display = 'block';
                        const batch = Math.max(1, data.batch || 1);
                        const totalStepsAll = data.total_steps * batch;
                        const stepsDone = (data.current - 1) * data.total_steps + data.step;
                        if (batch > 1) {
                            progressBar.style.width = `${(stepsDone / totalStepsAll) * 100}%`;
                        } else {
                            progressBar.style.width = `${(data.step / data.total_steps) * 100}%`;
                        }

                        const pct = Math.min(100, Math.round((stepsDone / totalStepsAll) * 100));
                        progressText.textContent = `${pct}% complete`;
                    } else {
                        progressTracker.style.display = 'none';
                    }
                    
                    // Live preview (only when user enabled it; server emits preview_step > 0).
                    // Use preview_ts as the cache buster so new generations don't collide
                    // with cached step-N images from previous generations.
                    if (data.preview && data.preview_ts && data.preview_ts !== lastPreviewStep) {
                        const pwrap = document.getElementById('previewWrap');
                        const pimg = document.getElementById('previewImgLive');
                        if (pwrap && pimg) {
                            pimg.src = `/images/${data.preview}?t=${data.preview_ts}`;
                            pwrap.style.display = 'block';
                        }
                        lastPreviewStep = data.preview_ts;
                    }

                    // Show images as they arrive
                    if (data.images && data.images.length > 0) {
                        result.className = 'result visible';
                        data.images.forEach((img, i) => {
                            if (!knownImageFilenames.has(img.filename)) {
                                addImageToGrid(img, i + 1);
                                knownImageFilenames.add(img.filename);
                            }
                        });
                    }
                } else {
                    // Generation finished
                    clearInterval(pollInterval);
                    pollInterval = null;
                    submitBtn.disabled = false;

                    if (data.done) {
                        status.className = 'status';
                        result.className = 'result visible';
                        document.getElementById('progressTracker').style.display = 'none';
                        const pwrapDone = document.getElementById('previewWrap');
                        if (pwrapDone) pwrapDone.style.display = 'none';
                        
                        // Add any remaining images
                        if (data.images) {
                            data.images.forEach((img, i) => {
                                if (!knownImageFilenames.has(img.filename)) {
                                    addImageToGrid(img, i + 1);
                                    knownImageFilenames.add(img.filename);
                                }
                            });
                        }
                        
                        // Add composite if present
                        if (data.composite && !knownImageFilenames.has(data.composite)) {
                            addCompositeToGrid(data.composite);
                            knownImageFilenames.add(data.composite);
                        }
                        
                        generationInfo.textContent = data.composite
                            ? `Generated ${data.images.length} images + 1 composite in ${data.generation_time.toFixed(1)}s`
                            : `Generated ${data.images.length} image(s) in ${data.generation_time.toFixed(1)}s`;
                        
                        loadHistory();
                    } else if (data.error) {
                        status.className = 'status error';
                        statusText.textContent = 'Error: ' + data.error;
                        document.getElementById('progressTracker').style.display = 'none';
                        const pwrapErr = document.getElementById('previewWrap');
                        if (pwrapErr) pwrapErr.style.display = 'none';
                    } else {
                        status.className = 'status'; // Idle
                    }
                }
            } catch (err) {
                console.error('Polling error:', err);
                // Don't clear interval here, server might be briefly down or network issue
            }
        }

        function addImageToGrid(img, index) {
            const t = Date.now();
            const meta = img.guidance != null ? `Guidance: ${img.guidance}${img.strength != null ? ', Strength: ' + img.strength : ''}` : '';
            const card = document.createElement('div');
            card.className = 'image-card';
            card.innerHTML = `
                <img src="/images/${img.filename}?t=${t}" alt="Generated image ${index}" loading="lazy">
                <div class="actions">
                    <a href="/images/${img.filename}" download="${img.filename}">Download</a>
                    <a href="#" class="seed-btn" onclick="useSeed(${img.seed}); return false;">Use Seed</a>
                </div>
                <p class="info">${meta ? meta + ' · Seed: ' + img.seed : 'Seed: ' + img.seed}</p>
                <div class="timings">
                    <span class="timing-item"><span class="timing-label">Encode:</span> ${img.timings.encoding}s</span>
                    <span class="timing-item"><span class="timing-label">Diffuse:</span> ${img.timings.diffusion}s</span>
                    <span class="timing-item"><span class="timing-label">Save:</span> ${img.timings.save}s</span>
                    <span class="timing-item timing-total"><span class="timing-label">Total:</span> ${img.timings.total}s</span>
                </div>
            `;
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

            const formData = {
                prompt: promptEl ? promptEl.value : '',
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
            if (currentInputImage && strengthSlider) {
                formData.input_image = currentInputImage;
                formData.strength = parseFloat(strengthSlider.value);
                formData.aspect_mode = aspectModeEl ? aspectModeEl.value : 'keep';
            }

            knownImageFilenames.clear();
            imageGrid.innerHTML = '';
            result.className = 'result';
            submitBtn.disabled = true;
            status.className = 'status generating';
            statusText.textContent = 'Starting generation...';
            lastPreviewStep = -1;
            const previewWrapEl = document.getElementById('previewWrap');
            if (previewWrapEl) previewWrapEl.style.display = 'none';

            try {
                const response = await fetch('/generate', {
                    method: 'POST',
                    headers: getAuthHeaders({ 'Content-Type': 'application/json' }),
                    body: JSON.stringify(formData)
                });
                
                if (response.status === 503) {
                    statusText.textContent = 'Generation already in progress... joining session.';
                    if (pollInterval) clearInterval(pollInterval);
                    pollInterval = setInterval(pollStatus, 1500);
                    return;
                }
                
                const data = await response.json();
                
                if (data.success) {
                    if (pollInterval) clearInterval(pollInterval);
                    pollInterval = setInterval(pollStatus, 1500);
                } else {
                    status.className = 'status error';
                    statusText.textContent = 'Error: ' + (data.error || 'Unknown error');
                    submitBtn.disabled = false;
                }
            } catch (err) {
                status.className = 'status error';
                statusText.textContent = 'Error starting generation: ' + err.message;
                submitBtn.disabled = false;
            }
        }

        // Handle visibility change for mobile robustness
        document.addEventListener('visibilitychange', function() {
            if (document.visibilityState === 'visible') {
                // If we were supposed to be polling, or just to check current status
                pollStatus();
                if (!pollInterval && submitBtn.disabled) {
                    pollInterval = setInterval(pollStatus, 1500);
                }
            }
        });

        // Initial check if server is already generating
        pollStatus().then(() => {
            if (submitBtn.disabled && !pollInterval) {
                pollInterval = setInterval(pollStatus, 1500);
            }
        });

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
                archiveBtn.style.display = hasImages ? 'block' : 'none';
                if (deleteAllBtn) deleteAllBtn.style.display = hasImages ? 'block' : 'none';
                if (!hasImages) {
                    historyGrid.innerHTML = '<p class="history-empty">No images generated today</p>';
                    return;
                }
                data.images.forEach(img => {
                    const item = document.createElement('div');
                    item.className = 'history-item';
                    const safeName = img.filename.replace(/"/g, '&quot;');
                    item.innerHTML = `
                        <img src="/images/${img.filename}" alt="${img.prompt || 'Generated image'}" loading="lazy">
                        <button type="button" class="item-delete" data-filename="${safeName}" title="Delete">X</button>
                        <div class="overlay">
                            <span class="time">${img.time}</span>
                        </div>
                    `;
                    item.title = img.prompt || img.filename;
                    item.addEventListener('click', (e) => {
                        if (e.target.classList.contains('item-delete')) return;
                        window.open(`/images/${img.filename}`, '_blank');
                    });
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
        const resetLockBtn = document.getElementById('resetLockBtn');

        if (resetLockBtn) {
            resetLockBtn.addEventListener('click', async () => {
                if (!confirm("This will FORCE unlock the server. Only do this if you are certain no image is actually generating!")) return;
                try {
                    const response = await fetch('/reset-lock', { 
                        method: 'POST',
                        headers: getAuthHeaders()
                    });
                    const data = await response.json();
                    if (data.success) {
                        alert("Server lock released.");
                        pollStatus();
                    }
                } catch (err) { alert("Failed to reset lock: " + err.message); }
            });
        }

        archiveBtn.addEventListener('click', async () => {
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

        loadHistory();
    </script>
</body>
</html>
"""


def background_generation_task(data):
    """Background task for image generation, ensuring it continues if client disconnects."""
    global _current_status
    
    try:
        prompt = data.get('prompt', '').strip()
        orientation = data.get('orientation', 'landscape')
        size = data.get('size', '1mp')
        steps = int(data.get('steps', 25))
        seed = data.get('seed')
        guidance_scale = data.get('guidance')
        batch = min(max(int(data.get('batch', 1)), 1), 128)
        
        # Handle input image
        input_image = None
        strength = float(data.get('strength', 0.5))
        input_image_b64 = data.get('input_image')
        if input_image_b64:
            if ',' in input_image_b64:
                input_image_b64 = input_image_b64.split(',', 1)[1]
            image_data = base64.b64decode(input_image_b64)
            input_image = Image.open(io.BytesIO(image_data)).convert('RGB')

        # Dimensions
        scale = SIZES.get(size, 1.0)
        aspect_mode = data.get('aspect_mode', 'keep')

        if input_image is not None and aspect_mode == 'keep':
            in_w, in_h = input_image.size
            target_pixels = 1_000_000 * (scale ** 2)
            current_pixels = in_w * in_h
            factor = (target_pixels / current_pixels) ** 0.5
            width = int(round(in_w * factor / 8) * 8)
            height = int(round(in_h * factor / 8) * 8)
        else:
            base_w, base_h = ORIENTATIONS_1K.get(orientation, ORIENTATIONS_1K['landscape'])
            width, height = int(base_w * scale), int(base_h * scale)

        spectrum_grid = data.get('spectrum_grid', False)
        spectrum_same_seed = data.get('spectrum_same_seed', True)
        selected_cells = data.get('selected_cells', []) # Indices 0-15

        if spectrum_grid:
            guidance_values = [0] if _schnell else [1.0, 3.0, 5.0, 7.0]
            strength_values = [0.2, 0.4, 0.6, 0.8] if input_image else [0.0, 0.0, 0.0, 0.0] # Dummy if no image

            if selected_cells:
                total_batch = len(selected_cells)
            elif input_image and not _schnell:
                # Fallback to diagonals if nothing selected but somehow grid is on
                total_batch = 8
            else:
                total_batch = len(strength_values) * len(guidance_values)
        else:
            total_batch = batch

        _current_status["batch"] = total_batch
        _current_status["total_steps"] = steps

        show_preview = bool(data.get('show_preview', False))
        preview_state = {"last_decode": 0.0}

        def _step_callback(pipe_obj, step_index, timestep, callback_kwargs):
            _current_status["step"] = step_index + 1
            if show_preview:
                now = time.perf_counter()
                is_final = (step_index + 1) >= steps
                if is_final or (now - preview_state["last_decode"]) >= PREVIEW_MIN_INTERVAL_S:
                    latents = callback_kwargs.get("latents")
                    preview_img = fl24bit.decode_latents_to_preview(
                        pipe_obj, latents, height, width
                    )
                    if preview_img is not None:
                        try:
                            preview_img.save(os.path.join(OUTPUT_DIR, PREVIEW_FILENAME))
                            _current_status["preview"] = PREVIEW_FILENAME
                            _current_status["preview_step"] = step_index + 1
                            _current_status["preview_ts"] = int(time.time() * 1000)
                            preview_state["last_decode"] = now
                        except Exception as e:
                            print(f"[preview] save failed: {e}", flush=True)
            return callback_kwargs

        start_time = time.perf_counter()

        if spectrum_grid:
            grid_seed = seed if seed is not None else random.randint(0, 2**32 - 1)
            grid_cells = []

            # We still want to build a full 4x4 grid for the composite, but only generate selected
            guidance_values = [0] if _schnell else [1.0, 3.0, 5.0, 7.0]
            strength_values = [0.2, 0.4, 0.6, 0.8] if input_image else [0.0, 0.2, 0.4, 0.6] # Use some defaults if no image for grid

            generated_count = 0
            for r_idx, s_val in enumerate(strength_values):
                row_images = []
                for c_idx, g_val in enumerate(guidance_values):
                    cell_idx = r_idx * 4 + c_idx

                    # Check if this cell should be generated
                    should_gen = False
                    if selected_cells:
                        should_gen = cell_idx in selected_cells
                    elif input_image and not _schnell:
                        # Legacy diagonal logic
                        is_main_diag = (r_idx == c_idx)
                        is_anti_diag = (r_idx == len(guidance_values) - 1 - c_idx)
                        should_gen = is_main_diag or is_anti_diag
                    else:
                        should_gen = True

                    if not should_gen:
                        row_images.append(None)
                        continue

                    generated_count += 1
                    _current_status["current"] = generated_count
                    _current_status["step"] = 0
                    current_seed = grid_seed if spectrum_same_seed else random.randint(0, 2**32 - 1)

                    image, used_seed, timings = generate_image(
                        prompt, seed=current_seed, steps=steps, width=width, height=height,
                        local_encoder=_local_encoder, input_image=input_image,
                        strength=(s_val if input_image else 0.5),
                        guidance_scale=g_val,
                        callback_on_step_end=_step_callback
                    )                    
                    t_save = time.perf_counter()
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    unique_id = uuid.uuid4().hex[:8]
                    g_str = str(g_val).replace('.', '_')
                    s_str = f"str_{s_val}" if s_val is not None else "txt2img"
                    output_filename = f"flux{fl24bit._flux_version}_{timestamp}_g{g_str}_{s_str}_{unique_id}.png"
                    output_path = os.path.join(OUTPUT_DIR, output_filename)
                    image.save(output_path)
                    timings['save'] = time.perf_counter() - t_save
                    save_prompt_file(output_path, prompt, prompt, width, height, used_seed, steps, timings, g_val, s_val if input_image else None)
                    
                    img_data = {
                        "filename": output_filename,
                        "seed": used_seed,
                        "guidance": g_val,
                        "strength": s_val,
                        "timings": {
                            'encoding': round(timings['encoding'], 2),
                            'diffusion': round(timings['diffusion'], 2),
                            'save': round(timings['save'], 2),
                            'total': round(timings['encoding'] + timings['diffusion'] + timings['save'], 2)
                        }
                    }
                    _current_status["images"].append(img_data)
                    row_images.append((image.copy(), generated_count))
                grid_cells.append(row_images)

            # Create composite
            # Calculate cell size based on aspect ratio
            aspect_ratio = width / height
            if width >= height:
                cell_width = 256
                cell_height = int(round(cell_width / aspect_ratio))
            else:
                cell_height = 256
                cell_width = int(round(cell_height * aspect_ratio))

            n_rows, n_cols = len(grid_cells), len(grid_cells[0])
            composite = Image.new('RGB', (n_cols * cell_width, n_rows * cell_height), (32, 32, 32))
            for row_idx, row_images in enumerate(grid_cells):
                for col_idx, cell in enumerate(row_images):
                    if cell is None: continue # Skip empty diagonal cells
                    img, _seq = cell
                    img_small = img.resize((cell_width, cell_height), Image.Resampling.LANCZOS)
                    composite.paste(img_small, (col_idx * cell_width, row_idx * cell_height))

            # Overlay the generation-sequence number on each populated cell
            draw = ImageDraw.Draw(composite)
            font_size = max(14, cell_height // 12)
            font = None
            for font_path in (
                "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
                "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
            ):
                if os.path.exists(font_path):
                    try:
                        font = ImageFont.truetype(font_path, font_size)
                        break
                    except Exception:
                        font = None
            if font is None:
                font = ImageFont.load_default()
            for row_idx, row_images in enumerate(grid_cells):
                for col_idx, cell in enumerate(row_images):
                    if cell is None: continue
                    _img, seq = cell
                    label = str(seq)
                    pad = 4
                    bbox = draw.textbbox((0, 0), label, font=font)
                    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
                    x0 = col_idx * cell_width + 6
                    y0 = row_idx * cell_height + 6
                    draw.rectangle(
                        [x0 - pad, y0 - pad, x0 + tw + pad, y0 + th + pad],
                        fill=(0, 0, 0),
                    )
                    draw.text((x0 - bbox[0], y0 - bbox[1]), label, fill=(255, 255, 255), font=font)

            comp_filename = f"flux{fl24bit._flux_version}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_spectrum_grid.png"
            composite.save(os.path.join(OUTPUT_DIR, comp_filename))
            _current_status["composite"] = comp_filename
        else:
            for i in range(batch):
                _current_status["current"] = i + 1
                _current_status["step"] = 0
                current_seed = (seed + i) if seed is not None else None
                image, used_seed, timings = generate_image(
                    prompt, seed=current_seed, steps=steps, width=width, height=height,
                    local_encoder=_local_encoder, input_image=input_image, strength=strength,
                    guidance_scale=guidance_scale,
                    callback_on_step_end=_step_callback
                )
                
                t_save = time.perf_counter()
                output_filename = f"flux{fl24bit._flux_version}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}.png"
                output_path = os.path.join(OUTPUT_DIR, output_filename)
                image.save(output_path)
                timings['save'] = time.perf_counter() - t_save
                save_prompt_file(output_path, prompt, prompt, width, height, used_seed, steps, timings, guidance_scale, strength if input_image else None)

                img_data = {
                    'filename': output_filename,
                    'seed': used_seed,
                    'guidance': guidance_scale,
                    'strength': strength if input_image else None,
                    'timings': {
                        'encoding': round(timings['encoding'], 2),
                        'diffusion': round(timings['diffusion'], 2),
                        'save': round(timings['save'], 2),
                        'total': round(timings['encoding'] + timings['diffusion'] + timings['save'], 2)
                    }
                }
                _current_status["images"].append(img_data)

        _current_status["generation_time"] = time.perf_counter() - start_time
        _current_status["done"] = True
    except Exception as e:
        print(f"Background generation error: {e}")
        _current_status["error"] = str(e)
    finally:
        _current_status["generating"] = False
        if _generation_lock.locked():
            _generation_lock.release()


@app.route('/')
def index():
    return HTML_PAGE


@app.route('/generate', methods=['POST'])
def generate():
    if not _generation_lock.acquire(blocking=False):
        return jsonify({'success': False, 'error': 'Generation in progress'}), 503
    
    try:
        # Initialize status for new run BEFORE starting thread to avoid race conditions
        global _current_status
        data = request.json
        _current_status = {
            "generating": True,
            "prompt": data.get('prompt', ''),
            "current": 0,
            "batch": 0,
            "step": 0,
            "total_steps": 0,
            "images": [],
            "composite": None,
            "done": False,
            "error": None,
            "generation_time": 0,
            "preview": None,
            "preview_step": 0,
            "preview_ts": 0
        }
        
        # Logic is moved to thread, release is handled by thread finally
        threading.Thread(target=background_generation_task, args=(data,)).start()
        return jsonify({'success': True})
    except Exception as e:
        if _generation_lock.locked():
            _generation_lock.release()
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/images/<filename>')
def serve_image(filename):
    return send_from_directory(OUTPUT_DIR, filename)


@app.route('/status')
def status():
    return jsonify(_current_status)


@app.route('/reset-lock', methods=['POST'])
def reset_lock():
    if not check_auth():
        return jsonify({"success": False, "error": "Unauthorized"}), 401
        
    global _current_status
    if _generation_lock.locked():
        _generation_lock.release()
        
    _current_status["generating"] = False
    _current_status["done"] = False
    _current_status["error"] = "Lock reset by user."
    return jsonify({"success": True, "message": "Server lock has been manually released."})


@app.route('/model-info')
def model_info():
    flux_name = f"FLUX.{fl24bit._flux_version}"
    variant = "-klein" if _klein else "-dev"
    if _schnell:
        model_type = f"{flux_name}-schnell (4-step)"
    elif _gguf_quant:
        model_type = f"{flux_name}-dev GGUF {_gguf_quant.upper()}"
    elif _full_model:
        model_type = f"{flux_name}{variant} (full)"
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
        'hostname': socket.gethostname(),
        'version': VERSION,
        'description': f"{model_type}{turbo_str}{uncensored_str} with {encoder_type}"
    })


@app.route('/history')
def history():
    """Return today's generated images, newest first."""
    today = datetime.now().strftime("%Y%m%d")
    images = []
    try:
        for filename in os.listdir(OUTPUT_DIR):
            if not filename.endswith('.png'): continue
            parts = filename.split('_')
            if len(parts) >= 3 and parts[1] == today:
                time_str = parts[2]
                display_time = f"{time_str[:2]}:{time_str[2:4]}:{time_str[4:6]}" if len(time_str) == 6 else time_str
                prompt = None
                prompt_file = os.path.join(OUTPUT_DIR, filename.rsplit('.', 1)[0] + '.prompt')
                if os.path.exists(prompt_file):
                    try:
                        with open(prompt_file, 'r') as f:
                            for line in f:
                                if line.startswith('# Prompt: '):
                                    prompt = line[10:].strip()
                                    break
                    except Exception: pass
                images.append({'filename': filename, 'time': display_time, 'prompt': prompt, 'sort_key': parts[2] if len(parts) >= 3 else '000000'})
        images.sort(key=lambda x: x['sort_key'], reverse=True)
        for img in images: del img['sort_key']
    except Exception as e:
        print(f"Error reading history: {e}")
    return jsonify({'images': images})


@app.route('/archive', methods=['POST'])
def archive_today():
    today = datetime.now().strftime("%Y%m%d")
    archive_dir = os.path.join(OUTPUT_DIR, "archive")
    os.makedirs(archive_dir, exist_ok=True)
    moved = 0
    try:
        for filename in os.listdir(OUTPUT_DIR):
            filepath = os.path.join(OUTPUT_DIR, filename)
            if not os.path.isfile(filepath): continue
            parts = filename.split('_')
            if len(parts) >= 3 and parts[1] == today:
                shutil.move(filepath, os.path.join(archive_dir, filename))
                moved += 1
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})
    return jsonify({'success': True, 'moved': moved})


@app.route('/delete', methods=['POST'])
def delete_today():
    """Permanently delete today's generated files (PNG + .prompt).

    If a single filename is provided in the JSON body, only that image (and its
    sidecar .prompt) is deleted. Otherwise all of today's files are removed.
    """
    today = datetime.now().strftime("%Y%m%d")
    body = request.get_json(silent=True) or {}
    target = body.get('filename')
    deleted = 0

    def _remove_pair(png_name):
        nonlocal deleted
        png_path = os.path.join(OUTPUT_DIR, png_name)
        if os.path.isfile(png_path):
            os.remove(png_path)
            deleted += 1
        prompt_path = os.path.join(OUTPUT_DIR, png_name.rsplit('.', 1)[0] + '.prompt')
        if os.path.isfile(prompt_path):
            os.remove(prompt_path)

    try:
        if target:
            # Guard against path traversal and ensure it's a today image
            if '/' in target or '\\' in target or not target.endswith('.png'):
                return jsonify({'success': False, 'error': 'Invalid filename'}), 400
            parts = target.split('_')
            if len(parts) < 3 or parts[1] != today:
                return jsonify({'success': False, 'error': 'Not a today image'}), 400
            _remove_pair(target)
        else:
            for filename in os.listdir(OUTPUT_DIR):
                filepath = os.path.join(OUTPUT_DIR, filename)
                if not os.path.isfile(filepath): continue
                parts = filename.split('_')
                if len(parts) >= 3 and parts[1] == today:
                    os.remove(filepath)
                    deleted += 1
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})
    return jsonify({'success': True, 'deleted': deleted})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="FLUX Web Server")
    parser.add_argument("--local-encoder", action="store_true", help="Use local text encoder instead of remote API")
    parser.add_argument("--full-model", action="store_true", help="Use full FLUX model instead of 4-bit quantized")
    parser.add_argument("--gguf", type=str, choices=["bf16", "q8", "q4"], default=None, help="Use GGUF model")
    parser.add_argument("--flux2", action="store_true", help="Use FLUX.2 model")
    parser.add_argument("--schnell", action="store_true", help="Use FLUX.1-schnell")
    parser.add_argument("--klein", action="store_true", help="Use FLUX.2-klein (9B) instead of FLUX.2-dev (32B). Implies --flux2 --full-model")
    parser.add_argument("--turbo", action="store_true", default=None, help="Enable turbo LoRA")
    parser.add_argument("--no-turbo", action="store_true", help="Disable turbo LoRA")
    parser.add_argument("--uncensored", action="store_true", help="Load Flux-Uncensored-V2 LoRA")
    parser.add_argument("--port", type=int, default=PORT, help=f"Port (default: {PORT})")
    args = parser.parse_args()

    if args.klein:
        args.flux2 = True
        args.full_model = True
    _full_model, _gguf_quant, _flux2, _schnell, _uncensored, _klein = args.full_model, args.gguf, args.flux2, args.schnell, args.uncensored, args.klein
    _local_encoder = args.local_encoder or args.full_model or args.schnell or args.uncensored
    if args.uncensored and not args.full_model: _full_model = True
    # Turbo LoRA is a FLUX.2-dev LoRA — don't auto-enable for klein (different architecture)
    _turbo = (args.turbo or (args.flux2 and not _klein)) and not args.no_turbo

    print(f"Loading {'FLUX.2' if _flux2 else 'FLUX.1'}...")
    load_model(local_encoder=_local_encoder, full_model=_full_model, gguf_quant=_gguf_quant, flux2=_flux2, schnell=_schnell, for_lora=_uncensored, klein=_klein)
    if _turbo: load_turbo_lora()
    if _uncensored: load_uncensored_lora()
    print(f"\nStarting web server on http://0.0.0.0:{args.port}")
    app.run(host='0.0.0.0', port=args.port, threaded=True)
