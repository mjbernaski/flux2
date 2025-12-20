import argparse
import torch
from diffusers import Flux2Pipeline, Flux2Transformer2DModel
from huggingface_hub import get_token
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import io
from datetime import datetime
import uuid
import time
import subprocess
import shutil

repo_id = "diffusers/FLUX.2-dev-bnb-4bit"
device = "cuda:0"
torch_dtype = torch.bfloat16

# Lazy-loaded model components
transformer = None
pipe = None

# Connection pooling with retry strategy for transient failures
_session = requests.Session()
_retry_strategy = Retry(
    total=3,
    backoff_factor=0.5,
    status_forcelist=[500, 502, 503, 504],
    allowed_methods=["POST"],
    raise_on_status=False,
)
_adapter = HTTPAdapter(max_retries=_retry_strategy)
_session.mount("https://", _adapter)

# Embedding cache - avoids redundant API calls for same prompts
_embedding_cache = {}


def load_model(local_encoder=False):
    """Load the FLUX.2 model components. Call this before generating images."""
    global transformer, pipe
    if pipe is not None:
        return {}  # Already loaded

    load_timings = {}
    total_start = time.perf_counter()

    print("Loading FLUX.2 transformer...")
    t0 = time.perf_counter()
    transformer = Flux2Transformer2DModel.from_pretrained(
        repo_id, subfolder="transformer", torch_dtype=torch_dtype
    )
    load_timings['transformer'] = time.perf_counter() - t0
    print(f"  Transformer loaded in {load_timings['transformer']:.2f}s")

    print("Loading FLUX.2 pipeline...")
    t0 = time.perf_counter()
    if local_encoder:
        print("Loading local text encoders (this requires more VRAM)...")
        pipe = Flux2Pipeline.from_pretrained(
            repo_id, transformer=transformer, torch_dtype=torch_dtype
        )
        load_timings['pipeline'] = time.perf_counter() - t0
        print(f"  Pipeline loaded in {load_timings['pipeline']:.2f}s")

        print("Moving model to GPU...")
        t0 = time.perf_counter()
        pipe = pipe.to(device)
        load_timings['to_device'] = time.perf_counter() - t0
        print(f"  Moved to GPU in {load_timings['to_device']:.2f}s")
    else:
        pipe = Flux2Pipeline.from_pretrained(
            repo_id, transformer=transformer, text_encoder=None, torch_dtype=torch_dtype
        )
        load_timings['pipeline'] = time.perf_counter() - t0
        print(f"  Pipeline loaded in {load_timings['pipeline']:.2f}s")

        print("Moving model to GPU...")
        t0 = time.perf_counter()
        pipe = pipe.to(device)
        load_timings['to_device'] = time.perf_counter() - t0
        print(f"  Moved to GPU in {load_timings['to_device']:.2f}s")

    load_timings['total'] = time.perf_counter() - total_start
    print(f"Model loaded successfully in {load_timings['total']:.2f}s")
    print(f"  Summary: transformer={load_timings['transformer']:.2f}s, pipeline={load_timings['pipeline']:.2f}s, to_gpu={load_timings['to_device']:.2f}s")

    return load_timings


def remote_text_encoder(prompt, use_cache=True):
    if use_cache and prompt in _embedding_cache:
        return _embedding_cache[prompt]

    response = _session.post(
        "https://remote-text-encoder-flux-2.huggingface.co/predict",
        json={"prompt": prompt},
        headers={
            "Authorization": f"Bearer {get_token()}",
            "Content-Type": "application/json"
        },
        timeout=(10, 60),  # (connect timeout, read timeout) in seconds
    )
    response.raise_for_status()
    prompt_embeds = torch.load(io.BytesIO(response.content))
    result = prompt_embeds.to(device)

    if use_cache:
        _embedding_cache[prompt] = result

    return result

def generate_image(prompt, seed=None, steps=6, width=1024, height=1024, local_encoder=False):
    if seed is None:
        seed = torch.randint(0, 2**32, (1,)).item()
    print(f"Using seed: {seed}")

    timings = {}

    # Generate with inference_mode for better performance
    with torch.inference_mode():
        if local_encoder:
            # Use local text encoder - encoding happens inside pipe()
            t0 = time.perf_counter()
            image = pipe(
                prompt=prompt,
                generator=torch.Generator(device=device).manual_seed(seed),
                num_inference_steps=steps,
                guidance_scale=4,
                width=width,
                height=height,
            ).images[0]
            timings['diffusion'] = time.perf_counter() - t0
            timings['encoding'] = 0  # Included in diffusion for local
        else:
            # Use remote text encoder - get embeddings first
            t0 = time.perf_counter()
            embeds = remote_text_encoder(prompt)
            timings['encoding'] = time.perf_counter() - t0

            t0 = time.perf_counter()
            image = pipe(
                prompt_embeds=embeds,
                generator=torch.Generator(device=device).manual_seed(seed),
                num_inference_steps=steps,
                guidance_scale=4,
                width=width,
                height=height,
            ).images[0]
            timings['diffusion'] = time.perf_counter() - t0

    timings['total'] = timings['encoding'] + timings['diffusion']

    print(f"Timing: encoding={timings['encoding']:.2f}s, diffusion={timings['diffusion']:.2f}s, total={timings['total']:.2f}s")
    return image, seed, timings

def compile_pipeline():
    """Compile transformer for faster inference (slower first run, faster subsequent)"""
    print("Compiling transformer (this may take a minute)...")
    t0 = time.perf_counter()
    pipe.transformer = torch.compile(pipe.transformer, mode="reduce-overhead")
    print(f"Compilation done in {time.perf_counter() - t0:.1f}s")


def play_completion_sound():
    """Play a pleasant notification sound when image generation is complete."""
    # Try paplay (PulseAudio) with system sounds first
    if shutil.which("paplay"):
        sound_paths = [
            "/usr/share/sounds/freedesktop/stereo/complete.oga",
            "/usr/share/sounds/gnome/default/alerts/glass.ogg",
            "/usr/share/sounds/ubuntu/stereo/message.ogg",
            "/usr/share/sounds/freedesktop/stereo/message.oga",
        ]
        for sound in sound_paths:
            try:
                subprocess.run(["paplay", sound], stderr=subprocess.DEVNULL, timeout=2)
                return
            except (subprocess.SubprocessError, FileNotFoundError):
                continue

    # Try aplay with a beep
    if shutil.which("aplay"):
        try:
            subprocess.run(["aplay", "-q", "/usr/share/sounds/alsa/Front_Center.wav"],
                         stderr=subprocess.DEVNULL, timeout=2)
            return
        except (subprocess.SubprocessError, FileNotFoundError):
            pass

    # Fallback to terminal bell
    print("\a", end="", flush=True)

def main():
    parser = argparse.ArgumentParser(description="FLUX.2 Image Generator")
    parser.add_argument("--steps", type=int, default=25, help="Number of inference steps (default: 25)")
    parser.add_argument("--compile", action="store_true", help="Compile model for faster inference (slower startup)")
    parser.add_argument("--local-encoder", action="store_true", help="Use local text encoder instead of remote API (requires more VRAM)")
    args = parser.parse_args()

    # Load the model
    load_model(local_encoder=args.local_encoder)

    if args.compile:
        compile_pipeline()

    image_count = 0
    current_prompt = None
    last_seed = None

    steps = args.steps

    # Orientation presets (width, height) at 1K base
    orientations_1k = {
        'square': (1024, 1024),
        'portrait': (768, 1344),
        'landscape': (1344, 768),
        '16:9': (1360, 768),
    }
    # Size presets
    sizes = {
        '1k': 1.0,
        '2k': 2.0,
        '4k': 4.0,
    }
    orientation = 'landscape'
    size = '1k'
    base_w, base_h = orientations_1k[orientation]
    width, height = int(base_w * sizes[size]), int(base_h * sizes[size])

    print("\n=== FLUX.2 Image Generator ===")
    encoder_mode = "local encoder" if args.local_encoder else "remote encoder"
    print(f"Using {steps} inference steps, {size} {orientation} ({width}x{height}), {encoder_mode}" + (" (compiled)" if args.compile else ""))
    print("Commands:")
    print("  'quit' or 'q' - Exit the program")
    print("  'same' or 's' - Regenerate with same prompt (uses cached embeddings)")
    print("  'reseed <number>' - Regenerate with specific seed")
    print("  '/steps <number>' - Change inference steps (current: {})".format(steps))
    print("  '/square' - Set square aspect ratio")
    print("  '/portrait' - Set portrait aspect ratio")
    print("  '/landscape' - Set landscape aspect ratio")
    print("  '/16:9' - Set 16:9 widescreen aspect ratio")
    print("  '/1k' - Set 1K resolution (default)")
    print("  '/2k' - Set 2K resolution")
    print("  '/4k' - Set 4K resolution")
    print("  Or enter a prompt (can include modifiers: 'a cat /4k /portrait')\n")

    while True:
        if current_prompt is None:
            user_input = input("Enter your prompt: ").strip()
        else:
            print(f"\nCurrent prompt: {current_prompt[:80]}{'...' if len(current_prompt) > 80 else ''}")
            user_input = input("Enter new prompt, modification, or command: ").strip()

        if not user_input:
            print("Please enter a prompt or command.")
            continue

        lower_input = user_input.lower()

        if lower_input in ('quit', 'q'):
            print("Goodbye!")
            break

        if lower_input.startswith('/steps '):
            try:
                new_steps = int(user_input.split()[1])
                if new_steps < 1:
                    print("Steps must be at least 1.")
                    continue
                steps = new_steps
                print(f"Inference steps set to {steps}")
            except (ValueError, IndexError):
                print("Invalid steps. Usage: /steps 10")
            continue

        if lower_input in ('/square', '/portrait', '/landscape', '/16:9'):
            orientation = lower_input[1:]  # Remove the leading /
            base_w, base_h = orientations_1k[orientation]
            width, height = int(base_w * sizes[size]), int(base_h * sizes[size])
            print(f"Orientation set to {orientation} ({width}x{height})")
            continue

        if lower_input in ('/1k', '/2k', '/4k'):
            size = lower_input[1:]  # Remove the leading /
            base_w, base_h = orientations_1k[orientation]
            width, height = int(base_w * sizes[size]), int(base_h * sizes[size])
            print(f"Size set to {size} ({width}x{height})")
            continue

        # Parse inline modifiers from prompt (e.g., "a cat /4k /portrait")
        words = user_input.split()
        modifiers_found = []
        prompt_words = []
        for word in words:
            lower_word = word.lower()
            if lower_word in ('/1k', '/2k', '/4k'):
                size = lower_word[1:]
                modifiers_found.append(f"size={size}")
            elif lower_word in ('/square', '/portrait', '/landscape', '/16:9'):
                orientation = lower_word[1:]
                modifiers_found.append(f"orientation={orientation}")
            else:
                prompt_words.append(word)

        if modifiers_found:
            base_w, base_h = orientations_1k[orientation]
            width, height = int(base_w * sizes[size]), int(base_h * sizes[size])
            print(f"Applied: {', '.join(modifiers_found)} -> {width}x{height}")

        # Reconstruct prompt without modifiers
        user_input = ' '.join(prompt_words)
        lower_input = user_input.lower()

        if not user_input:
            # Input was only modifiers, no prompt
            continue

        if lower_input in ('same', 's') and current_prompt:
            prompt = current_prompt
        elif lower_input.startswith('reseed ') and current_prompt:
            try:
                last_seed = int(lower_input.split()[1])
                prompt = current_prompt
            except (ValueError, IndexError):
                print("Invalid seed. Usage: reseed 12345")
                continue
        else:
            prompt = user_input
            current_prompt = prompt
            last_seed = None

        print(f"\nGenerating image ({steps} steps, {size} {orientation} {width}x{height})...")
        image, last_seed, timings = generate_image(prompt, last_seed if lower_input.startswith('reseed ') else None, steps, width, height, args.local_encoder)

        image_count += 1
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_id = uuid.uuid4().hex[:8]
        filename = f"flux2_{timestamp}_{unique_id}.png"

        t0 = time.perf_counter()
        image.save(filename)
        timings['save'] = time.perf_counter() - t0

        print(f"Image saved as: {filename}")
        print(f"  Steps breakdown: encoding={timings['encoding']:.2f}s, diffusion={timings['diffusion']:.2f}s, save={timings['save']:.2f}s")
        play_completion_sound()

if __name__ == "__main__":
    main()
