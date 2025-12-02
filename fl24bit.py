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

repo_id = "diffusers/FLUX.2-dev-bnb-4bit"
device = "cuda:0"
torch_dtype = torch.bfloat16

transformer = Flux2Transformer2DModel.from_pretrained(
    repo_id, subfolder="transformer", torch_dtype=torch_dtype
)

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

pipe = Flux2Pipeline.from_pretrained(
    repo_id, transformer=transformer, text_encoder=None, torch_dtype=torch_dtype
).to(device)

def generate_image(prompt, seed=None, steps=6, width=1024, height=1024):
    if seed is None:
        seed = torch.randint(0, 2**32, (1,)).item()
    print(f"Using seed: {seed}")

    # Get embeddings (cached if same prompt)
    t0 = time.perf_counter()
    embeds = remote_text_encoder(prompt)
    t_embed = time.perf_counter() - t0

    # Generate with inference_mode for better performance
    t1 = time.perf_counter()
    with torch.inference_mode():
        image = pipe(
            prompt_embeds=embeds,
            generator=torch.Generator(device=device).manual_seed(seed),
            num_inference_steps=steps,
            guidance_scale=4,
            width=width,
            height=height,
        ).images[0]
    t_gen = time.perf_counter() - t1

    print(f"Timing: embed={t_embed:.2f}s, generate={t_gen:.2f}s, total={t_embed+t_gen:.2f}s")
    return image, seed

def compile_pipeline():
    """Compile transformer for faster inference (slower first run, faster subsequent)"""
    print("Compiling transformer (this may take a minute)...")
    t0 = time.perf_counter()
    pipe.transformer = torch.compile(pipe.transformer, mode="reduce-overhead")
    print(f"Compilation done in {time.perf_counter() - t0:.1f}s")

def main():
    parser = argparse.ArgumentParser(description="FLUX.2 Image Generator")
    parser.add_argument("--steps", type=int, default=25, help="Number of inference steps (default: 25)")
    parser.add_argument("--compile", action="store_true", help="Compile model for faster inference (slower startup)")
    args = parser.parse_args()

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
    }
    # Size presets
    sizes = {
        '1k': 1.0,
        '2k': 2.0,
    }
    orientation = 'landscape'
    size = '1k'
    base_w, base_h = orientations_1k[orientation]
    width, height = int(base_w * sizes[size]), int(base_h * sizes[size])

    print("\n=== FLUX.2 Image Generator ===")
    print(f"Using {steps} inference steps, {size} {orientation} ({width}x{height})" + (" (compiled)" if args.compile else ""))
    print("Commands:")
    print("  'quit' or 'q' - Exit the program")
    print("  'same' or 's' - Regenerate with same prompt (uses cached embeddings)")
    print("  'reseed <number>' - Regenerate with specific seed")
    print("  '/steps <number>' - Change inference steps (current: {})".format(steps))
    print("  '/square' - Set square aspect ratio")
    print("  '/portrait' - Set portrait aspect ratio")
    print("  '/landscape' - Set landscape aspect ratio")
    print("  '/1k' - Set 1K resolution (default)")
    print("  '/2k' - Set 2K resolution")
    print("  Or enter a new/modified prompt\n")

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

        if lower_input in ('/square', '/portrait', '/landscape'):
            orientation = lower_input[1:]  # Remove the leading /
            base_w, base_h = orientations_1k[orientation]
            width, height = int(base_w * sizes[size]), int(base_h * sizes[size])
            print(f"Orientation set to {orientation} ({width}x{height})")
            continue

        if lower_input in ('/1k', '/2k'):
            size = lower_input[1:]  # Remove the leading /
            base_w, base_h = orientations_1k[orientation]
            width, height = int(base_w * sizes[size]), int(base_h * sizes[size])
            print(f"Size set to {size} ({width}x{height})")
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
        image, last_seed = generate_image(prompt, last_seed if lower_input.startswith('reseed ') else None, steps, width, height)

        image_count += 1
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_id = uuid.uuid4().hex[:8]
        filename = f"flux2_{timestamp}_{unique_id}.png"
        image.save(filename)
        print(f"Image saved as: {filename}")

if __name__ == "__main__":
    main()
