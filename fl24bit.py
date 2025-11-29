import argparse
import torch
from diffusers import Flux2Pipeline, Flux2Transformer2DModel
from diffusers.utils import load_image
from huggingface_hub import get_token
import requests
import io
from datetime import datetime
import uuid

repo_id = "diffusers/FLUX.2-dev-bnb-4bit"
device = "cuda:0"
torch_dtype = torch.bfloat16

transformer = Flux2Transformer2DModel.from_pretrained(
    repo_id, subfolder="transformer", torch_dtype=torch_dtype
)

def remote_text_encoder(prompts):
    response = requests.post(
        "https://remote-text-encoder-flux-2.huggingface.co/predict",
        json={"prompt": prompts},
        headers={
            "Authorization": f"Bearer {get_token()}",
            "Content-Type": "application/json"
        }
    )
    prompt_embeds = torch.load(io.BytesIO(response.content))

    return prompt_embeds.to(device)

pipe = Flux2Pipeline.from_pretrained(
    repo_id, transformer=transformer, text_encoder=None, torch_dtype=torch_dtype
).to(device)

def generate_image(prompt, seed=None, steps=6):
    if seed is None:
        seed = torch.randint(0, 2**32, (1,)).item()
    print(f"Using seed: {seed}")

    image = pipe(
        prompt_embeds=remote_text_encoder(prompt),
        generator=torch.Generator(device=device).manual_seed(seed),
        num_inference_steps=steps,
        guidance_scale=4,
    ).images[0]

    return image, seed

def main():
    parser = argparse.ArgumentParser(description="FLUX.2 Image Generator")
    parser.add_argument("--steps", type=int, default=6, help="Number of inference steps (default: 6)")
    args = parser.parse_args()

    image_count = 0
    current_prompt = None
    last_seed = None

    print("\n=== FLUX.2 Image Generator ===")
    print(f"Using {args.steps} inference steps")
    print("Commands:")
    print("  'quit' or 'q' - Exit the program")
    print("  'same' or 's' - Regenerate with same prompt (new seed)")
    print("  'reseed <number>' - Regenerate with specific seed")
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

        print(f"\nGenerating image...")
        image, last_seed = generate_image(prompt, last_seed if lower_input.startswith('reseed ') else None, args.steps)

        image_count += 1
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_id = uuid.uuid4().hex[:8]
        filename = f"flux2_{timestamp}_{unique_id}.png"
        image.save(filename)
        print(f"Image saved as: {filename}")

if __name__ == "__main__":
    main()
