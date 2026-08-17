#!/usr/bin/env python3
"""Interactive CLI (REPL) for the FLUX image generator.

Model loading and generation live in flux_core; this module is just the
command-line front end.
"""
import argparse
import os
import socket
import time
import uuid
from datetime import datetime

from PIL import Image

import flux_core
from flux_core import (
    load_model,
    generate_image,
    compile_pipeline,
    save_prompt_file,
    play_completion_sound,
    MAX_REFERENCE_IMAGES,
)


def main():
    parser = argparse.ArgumentParser(description="FLUX Image Generator")
    parser.add_argument("--steps", type=int, default=25, help="Number of inference steps (default: 25)")
    parser.add_argument("--guidance", type=float, default=None, help="Guidance scale (default: 4 for normal, 2.5 for turbo). Lower values give more variety.")
    parser.add_argument("--compile", action="store_true", help="Compile model for faster inference (slower startup)")
    parser.add_argument("--local-encoder", action="store_true", help="Use local text encoder instead of remote API (requires more VRAM)")
    parser.add_argument("--full-model", action="store_true", help="Use full FLUX model instead of 4-bit quantized (requires more VRAM)")
    parser.add_argument("--gguf", type=str, choices=["bf16", "q8", "q4"], default=None,
                        help="Use GGUF model (FLUX.1 only, recommended for DGX Spark). Options: bf16 (full quality), q8 (8-bit), q4 (4-bit smallest)")
    parser.add_argument("--flux2", action="store_true", help="Use FLUX.2 model instead of FLUX.1 (requires more VRAM)")
    parser.add_argument("--schnell", action="store_true", help="Use FLUX.1-schnell (fast 4-step model, Apache 2.0 license)")
    parser.add_argument("--kontext", action="store_true", help="Use FLUX.1 Kontext, an instruction-based image editor (4-bit; add --full-model for full bf16). Pass --image and a prompt describing the edit.")
    parser.add_argument("--vae-tiling", action="store_true", help="Decode the VAE in overlapping tiles, bounding the peak memory of the full-resolution final decode. Use when generations above ~1MP stall at the end")
    parser.add_argument("--image", type=str, nargs='+', default=None, metavar="PATH",
                        help=f"Reference image path(s) for img2img/editing, up to {MAX_REFERENCE_IMAGES}. "
                             "Multiple references need --kontext (stitched side-by-side) or --flux2 (native)")
    parser.add_argument("--strength", type=float, default=0.75, help="Denoising strength for img2img (0.0-1.0, default: 0.75). Higher = more change from original")
    args = parser.parse_args()

    # Full model, schnell, and Kontext always use local encoder
    use_local_encoder = args.local_encoder or args.full_model or args.schnell or args.kontext

    # Load the model
    load_model(local_encoder=use_local_encoder, full_model=args.full_model, gguf_quant=args.gguf, flux2=args.flux2, schnell=args.schnell, kontext=args.kontext, vae_tiling=args.vae_tiling)

    if args.compile:
        compile_pipeline()

    image_count = 0
    current_prompt = None
    last_seed = None

    steps = args.steps
    guidance_scale = args.guidance  # None means use default (4 for normal, 2.5 for turbo)
    strength = args.strength  # Denoising strength for img2img
    def load_reference_images(paths):
        """Load up to MAX_REFERENCE_IMAGES paths; returns [] if any fails."""
        if len(paths) > MAX_REFERENCE_IMAGES:
            print(f"At most {MAX_REFERENCE_IMAGES} reference images are supported.")
            return []
        loaded = []
        for path in paths:
            try:
                img = Image.open(path).convert("RGB")
            except Exception as e:
                print(f"Could not load image '{path}': {e}")
                return []
            loaded.append(img)
            print(f"Loaded reference image {len(loaded)}: {path} ({img.size[0]}x{img.size[1]})")
        return loaded

    input_images = []
    if args.image:
        input_images = load_reference_images(args.image)
        if not input_images:
            print("Continuing in text-to-image mode (no img2img).")

    if args.kontext and not input_images:
        print("Note: --kontext is an image editor; load a source image with --image "
              "or '/image <path>' before prompting, or output will be plain txt2img.")

    # Orientation presets (width, height) at 1K base
    orientations_1k = {
        'square': (1024, 1024),
        'portrait': (768, 1344),
        'landscape': (1360, 768),  # 16:9
        'widescreen': (1568, 672),  # ~21:9 extra-wide
        'extra-tall': (672, 1568),  # ~9:21 mirror of widescreen
    }
    # Size presets
    sizes = {
        '0.75': 0.75,
        '1k': 1.0,
        '2k': 2.0,
        '4k': 4.0,
    }
    orientation = 'landscape'
    size = '1k'
    base_w, base_h = orientations_1k[orientation]
    width, height = int(base_w * sizes[size]), int(base_h * sizes[size])

    flux_name = f"FLUX.{flux_core._flux_version}"
    hostname = socket.gethostname()
    print(f"\n=== {flux_name} Image Generator on {hostname} ===")
    if args.kontext:
        model_mode = "Kontext editor (full bf16)" if args.full_model else "Kontext editor (4-bit)"
    elif args.gguf:
        model_mode = f"GGUF {args.gguf.upper()}"
    elif args.full_model:
        model_mode = "full model"
    else:
        model_mode = "4-bit BNB"
    encoder_mode = "local encoder" if use_local_encoder else "remote encoder"
    compiled_str = " (compiled)" if args.compile else ""
    print(f"Model: {flux_name} {model_mode}, {encoder_mode}{compiled_str}")
    print(f"Output: {size} {orientation} ({width}x{height}), {steps} steps")
    def print_commands():
        print("Commands:")
        print("  'quit' or 'q' - Exit the program")
        print("  'same' or 's' - Regenerate with same prompt (cached embeddings with remote encoder)")
        print("  'reseed <number>' - Regenerate with specific seed")
        print("  '/help' - Show this command list")
        print("  '/steps <number>' - Change inference steps (current: {})".format(steps))
        print("  '/guidance <number>' - Change guidance scale (current: {}, lower=more variety)".format(guidance_scale if guidance_scale is not None else "auto"))
        print("  '/image <path> [path2] [path3]' - Load reference image(s); multiple need Kontext or FLUX.2 ('/image clear' to remove)")
        print("  '/strength <number>' - Set img2img denoising strength 0.0-1.0 (current: {})".format(strength))
        print("  '/square' - Set square aspect ratio")
        print("  '/portrait' - Set portrait aspect ratio")
        print("  '/landscape' - Set 16:9 landscape aspect ratio")
        print("  '/widescreen' - Set 21:9 extra-wide aspect ratio")
        print("  '/extra-tall' - Set 9:21 extra-tall aspect ratio")
        print("  '/0.75' - Set 0.75K resolution (smaller/faster)")
        print("  '/1k' - Set 1K resolution (default)")
        print("  '/2k' - Set 2K resolution")
        print("  '/4k' - Set 4K resolution")
        print("  Or enter a prompt (can include modifiers: 'a cat /4k /portrait')\n")

    print_commands()

    size_modifiers = ('/0.75', '/1k', '/2k', '/4k')
    orientation_modifiers = ('/square', '/portrait', '/landscape', '/widescreen', '/extra-tall')

    while True:
        try:
            if current_prompt is None:
                user_input = input("Enter your prompt: ").strip()
            else:
                print(f"\nCurrent prompt: {current_prompt[:80]}{'...' if len(current_prompt) > 80 else ''}")
                user_input = input("Enter new prompt, modification, or command: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not user_input:
            print("Please enter a prompt or command.")
            continue

        lower_input = user_input.lower()

        if lower_input in ('quit', 'q'):
            print("Goodbye!")
            break

        tokens = user_input.split()
        cmd = tokens[0].lower()

        if cmd == '/help':
            print_commands()
            continue

        if cmd == '/steps':
            try:
                new_steps = int(tokens[1])
                if new_steps < 1:
                    print("Steps must be at least 1.")
                    continue
                steps = new_steps
                print(f"Inference steps set to {steps}")
            except (ValueError, IndexError):
                print("Invalid steps. Usage: /steps 10")
            continue

        if cmd == '/guidance':
            try:
                new_guidance = float(tokens[1])
                if new_guidance < 0:
                    print("Guidance scale must be non-negative.")
                    continue
                guidance_scale = new_guidance
                print(f"Guidance scale set to {guidance_scale}")
            except (ValueError, IndexError):
                print("Invalid guidance scale. Usage: /guidance 3.5")
            continue

        if cmd == '/strength':
            try:
                new_strength = float(tokens[1])
                if new_strength <= 0 or new_strength > 1:
                    print("Strength must be between 0.01 and 1.0 (0.0 would result in zero pipeline steps).")
                    continue
                strength = new_strength
                print(f"Img2img strength set to {strength}")
            except (ValueError, IndexError):
                print("Invalid strength. Usage: /strength 0.75")
            continue

        if cmd == '/image':
            image_arg = user_input.split(maxsplit=1)[1] if len(tokens) > 1 else ""
            if not image_arg:
                print(f"Usage: /image <path> [path2] [path3] (up to {MAX_REFERENCE_IMAGES}; '/image clear' to remove)")
            elif image_arg.lower() == 'clear':
                input_images = []
                print("Reference images cleared. Using text-to-image mode.")
            else:
                # A single path may contain spaces, so try the whole argument as
                # one path first; only then treat it as space-separated paths.
                paths = [image_arg] if os.path.exists(image_arg) else image_arg.split()
                loaded = load_reference_images(paths)
                if loaded:
                    input_images = loaded
            continue

        if lower_input in orientation_modifiers:
            orientation = lower_input[1:]  # Remove the leading /
            base_w, base_h = orientations_1k[orientation]
            width, height = int(base_w * sizes[size]), int(base_h * sizes[size])
            print(f"Orientation set to {orientation} ({width}x{height})")
            continue

        if lower_input in size_modifiers:
            size = lower_input[1:]  # Remove the leading /
            base_w, base_h = orientations_1k[orientation]
            width, height = int(base_w * sizes[size]), int(base_h * sizes[size])
            print(f"Size set to {size} ({width}x{height})")
            continue

        # Save the true raw input before modifiers are stripped, so the .prompt
        # sidecar records exactly what the user typed (including /4k etc).
        raw_input_text = user_input

        # Reject typo'd slash-commands instead of silently baking them into the
        # prompt (e.g. '/stepss 10' or '/potrait' should not become image text).
        # Leading commands were handled above, so only size/orientation modifiers
        # are valid inside a prompt at this point.
        unknown_slash = [w for w in user_input.split() if w.startswith('/') and w.lower() not in size_modifiers + orientation_modifiers]
        if unknown_slash:
            print(f"Unknown command/modifier: {', '.join(unknown_slash)}. Type /help for commands. Nothing generated.")
            continue

        # Parse inline modifiers from prompt (e.g., "a cat /4k /portrait")
        words = user_input.split()
        modifiers_found = []
        prompt_words = []
        for word in words:
            lower_word = word.lower()
            if lower_word in size_modifiers:
                size = lower_word[1:]
                modifiers_found.append(f"size={size}")
            elif lower_word in orientation_modifiers:
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

        # Build generation info string
        gen_info_parts = [f"{steps} steps", f"{width}x{height}"]
        if guidance_scale is not None:
            gen_info_parts.append(f"guidance={guidance_scale}")
        if len(input_images) > 1:
            gen_info_parts.append(f"{len(input_images)} reference images")
        elif input_images:
            gen_info_parts.append(f"img2img strength={strength}")
        gen_info = ", ".join(gen_info_parts)

        gen_input = input_images if len(input_images) > 1 else (input_images[0] if input_images else None)
        print(f"\n[{hostname}] Generating: {gen_info}")
        try:
            image, last_seed, timings = generate_image(prompt, last_seed if lower_input.startswith('reseed ') else None, steps, width, height, use_local_encoder, input_image=gen_input, strength=strength, guidance_scale=guidance_scale)
        except KeyboardInterrupt:
            print("\nGeneration cancelled.")
            continue
        except Exception as e:
            print(f"Generation failed: {e}")
            continue

        image_count += 1
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_id = uuid.uuid4().hex[:8]
        filename = f"flux{flux_core._flux_version}_{timestamp}_{unique_id}.png"

        t0 = time.perf_counter()
        image.save(filename)
        timings['save'] = time.perf_counter() - t0

        # Save prompt file alongside image
        save_prompt_file(filename, raw_input_text, prompt, width, height, last_seed, steps, timings, guidance_scale, strength if len(input_images) == 1 else None)

        # Clean summary output
        total_time = timings['total'] + timings['save']
        print(f"[{hostname}] Saved: {filename}")
        print(f"  seed={last_seed}, {total_time:.2f}s total (encode={timings['encoding']:.2f}s, diffuse={timings['diffusion']:.2f}s, save={timings['save']:.2f}s)")
        play_completion_sound()


if __name__ == "__main__":
    main()
