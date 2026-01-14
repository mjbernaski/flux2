#!/usr/bin/env python3
"""Test all model configurations with and without reference images."""

import sys
import os
import gc
import torch
from PIL import Image
import io

# Test configurations - all use local_encoder=True to avoid remote API dependency
CONFIGS = [
    {"name": "FLUX.1 4-bit BNB", "args": {"flux2": False, "full_model": False, "gguf_quant": None, "local_encoder": True}},
    {"name": "FLUX.1 Full", "args": {"flux2": False, "full_model": True, "gguf_quant": None, "local_encoder": True}},
    {"name": "FLUX.1 GGUF Q8", "args": {"flux2": False, "full_model": False, "gguf_quant": "q8", "local_encoder": True}},
    {"name": "FLUX.1-schnell", "args": {"flux2": False, "full_model": False, "gguf_quant": None, "local_encoder": True, "schnell": True}},
    {"name": "FLUX.2 4-bit BNB", "args": {"flux2": True, "full_model": False, "gguf_quant": None, "local_encoder": True}},
    {"name": "FLUX.2 Full", "args": {"flux2": True, "full_model": True, "gguf_quant": None, "local_encoder": True}},
    {"name": "FLUX.2 Full + Turbo", "args": {"flux2": True, "full_model": True, "gguf_quant": None, "local_encoder": True}, "turbo": True},
]

def create_test_image(width=256, height=256):
    """Create a simple test image for img2img."""
    img = Image.new('RGB', (width, height), color=(128, 128, 200))
    return img

def reset_model_state():
    """Reset the model state between tests."""
    import fl24bit
    fl24bit.transformer = None
    fl24bit.pipe = None
    fl24bit.pipe_img2img = None
    fl24bit._model_type = None
    fl24bit._flux_version = 1
    fl24bit._turbo_enabled = False
    fl24bit._schnell_enabled = False
    gc.collect()
    torch.cuda.empty_cache()

def test_config(config, test_img2img=True):
    """Test a single configuration."""
    import fl24bit
    from fl24bit import load_model, generate_image, load_turbo_lora

    name = config["name"]
    args = config["args"]
    use_turbo = config.get("turbo", False)

    print(f"\n{'='*60}")
    print(f"TESTING: {name}")
    print(f"{'='*60}")

    try:
        # Load model
        print(f"\n[1/3] Loading model...")
        load_model(**args)

        if use_turbo:
            print(f"\n[1b] Loading Turbo LoRA...")
            load_turbo_lora()

        # Test text-to-image (2 steps for speed)
        print(f"\n[2/3] Testing text-to-image...")
        image, seed, timings = generate_image(
            "a red cube on a white background",
            steps=2,
            width=256,
            height=256,
            local_encoder=args.get("local_encoder", False)
        )
        print(f"  SUCCESS: Generated {image.size}, seed={seed}")
        print(f"  Timings: encoding={timings['encoding']:.2f}s, diffusion={timings['diffusion']:.2f}s")

        # Test img2img (only for FLUX.1, FLUX.2 uses different path)
        if test_img2img:
            print(f"\n[3/3] Testing img2img...")
            test_image = create_test_image()
            image2, seed2, timings2 = generate_image(
                "a blue cube on a white background",
                steps=2,
                width=256,
                height=256,
                local_encoder=args.get("local_encoder", False),
                input_image=test_image,
                strength=0.75
            )
            print(f"  SUCCESS: Generated {image2.size}, seed={seed2}")
            print(f"  Timings: encoding={timings2['encoding']:.2f}s, diffusion={timings2['diffusion']:.2f}s")

        print(f"\n[PASS] {name} - All tests passed!")
        return True

    except Exception as e:
        print(f"\n[FAIL] {name} - Error: {e}")
        import traceback
        traceback.print_exc()
        return False

    finally:
        # Clean up
        reset_model_state()

def main():
    # Check which config to test (or all)
    if len(sys.argv) > 1:
        config_num = int(sys.argv[1])
        if 1 <= config_num <= len(CONFIGS):
            configs_to_test = [CONFIGS[config_num - 1]]
        else:
            print(f"Invalid config number. Choose 1-{len(CONFIGS)}")
            sys.exit(1)
    else:
        configs_to_test = CONFIGS

    print("=" * 60)
    print("FLUX Model Configuration Test Suite")
    print("=" * 60)
    print(f"\nConfigurations to test:")
    for i, config in enumerate(CONFIGS, 1):
        marker = "*" if config in configs_to_test else " "
        print(f"  {marker} {i}. {config['name']}")

    print(f"\nUsage: python test_all_configs.py [config_number]")
    print(f"  No argument = test all configs sequentially")
    print(f"  With number = test only that config\n")

    results = {}
    for config in configs_to_test:
        success = test_config(config)
        results[config["name"]] = success

    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    for name, success in results.items():
        status = "PASS" if success else "FAIL"
        print(f"  [{status}] {name}")

    passed = sum(1 for s in results.values() if s)
    total = len(results)
    print(f"\nTotal: {passed}/{total} passed")

    return 0 if passed == total else 1

if __name__ == "__main__":
    sys.exit(main())
