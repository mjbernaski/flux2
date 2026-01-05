# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

FLUX Image Generator - An interactive CLI tool for generating images using FLUX.1 or FLUX.2 diffusion models via the diffusers library. Supports 4-bit quantized, GGUF (FLUX.1 only), and full precision models. Uses a remote text encoder API from Hugging Face for prompt embeddings (or local encoder for full model).

## Running the Application

```bash
# Basic usage (FLUX.1 4-bit)
python fl24bit.py

# FLUX.2 model
python fl24bit.py --flux2

# Full precision model
python fl24bit.py --full-model
python fl24bit.py --flux2 --full-model

# GGUF quantized (FLUX.1 only, recommended for DGX Spark)
python fl24bit.py --gguf q8

# With more inference steps
python fl24bit.py --steps 10

# With torch.compile for faster inference (slower startup)
python fl24bit.py --compile
```

## Web Server

```bash
# FLUX.1 servers
./run_flux1_4bit_server.sh      # 4-bit quantized
./run_flux1_full_server.sh      # Full precision
./run_flux1_gguf_server.sh      # GGUF Q8 (DGX Spark optimized)

# FLUX.2 servers
./run_flux2_4bit_server.sh      # 4-bit quantized
./run_flux2_full_server.sh      # Full precision
```

## Dependencies

Uses `uv pip` for package management. Key dependencies:
- torch (with CUDA support)
- diffusers (FluxPipeline, FluxImg2ImgPipeline, FluxTransformer2DModel)
- huggingface_hub (for authentication and model downloads)
- requests (for remote text encoder API)

## Architecture

Single-file application (`fl24bit.py`) with:

- **Model Loading**: Supports both FLUX.1 and FLUX.2 models:
  - FLUX.1: `diffusers/FLUX.1-dev-bnb-4bit`, `black-forest-labs/FLUX.1-dev`, GGUF variants
  - FLUX.2: `diffusers/FLUX.2-dev-bnb-4bit`, `black-forest-labs/FLUX.2-dev`
- **Remote Text Encoding**: Uses Hugging Face's remote text encoder API (4-bit mode only)
- **Img2Img Support**: FluxImg2ImgPipeline for image-to-image generation
- **Embedding Cache**: Caches prompt embeddings in memory to avoid redundant API calls
- **Connection Pooling**: Uses requests Session with retry strategy for reliable API communication
- **Interactive CLI**: REPL-style interface with commands for regeneration, reseeding, changing steps, and adjusting output dimensions

## Interactive Commands

- `quit`/`q` - Exit
- `same`/`s` - Regenerate with same prompt (uses cached embeddings)
- `reseed <number>` - Regenerate with specific seed
- `/steps <number>` - Change inference steps (default: 25)
- `/square`, `/portrait`, `/landscape` - Change aspect ratio (default: landscape)
- `/1k`, `/2k`, `/4k` - Change resolution multiplier (default: 1k)

**Inline modifiers**: Commands can be embedded in prompts, e.g., `a cat /4k /portrait` will apply settings and generate with "a cat".

## Output

Generated images are saved as `flux{version}_{timestamp}_{uuid}.png` in the working directory (e.g., `flux1_...` or `flux2_...`).

## Requirements

- CUDA-capable GPU
- Hugging Face token (via `huggingface-cli login` or `HF_TOKEN` env var)
- FLUX.2 requires more VRAM than FLUX.1 (32B vs 12B parameters)
