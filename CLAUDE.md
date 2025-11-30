# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

FLUX.2 Image Generator - An interactive CLI tool for generating images using the FLUX.2-dev-bnb-4bit diffusion model via the diffusers library. Uses a remote text encoder API from Hugging Face for prompt embeddings.

## Running the Application

```bash
# Basic usage
python fl24bit.py

# With more inference steps
python fl24bit.py --steps 10

# With torch.compile for faster inference (slower startup)
python fl24bit.py --compile
```

## Dependencies

Uses `uv pip` for package management. Key dependencies:
- torch (with CUDA support)
- diffusers (Flux2Pipeline, Flux2Transformer2DModel)
- huggingface_hub (for authentication and model downloads)
- requests (for remote text encoder API)

## Architecture

Single-file application (`fl24bit.py`) with:

- **Model Loading**: Loads 4-bit quantized FLUX.2 transformer from `diffusers/FLUX.2-dev-bnb-4bit`
- **Remote Text Encoding**: Uses Hugging Face's remote text encoder API (`remote-text-encoder-flux-2.huggingface.co`) instead of local text encoder
- **Embedding Cache**: Caches prompt embeddings in memory to avoid redundant API calls
- **Connection Pooling**: Uses requests Session with retry strategy for reliable API communication
- **Interactive CLI**: REPL-style interface with commands for regeneration, reseeding, changing steps, and adjusting output dimensions

## Interactive Commands

- `quit`/`q` - Exit
- `same`/`s` - Regenerate with same prompt (uses cached embeddings)
- `reseed <number>` - Regenerate with specific seed
- `/steps <number>` - Change inference steps at runtime
- `/square`, `/portrait`, `/landscape` - Change aspect ratio
- `/small`, `/medium`, `/large` - Change resolution multiplier

## Output

Generated images are saved as `flux2_{timestamp}_{uuid}.png` in the working directory.

## Requirements

- CUDA-capable GPU
- Hugging Face token (via `huggingface-cli login` or `HF_TOKEN` env var)
