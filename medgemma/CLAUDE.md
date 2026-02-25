# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

MedGemma is a medical image analysis tool using Google's MedGemma 1.5 4B instruction-tuned model (`google/medgemma-1.5-4b-it`) via the HuggingFace transformers library. It supports multimodal chat (text + images) for medical imaging tasks like X-ray analysis.

## Dependencies

Requires: `transformers`, `torch`, `accelerate`, `pillow`, `requests`, `textual`

```bash
uv pip install transformers torch accelerate pillow requests textual
```

## Running

```bash
python mg_chat.py     # Single-shot script
python mg_tui.py      # Interactive TUI chat
```

Requires a CUDA-capable GPU with `device_map="auto"` and bfloat16 support. The model (~4GB+) downloads on first run.

## Architecture

- `mg_chat.py` — Single-script inference pipeline: loads model/processor, accepts image+text input, generates medical analysis via chat template format
- `mg_tui.py` — Textual TUI chat interface: multi-turn conversation, background model loading/generation, image attachment via `/image <path>`, `/clear` to reset
- Uses HuggingFace's `AutoProcessor.apply_chat_template` for message formatting with `{"role": "user", "content": [...]}` structure
- Images are passed inline in the message content array as `{"type": "image", "image": <PIL.Image>}`
