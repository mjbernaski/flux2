from fastapi import FastAPI
from pydantic import BaseModel
from PIL import Image
import base64
import io
import torch

MODEL_ID = "google/medgemma-1.5-4b-it"

app = FastAPI()

model = None
processor = None


class ChatRequest(BaseModel):
    messages: list[dict]
    image: str | None = None


class ChatResponse(BaseModel):
    response: str


def load_model():
    global model, processor
    from transformers import AutoProcessor, AutoModelForImageTextToText

    print("Loading MedGemma model...")
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    print("Model loaded.")


@app.on_event("startup")
async def startup():
    load_model()


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    # Build conversation in the format the model expects
    conversation = []
    pil_image = None

    # Decode base64 image if provided
    if req.image:
        image_data = base64.b64decode(req.image)
        pil_image = Image.open(io.BytesIO(image_data)).convert("RGB")

    for msg in req.messages:
        role = msg["role"]
        text = msg.get("text", msg.get("content", ""))

        content = []
        # Attach image to the last user message that has one
        if role == "user" and pil_image and msg is req.messages[-1]:
            content.append({"type": "image", "image": pil_image})
            pil_image = None  # only attach once

        content.append({"type": "text", "text": text})
        conversation.append({"role": role, "content": content})

    inputs = processor.apply_chat_template(
        conversation,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device, dtype=torch.bfloat16)

    input_len = inputs["input_ids"].shape[-1]

    with torch.inference_mode():
        generation = model.generate(**inputs, max_new_tokens=2000, do_sample=False)
        generation = generation[0][input_len:]

    decoded = processor.decode(generation, skip_special_tokens=True)
    return ChatResponse(response=decoded)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=5000)
