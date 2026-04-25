# app.py

import os
import uuid
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from gradio_client import Client
from huggingface_hub import InferenceClient

# =========================
# LOAD ENV
# =========================
from dotenv import load_dotenv
load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")

if not HF_TOKEN:
    raise ValueError("HF_TOKEN not found in .env file")

# =========================
# CONFIG
# =========================

SPACE_NAME = "amin1221/enrechirprompt"
IMAGE_MODEL = "black-forest-labs/FLUX.1-schnell"

# =========================
# APP
# =========================

app = FastAPI(title="Prompt To Image API")

os.makedirs("generated", exist_ok=True)
app.mount("/generated", StaticFiles(directory="generated"), name="generated")

# =========================
# CLIENTS
# =========================

prompt_client = Client(SPACE_NAME)

image_client = InferenceClient(
    provider="nscale",
    api_key=HF_TOKEN
)

# =========================
# REQUEST MODEL
# =========================

class ImageRequest(BaseModel):
    prompt: str
    mode: str   # simple / enriched

# =========================
# HELPERS
# =========================

def save_image(image):
    filename = f"{uuid.uuid4()}.png"
    path = f"generated/{filename}"
    image.save(path)
    return path

# =========================
# API
# =========================

@app.post("/generate-image")
def generate_image(data: ImageRequest):

    final_prompt = data.prompt
    enriched_prompt = None

    try:
        if data.mode.lower() == "enriched":

            enriched_prompt = prompt_client.predict(
                prompt=data.prompt,
                api_name="/predict_prompt"
            )

            final_prompt = enriched_prompt

        image = image_client.text_to_image(
            final_prompt,
            model=IMAGE_MODEL
        )

        path = save_image(image)

        return {
            "success": True,
            "mode": data.mode,
            "original_prompt": data.prompt,
            "used_prompt": final_prompt,
            "enriched_prompt": enriched_prompt,
            "image_url": f"http://127.0.0.1:8000/{path}"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
