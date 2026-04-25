# app.py

from fastapi import FastAPI
from pydantic import BaseModel
from gradio_client import Client
from huggingface_hub import InferenceClient
import tempfile
import shutil
import uuid
import os

app = FastAPI(title="Prompt To Image API")

# =========================
# CONFIG
# =========================

HF_TOKEN = "YOUR_HF_TOKEN"

SPACE_NAME = "amin1221/NOM-DE-TON-SPACE"

IMAGE_MODEL = "black-forest-labs/FLUX.1-schnell"

# =========================
# CLIENTS
# =========================

prompt_client = Client(SPACE_NAME)

image_client = InferenceClient(
    provider="nscale",
    api_key=HF_TOKEN
)

# =========================
# STATIC FOLDER
# =========================

os.makedirs("generated", exist_ok=True)

from fastapi.staticfiles import StaticFiles
app.mount("/generated", StaticFiles(directory="generated"), name="generated")

# =========================
# REQUEST MODEL
# =========================

class ImageRequest(BaseModel):
    prompt: str
    mode: str   # simple / enriched


# =========================
# HELPERS
# =========================

def save_image_temp(image):
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

    # =====================
    # ENRICH PROMPT
    # =====================
    if data.mode.lower() == "enriched":

        enriched_prompt = prompt_client.predict(
            prompt=data.prompt,
            api_name="/predict_prompt"
        )

        final_prompt = enriched_prompt

    # =====================
    # GENERATE IMAGE
    # =====================
    image = image_client.text_to_image(
        final_prompt,
        model=IMAGE_MODEL
    )

    path = save_image_temp(image)

    image_url = f"http://127.0.0.1:8000/{path}"

    return {
        "success": True,
        "mode": data.mode,
        "original_prompt": data.prompt,
        "used_prompt": final_prompt,
        "enriched_prompt": enriched_prompt,
        "image_url": image_url
    }
