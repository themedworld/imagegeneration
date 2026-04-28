import os
import uuid
import base64
from io import BytesIO

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from gradio_client import Client
from huggingface_hub import InferenceClient
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware

# =========================
# LOAD ENV
# =========================
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
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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
# BASE64 HELPER
# =========================

def image_to_base64(image):
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")

# =========================
# API
# =========================

@app.post("/generate-image")
def generate_image(data: ImageRequest):

    try:
        final_prompt = data.prompt
        enriched_prompt = None

        # =====================
        # ENRICH PROMPT
        # =====================
        if data.mode.lower() == "enriched":

            enriched_prompt = prompt_client.predict(
                data.prompt, 
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

        # =====================
        # CONVERT IMAGE TO BASE64
        # =====================
        image_base64 = image_to_base64(image)

        return {
            "success": True,
            "mode": data.mode,
            "original_prompt": data.prompt,
            "used_prompt": final_prompt,
            "enriched_prompt": enriched_prompt,
            "image_base64": image_base64
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
