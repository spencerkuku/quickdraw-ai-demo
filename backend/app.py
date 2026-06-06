import base64
import io
import time
from contextlib import asynccontextmanager
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from PIL import Image, ImageOps
from torchvision.models import mobilenet_v2


BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "model.pt"
LABELS_PATH = BASE_DIR / "labels.txt"
IMAGE_SIZE = 28


class SimpleCNN(nn.Module):
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.model = mobilenet_v2(weights=None, num_classes=num_classes)
        self.model.features[0][0] = nn.Conv2d(
            1,
            32,
            kernel_size=3,
            stride=2,
            padding=1,
            bias=False,
        )

    def forward(self, x):
        return self.model(x)


class PredictionRequest(BaseModel):
    image: str


class Prediction(BaseModel):
    label: str
    confidence: float


class PredictionResponse(BaseModel):
    predictions: list[Prediction]
    latency_ms: float


def load_labels() -> list[str]:
    with LABELS_PATH.open("r", encoding="utf-8") as file:
        return [line.strip() for line in file if line.strip()]


def load_model(labels: list[str]) -> SimpleCNN:
    model = SimpleCNN(num_classes=len(labels))
    state_dict = torch.load(MODEL_PATH, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()
    return model


@asynccontextmanager
async def lifespan(app: FastAPI):
    labels = load_labels()
    app.state.labels = labels
    app.state.model = load_model(labels)
    yield


app = FastAPI(
    title="SketchSense API",
    description="Quick Draw style image classification API",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def decode_image(data_url: str) -> Image.Image:
    try:
        encoded = data_url.split(",", 1)[1] if "," in data_url else data_url
        image_bytes = base64.b64decode(encoded, validate=True)
        image = Image.open(io.BytesIO(image_bytes)).convert("L")
    except Exception as exc:
        raise HTTPException(status_code=400, detail="Invalid image data") from exc

    # The browser sends a white canvas with dark strokes; training used white on black.
    image = ImageOps.invert(image)
    return image.resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.LANCZOS)


def image_to_tensor(image: Image.Image) -> torch.Tensor:
    byte_tensor = torch.ByteTensor(torch.ByteStorage.from_buffer(image.tobytes()))
    tensor = byte_tensor.reshape(IMAGE_SIZE, IMAGE_SIZE).float().div(255.0)
    return tensor.unsqueeze(0).unsqueeze(0)


@app.get("/api/health")
def health():
    return {
        "status": "ready",
        "classes": app.state.labels,
        "model": "MobileNetV2",
    }


@app.post("/api/predict", response_model=PredictionResponse)
def predict(payload: PredictionRequest):
    started_at = time.perf_counter()
    image = decode_image(payload.image)
    tensor = image_to_tensor(image)

    with torch.inference_mode():
        output = app.state.model(tensor)
        probabilities = F.softmax(output, dim=1)
        top_probs, top_indices = torch.topk(
            probabilities,
            k=min(3, len(app.state.labels)),
        )

    predictions = [
        Prediction(
            label=app.state.labels[index.item()],
            confidence=round(probability.item() * 100, 2),
        )
        for probability, index in zip(top_probs[0], top_indices[0])
    ]
    latency_ms = round((time.perf_counter() - started_at) * 1000, 1)
    return PredictionResponse(predictions=predictions, latency_ms=latency_ms)
