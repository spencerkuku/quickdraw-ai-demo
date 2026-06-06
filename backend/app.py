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
GRADCAM_OUTPUT_SIZE = 480
GRADCAM_VISIBILITY_THRESHOLD = 0.20
GRADCAM_INK_RADIUS = 3


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


class GradCAMResult(BaseModel):
    image: str
    target_label: str
    target_confidence: float
    layer: str
    method: str = "Ink-guided Grad-CAM"


class PredictionResponse(BaseModel):
    predictions: list[Prediction]
    latency_ms: float
    gradcam: GradCAMResult


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
        return Image.open(io.BytesIO(image_bytes)).convert("L")
    except Exception as exc:
        raise HTTPException(status_code=400, detail="Invalid image data") from exc


def preprocess_image(image: Image.Image) -> Image.Image:
    # The browser sends a white canvas with dark strokes; training used white on black.
    image = ImageOps.invert(image)
    return image.resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.LANCZOS)


def image_to_tensor(image: Image.Image) -> torch.Tensor:
    byte_tensor = torch.frombuffer(bytearray(image.tobytes()), dtype=torch.uint8)
    tensor = byte_tensor.reshape(IMAGE_SIZE, IMAGE_SIZE).float().div(255.0)
    return tensor.unsqueeze(0).unsqueeze(0)


def forward_with_gradcam(
    model: SimpleCNN,
    tensor: torch.Tensor,
    target_layer_index: int = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
    features = tensor
    target_activations = None

    for index, layer in enumerate(model.model.features):
        features = layer(features)
        if index == target_layer_index:
            target_activations = features

    pooled = F.adaptive_avg_pool2d(features, (1, 1))
    output = model.model.classifier(torch.flatten(pooled, 1))

    if target_activations is None:
        raise RuntimeError("Grad-CAM target layer was not reached")

    return output, target_activations


def turbo_color(value: float) -> tuple[int, int, int]:
    # Compact approximation of the Turbo palette: blue -> cyan -> yellow -> red.
    stops = (
        (0.00, (35, 23, 140)),
        (0.25, (32, 144, 204)),
        (0.50, (79, 214, 116)),
        (0.75, (249, 210, 42)),
        (1.00, (180, 4, 38)),
    )

    for (left_value, left_color), (right_value, right_color) in zip(stops, stops[1:]):
        if value <= right_value:
            ratio = (value - left_value) / (right_value - left_value)
            return tuple(
                round(left + (right - left) * ratio)
                for left, right in zip(left_color, right_color)
            )

    return stops[-1][1]


def create_gradcam_image(
    source_image: Image.Image,
    input_tensor: torch.Tensor,
    activations: torch.Tensor,
    class_index: int,
    output: torch.Tensor,
) -> str:
    source_image = source_image.resize(
        (GRADCAM_OUTPUT_SIZE, GRADCAM_OUTPUT_SIZE),
        Image.Resampling.LANCZOS,
    )
    gradients = torch.autograd.grad(
        outputs=output[0, class_index],
        inputs=activations,
        retain_graph=False,
        create_graph=False,
    )[0]

    weights = gradients.mean(dim=(2, 3), keepdim=True)
    cam = torch.relu((weights * activations).sum(dim=1, keepdim=True))
    cam = F.interpolate(
        cam,
        size=source_image.size[::-1],
        mode="bicubic",
        align_corners=False,
    )[0, 0]
    cam = cam.clamp_min(0)

    cam_min = cam.min()
    cam_max = cam.max()
    if (cam_max - cam_min).item() > 1e-8:
        cam = (cam - cam_min) / (cam_max - cam_min)
    else:
        cam = torch.zeros_like(cam)

    # Keep genuine class gradients, but suppress activations far from any ink.
    # The soft neighborhood accounts for convolutional receptive fields and
    # interpolation, so attention may remain near a stroke without being
    # artificially forced onto the exact black pixels.
    ink = (input_tensor > 0.06).float()
    kernel_size = GRADCAM_INK_RADIUS * 2 + 1
    ink_support = F.max_pool2d(
        ink,
        kernel_size=kernel_size,
        stride=1,
        padding=GRADCAM_INK_RADIUS,
    )
    ink_support = F.interpolate(
        ink_support,
        size=source_image.size[::-1],
        mode="bilinear",
        align_corners=False,
    )[0, 0].clamp(0, 1)
    cam = cam * (0.05 + 0.95 * ink_support)

    cam_max = cam.max()
    if cam_max.item() > 1e-8:
        cam = cam / cam_max

    heatmap = Image.new("RGBA", source_image.size)
    heat_pixels = [
        (
            *turbo_color(value),
            0
            if value < GRADCAM_VISIBILITY_THRESHOLD
            else round(
                220
                * (
                    (value - GRADCAM_VISIBILITY_THRESHOLD)
                    / (1 - GRADCAM_VISIBILITY_THRESHOLD)
                )
                ** 0.75
            ),
        )
        for value in cam.detach().cpu().flatten().tolist()
    ]
    heatmap.putdata(heat_pixels)

    base = source_image.convert("RGB").convert("RGBA")
    overlay = Image.alpha_composite(base, heatmap).convert("RGB")
    buffer = io.BytesIO()
    overlay.save(buffer, format="PNG", optimize=True)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


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
    source_image = decode_image(payload.image)
    processed_image = preprocess_image(source_image)
    tensor = image_to_tensor(processed_image)

    with torch.enable_grad():
        output, activations = forward_with_gradcam(app.state.model, tensor)
        probabilities = F.softmax(output, dim=1)
        top_probs, top_indices = torch.topk(
            probabilities,
            k=min(3, len(app.state.labels)),
        )
        target_index = top_indices[0, 0].item()
        gradcam_image = create_gradcam_image(
            source_image=source_image,
            input_tensor=tensor,
            activations=activations,
            class_index=target_index,
            output=output,
        )

    predictions = [
        Prediction(
            label=app.state.labels[index.item()],
            confidence=round(probability.item() * 100, 2),
        )
        for probability, index in zip(top_probs[0], top_indices[0])
    ]
    latency_ms = round((time.perf_counter() - started_at) * 1000, 1)
    return PredictionResponse(
        predictions=predictions,
        latency_ms=latency_ms,
        gradcam=GradCAMResult(
            image=gradcam_image,
            target_label=app.state.labels[target_index],
            target_confidence=round(probabilities[0, target_index].item() * 100, 2),
            layer="MobileNetV2 features[1] (14 × 14)",
        ),
    )
