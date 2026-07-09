"""
FastAPI backend for the AI-Assisted Chest X-Ray Triage System.

PIPELINE PER REQUEST:
  1. Receive uploaded X-ray image
  2. Preprocess (resize to 224x224, same transform as ConvNeXt training/eval)
  3. Run through ConvNeXt-Tiny -> raw logits
  4. Apply temperature scaling (T=0.5472) -> calibrated confidence per finding
  5. Run triage_logic.triage_case() -> per-class flags + overall urgency tier
  6. For each FLAGGED finding, generate a GradCAM heatmap (only flagged ones,
     to keep response time reasonable — no point generating 14 heatmaps
     per request when most findings are negative)
  7. Return everything as JSON (heatmaps as base64-encoded PNG strings)

RUN WITH:
  cd D:\\cxr-triage
  uvicorn app:app --host 0.0.0.0 --port 8000 --reload

TEST WITH (once running):
  http://localhost:8000/docs   <- interactive Swagger UI, upload a test image directly
"""

import sys
import io
import base64

sys.path.append('D:/cxr-triage')

import torch
import numpy as np
import cv2
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src.models.convnext import ConvNeXtModel
from src.data.transforms import get_val_transforms
from src.inference.gradcam import GradCAM

from triage_logic import (
    triage_case, load_thresholds, apply_calibration,
    LABELS, LABEL_TO_IDX, GLOBAL_TEMPERATURE, CALIBRATION_METHOD
)

# ─── Config — must match training/eval exactly ──────────────────────────
CHECKPOINT_PATH = 'D:/cxr-triage/checkpoints/convnext_focal_fixed/best_model.pth'
IMAGE_SIZE = 224
USE_CLAHE = False
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

app = FastAPI(title="Chest X-Ray Triage API", version="1.0")

# Allow the Streamlit UI (different port) to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten this to your actual UI origin before real deployment
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Loaded once at startup, reused across requests ─────────────────────
_model = None
_transform = None
_gradcam = None
_thresholds = None


@app.on_event("startup")
def load_everything():
    global _model, _transform, _gradcam, _thresholds

    print(f"Loading ConvNeXt checkpoint from {CHECKPOINT_PATH} on {DEVICE}...")
    _model = ConvNeXtModel(num_classes=14, pretrained=False).to(DEVICE)
    ckpt = torch.load(CHECKPOINT_PATH, map_location=DEVICE, weights_only=False)
    _model.load_state_dict(ckpt['model_state_dict'])
    _model.eval()

    _transform = get_val_transforms(image_size=IMAGE_SIZE, use_clahe=USE_CLAHE)
    _gradcam = GradCAM(_model)
    _thresholds = load_thresholds()

    print("Model, transforms, GradCAM, and thresholds loaded. API ready.")


def generate_heatmap_b64(img_tensor, class_idx, original_pil_image, label, confidence):
    """
    Runs GradCAM for one class, overlays the heatmap, draws a bounding box
    around the hottest region, and labels the finding name + confidence
    directly on the image — so it's readable on its own, not just a color
    wash that only makes sense next to separate text.
    """
    heatmap = _gradcam.generate(img_tensor.clone(), class_idx=class_idx)

    display_size = 320
    orig_resized = original_pil_image.resize((display_size, display_size))
    orig_np = np.array(orig_resized.convert('RGB'))

    heatmap_resized = cv2.resize(heatmap.astype(np.float32), (display_size, display_size))
    hmax = heatmap_resized.max()
    heatmap_norm = heatmap_resized / hmax if hmax > 0 else heatmap_resized

    heatmap_uint8 = np.uint8(255 * heatmap_norm)
    colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    colored = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
    overlay = cv2.addWeighted(orig_np, 0.6, colored, 0.4, 0)
    overlay = np.ascontiguousarray(overlay)

    # --- Draw a bounding box around the hottest region (top 20% activation) ---
    thresh_val = np.percentile(heatmap_norm, 80)
    binary_mask = np.uint8(heatmap_norm >= thresh_val) * 255
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        # Use the largest contour — the single most concentrated hot region
        largest = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest)
        # Only draw if the box is a meaningful size (skip tiny noise specks)
        if w * h > (display_size * display_size) * 0.01:
            cv2.rectangle(overlay, (x, y), (x + w, y + h), (255, 255, 0), 2)

    # --- Label: finding name + confidence, drawn directly on the image ---
    label_text = f"{label}: {confidence*100:.0f}%"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    thickness = 2
    (text_w, text_h), baseline = cv2.getTextSize(label_text, font, font_scale, thickness)

    # Background bar behind the text for readability regardless of image content
    cv2.rectangle(overlay, (0, 0), (text_w + 16, text_h + baseline + 14), (0, 0, 0), -1)
    cv2.putText(overlay, label_text, (8, text_h + 8), font, font_scale, (255, 255, 0), thickness)

    overlay_img = Image.fromarray(overlay)
    buf = io.BytesIO()
    overlay_img.save(buf, format='PNG')
    return base64.b64encode(buf.getvalue()).decode('utf-8')


@app.get("/health")
def health_check():
    return {
        "status": "ok" if _model is not None else "model not loaded",
        "device": str(DEVICE),
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...), include_heatmaps: bool = True):
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet — server still starting up.")

    # Basic validation
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail=f"Expected an image file, got {file.content_type}")

    try:
        contents = await file.read()
        img = Image.open(io.BytesIO(contents)).convert('RGB')
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not read image: {e}")

    # Preprocess + inference
    img_tensor = _transform(img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        logits = _model(img_tensor).cpu().numpy()[0]

    # Triage decision (calibration + thresholds + urgency happen inside triage_case)
    result = triage_case(logits, _thresholds)

    # Generate heatmaps only for flagged findings (keeps response time reasonable)
    heatmaps = {}
    if include_heatmaps and result['all_flagged_findings']:
        detail_by_label = {f['label']: f for f in result['per_finding_detail']}
        for label in result['all_flagged_findings']:
            class_idx = LABEL_TO_IDX[label]
            confidence = detail_by_label[label]['calibrated_confidence']
            heatmaps[label] = generate_heatmap_b64(img_tensor, class_idx, img, label, confidence)

    return JSONResponse({
        "case_tier": result['case_tier'],
        "driving_findings": result['driving_findings'],
        "all_flagged_findings": result['all_flagged_findings'],
        "per_finding_detail": result['per_finding_detail'],
        "heatmaps_base64": heatmaps,  # {finding_label: base64 PNG string}
        "model_info": {
            "model": "ConvNeXt-Tiny (Focal loss)",
            "calibration_method": CALIBRATION_METHOD,
            "temperature": GLOBAL_TEMPERATURE if CALIBRATION_METHOD == 'global_temperature' else None,
            "image_size": IMAGE_SIZE,
        }
    })


if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)