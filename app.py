from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import torch
import timm
import cv2
import numpy as np
from retinaface import RetinaFace
import sys
import os
import io

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import MODEL_SAVE_PATH, IMG_SIZE
from src.dataset import get_transforms

app = FastAPI(title="DeepDetect 2025 API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

DEVICE = "cpu" # Inference is fine on CPU
MODEL_NAME = "tf_efficientnetv2_b0"
model = None

@app.on_event("startup")
def load_model():
    global model
    print("Loading model weights...")
    try:
        model = timm.create_model(MODEL_NAME, pretrained=False, num_classes=1)
        model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=DEVICE))
        model.to(DEVICE)
        model.eval()
        print("Model loaded successfully!")
    except Exception as e:
        print(f"CRITICAL ERROR: Could not load model from {MODEL_SAVE_PATH}")
        print(f"Details: {e}")

infer_transform = get_transforms(IMG_SIZE, train=False)

def crop_face(img_array):
    """Detects and crops the largest face with a margin."""
    resp = RetinaFace.detect_faces(img_array)
    if not isinstance(resp, dict):
        return None
    
    best_face = None
    max_area = 0
    for key in resp.keys():
        area = resp[key]["facial_area"]
        width = area[2] - area[0]
        height = area[3] - area[1]
        if width * height > max_area:
            max_area = width * height
            best_face = area
            
    if best_face is None:
        return None

    # Crop with margin
    x1, y1, x2, y2 = best_face
    h, w, _ = img_array.shape
    margin = int((x2 - x1) * 0.2)
    x1 = max(0, x1 - margin)
    y1 = max(0, y1 - margin)
    x2 = min(w, x2 + margin)
    y2 = min(h, y2 + margin)
    
    return img_array[y1:y2, x1:x2]

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    # 1. Read Image
    if file.content_type not in ["image/jpeg", "image/png", "image/webp"]:
        raise HTTPException(status_code=400, detail="Invalid file type. Use JPG/PNG.")
    
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if img is None:
        raise HTTPException(status_code=400, detail="Could not decode image")

    # 2. Face Detection & Crop
    face = crop_face(img)
    if face is None:
        return {
            "is_ai_generated": False,
            "confidence_score": 0,
            "label": "No Face Detected"
        }
    
    # 3. Preprocess
    face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
    augmented = infer_transform(image=face_rgb)
    img_tensor = augmented['image'].unsqueeze(0).to(DEVICE)
    
    # 4. Inference
    with torch.no_grad():
        output = model(img_tensor)
        probability = torch.sigmoid(output).item()
        
    # 5. Result
    # 0 = Real, 1 = Fake
    is_fake = probability > 0.5
    confidence = probability if is_fake else 1 - probability
    
    return {
        "is_ai_generated": is_fake,
        "confidence_score": round(confidence * 100, 2),
        "label": "AI Generated" if is_fake else "Real Person"
    }