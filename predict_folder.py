import torch
import timm
import cv2
import numpy as np
import sys
import os
import argparse
from facenet_pytorch import MTCNN
from pathlib import Path
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import MODEL_SAVE_PATH, IMG_SIZE, MODEL_NAME
from src.dataset import get_transforms

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def load_model():
    print(f"Loading architecture: {MODEL_NAME}...")
    model = timm.create_model(MODEL_NAME, pretrained=False, num_classes=1)
    
    try:
        model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=DEVICE))
    except FileNotFoundError:
        print(f"ERROR: Model file not found at {MODEL_SAVE_PATH}")
        sys.exit(1)
        
    model.to(DEVICE)
    model.eval()
    return model

def get_face_detector():
    return MTCNN(keep_all=False, select_largest=True, margin=0, device=DEVICE, post_process=False)

def predict_folder(folder_path):
    # 1. Setup
    folder = Path(folder_path)
    if not folder.exists():
        print(f"Error: Folder not found at {folder_path}")
        return

    # Collect images
    extensions = {"*.jpg", "*.jpeg", "*.png", "*.webp"}
    files = []
    for ext in extensions:
        files.extend(list(folder.rglob(ext)))
    
    if len(files) == 0:
        print("No images found in folder.")
        return

    print(f"Found {len(files)} images in {folder.name}")
    
    model = load_model()
    mtcnn = get_face_detector()
    transform = get_transforms(IMG_SIZE, train=False)

    ai_count = 0
    real_count = 0
    errors = 0
    results = []

    print("\nStarting Analysis...")
    print("-" * 50)

    # 2. Process Loop
    for file_path in tqdm(files, desc="Scanning"):
        try:
            img = cv2.imread(str(file_path))
            if img is None:
                errors += 1
                continue

            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            h, w, _ = img_rgb.shape

            boxes, _ = mtcnn.detect(img_rgb)
            
            if boxes is None:
                continue

            # Crop Face
            box = boxes[0]
            x1, y1, x2, y2 = [int(b) for b in box]

            # Margin
            margin_x = int((x2 - x1) * 0.2)
            margin_y = int((y2 - y1) * 0.2)
            x1 = max(0, x1 - margin_x)
            y1 = max(0, y1 - margin_y)
            x2 = min(w, x2 + margin_x)
            y2 = min(h, y2 + margin_y)

            face_crop = img_rgb[y1:y2, x1:x2]
            
            if face_crop.size == 0:
                continue

            augmented = transform(image=face_crop)
            img_tensor = augmented['image'].unsqueeze(0).to(DEVICE)

            with torch.no_grad():
                output = model(img_tensor)
                probability = torch.sigmoid(output).item()

            is_fake = probability > 0.5
            confidence = probability if is_fake else 1 - probability
            
            label = "AI" if is_fake else "REAL"
            
            if is_fake:
                ai_count += 1
            else:
                real_count += 1
                
            results.append((file_path.name, label, confidence))

        except Exception as e:
            errors += 1

    print("\n" + "=" * 30)
    print("       FINAL REPORT       ")
    print("=" * 30)
    print(f"Total Scanned: {len(files)}")
    print(f"Real Faces:    \033[92m{real_count}\033[0m")
    print(f"AI Generated:  \033[91m{ai_count}\033[0m")
    print(f"Errors/No Face: {errors + (len(files) - len(results))}")
    print("=" * 30)
    
    print("\nLast 5 Detections:")
    for name, label, conf in results[-5:]:
        color = "\033[91m" if label == "AI" else "\033[92m"
        print(f"{name}: {color}{label}\033[0m ({conf*100:.1f}%)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DeepDetect Folder Scanner")
    parser.add_argument("folder", help="Path to the folder containing images")
    
    args = parser.parse_args()
    predict_folder(args.folder)