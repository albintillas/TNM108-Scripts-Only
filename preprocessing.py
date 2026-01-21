import os
import sys
import cv2
from retinaface import RetinaFace
from tqdm import tqdm
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import RAW_DATA_DIR, PROCESSED_DATA_DIR

CONFIDENCE_THRESHOLD = 0.9

def crop_faces(image_path, save_path):
    try:
        img = cv2.imread(str(image_path))
        if img is None:
            return

        # Detect faces
        faces = RetinaFace.detect_faces(img)

        if not isinstance(faces, dict):
            return 

        best_face = None
        max_area = 0

        for key in faces.keys():
            identity = faces[key]
            if identity["score"] < CONFIDENCE_THRESHOLD:
                continue
            
            area = identity["facial_area"]
            width = area[2] - area[0]
            height = area[3] - area[1]
            current_area = width * height
            
            if current_area > max_area:
                max_area = current_area
                best_face = identity["facial_area"]

        if best_face is not None:
            x1, y1, x2, y2 = best_face
            h, w, _ = img.shape
            
            margin_x = int((x2 - x1) * 0.2)
            margin_y = int((y2 - y1) * 0.2)
            
            x1 = max(0, x1 - margin_x)
            y1 = max(0, y1 - margin_y)
            x2 = min(w, x2 + margin_x)
            y2 = min(h, y2 + margin_y)
            
            face_crop = img[y1:y2, x1:x2]
            
            os.makedirs(save_path.parent, exist_ok=True)
            cv2.imwrite(str(save_path), face_crop)
            
    except Exception as e:
        print(f"Error processing {image_path}: {e}")

def process_dataset():
    print(f"Reading from: {RAW_DATA_DIR}")
    print(f"Saving to:   {PROCESSED_DATA_DIR}")
    
    files = list(RAW_DATA_DIR.rglob("*.jpg")) + list(RAW_DATA_DIR.rglob("*.png")) + list(RAW_DATA_DIR.rglob("*.jpeg"))
    
    if len(files) == 0:
        print("ERROR: No images found! Check your RAW_DATA_DIR in config.py")
        return

    print(f"Found {len(files)} images. Starting face extraction...")
    
    for file_path in tqdm(files):
        relative_path = file_path.relative_to(RAW_DATA_DIR)
        save_path = PROCESSED_DATA_DIR / relative_path
        
        if not save_path.exists():
            crop_faces(file_path, save_path)

if __name__ == "__main__":
    process_dataset()