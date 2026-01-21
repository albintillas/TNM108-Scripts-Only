import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import timm
from tqdm import tqdm
import time
import json
import csv
from datetime import datetime, timezone
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import PROCESSED_DATA_DIR, MODEL_SAVE_PATH, BATCH_SIZE, IMG_SIZE, MODEL_NAME
from src.dataset import DeepFakeDataset, get_transforms

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EPOCHS = 10 

def get_optimal_lr(model_name):
    """
    Returns the best Learning Rate based on the model architecture.
    """
    if "swin" in model_name or "vit" in model_name:
        print(f"Detected Transformer ({model_name}). Using Lower LR: 0.0001")
        return 0.0001
    elif "convnext" in model_name:
        print(f"Detected ConvNeXt ({model_name}). Using Fine-tune LR: 0.0003")
        return 0.0003
    else:
        print(f"Detected CNN ({model_name}). Using Standard LR: 0.001")
        return 0.001

def train():
    run_start = time.perf_counter()
    run_started_at = datetime.now(timezone.utc).isoformat()

    print(f"Starting training on device: {DEVICE}")
    print(f"Model Architecture: {MODEL_NAME}")
    print(f"Saving to: {MODEL_SAVE_PATH}")
    
    # 1. Dataset
    train_ds = DeepFakeDataset(PROCESSED_DATA_DIR, split="train", transform=get_transforms(IMG_SIZE, train=True))
    test_ds = DeepFakeDataset(PROCESSED_DATA_DIR, split="test", transform=get_transforms(IMG_SIZE, train=False))
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

    # 2. Model
    print(f"Loading Model Weights...")
    try:
        model = timm.create_model(MODEL_NAME, pretrained=True, num_classes=1)
    except Exception as e:
        print(f"Error loading model {MODEL_NAME}. Is the name correct in config.py?")
        raise e
        
    model = model.to(DEVICE)

    num_params = sum(p.numel() for p in model.parameters())
    num_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # 3. Setup Optimizer with AUTO-LR
    current_lr = get_optimal_lr(MODEL_NAME)
    
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=current_lr)
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=2)
    
    best_acc = 0.0
    best_epoch = -1

    history = []

    # 4. Training Loop
    for epoch in range(EPOCHS):
        epoch_start = time.perf_counter()

        model.train()
        train_loss = 0
        
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        for imgs, labels in loop:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE).unsqueeze(1)
            
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            loop.set_postfix(loss=f"{loss.item():.4f}")
        
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for imgs, labels in test_loader:
                imgs, labels = imgs.to(DEVICE), labels.to(DEVICE).unsqueeze(1)
                outputs = model(imgs)
                predicted = (torch.sigmoid(outputs) > 0.5).float()
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        acc = 100 * correct / total
        avg_train_loss = train_loss / max(1, len(train_loader))

        current_lr = optimizer.param_groups[0].get("lr", None)
        epoch_seconds = time.perf_counter() - epoch_start

        print(f"Validation Accuracy: {acc:.2f}%")
        print(f"Epoch Time: {epoch_seconds:.1f}s | Avg Train Loss: {avg_train_loss:.4f} | LR: {current_lr}")

        history.append({
            "epoch": epoch + 1,
            "train_loss_avg": float(avg_train_loss),
            "val_accuracy": float(acc),
            "lr": float(current_lr) if current_lr is not None else None,
            "epoch_seconds": float(epoch_seconds),
        })
        
        if acc > best_acc:
            best_acc = acc
            best_epoch = epoch + 1
            MODEL_SAVE_PATH.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"Saved NEW Record to: {MODEL_SAVE_PATH.name}")
            
        scheduler.step(acc)

    run_seconds = time.perf_counter() - run_start
    MODEL_SAVE_PATH.parent.mkdir(parents=True, exist_ok=True)

    # 1) JSON log per run
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    json_log_path = MODEL_SAVE_PATH.parent / f"trainlog_{MODEL_NAME}_{timestamp}.json"

    run_log = {
        "started_at_utc": run_started_at,
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_seconds": float(run_seconds),
        "device": DEVICE,
        "model_name": MODEL_NAME,
        "model_save_path": str(MODEL_SAVE_PATH),
        "epochs": int(EPOCHS),
        "batch_size": int(BATCH_SIZE),
        "img_size": int(IMG_SIZE),
        "train_dir": str(PROCESSED_DATA_DIR / 'train'),
        "test_dir": str(PROCESSED_DATA_DIR / 'test'),
        "train_samples": int(len(train_ds)),
        "test_samples": int(len(test_ds)),
        "num_params": int(num_params),
        "num_trainable_params": int(num_trainable_params),
        "best_val_accuracy": float(best_acc),
        "best_epoch": int(best_epoch),
        "history": history,
    }

    with open(json_log_path, "w", encoding="utf-8") as f:
        json.dump(run_log, f, indent=2)

    print(f"Saved training log: {json_log_path}")

    # 2) Append summary CSV for easy comparisons
    csv_path = MODEL_SAVE_PATH.parent / "training_runs.csv"
    csv_header = [
        "timestamp_utc",
        "model_name",
        "best_val_accuracy",
        "best_epoch",
        "run_seconds",
        "device",
        "epochs",
        "batch_size",
        "img_size",
        "train_samples",
        "test_samples",
        "num_params",
        "weights_path",
        "json_log_path",
    ]
    csv_row = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "model_name": MODEL_NAME,
        "best_val_accuracy": f"{best_acc:.4f}",
        "best_epoch": best_epoch,
        "run_seconds": f"{run_seconds:.2f}",
        "device": DEVICE,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "img_size": IMG_SIZE,
        "train_samples": len(train_ds),
        "test_samples": len(test_ds),
        "num_params": num_params,
        "weights_path": str(MODEL_SAVE_PATH),
        "json_log_path": str(json_log_path),
    }

    write_header = not csv_path.exists()
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=csv_header)
        if write_header:
            writer.writeheader()
        writer.writerow(csv_row)

    print(f"Appended run summary: {csv_path}")

if __name__ == "__main__":
    train()