import torch
from torch.utils.data import Dataset
from pathlib import Path
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import numpy as np

class DeepFakeDataset(Dataset):
    def __init__(self, root_dir, split="train", transform=None):
        self.root_dir = Path(root_dir) / split
        self.transform = transform
        
        self.real_paths = list((self.root_dir / "REAL").glob("*"))
        self.fake_paths = list((self.root_dir / "FAKE").glob("*"))
        
        valid_exts = {".jpg", ".jpeg", ".png", ".webp"}
        self.real_paths = [p for p in self.real_paths if p.suffix.lower() in valid_exts]
        self.fake_paths = [p for p in self.fake_paths if p.suffix.lower() in valid_exts]

        self.all_paths = self.real_paths + self.fake_paths
        self.labels = [0] * len(self.real_paths) + [1] * len(self.fake_paths)
        
        print(f"Loaded {split} set: {len(self.real_paths)} Real, {len(self.fake_paths)} Fake")

    def __len__(self):
        return len(self.all_paths)

    def __getitem__(self, idx):
        img_path = str(self.all_paths[idx])
        label = self.labels[idx]

        image = cv2.imread(img_path)
        if image is None:
            return torch.zeros((3, 224, 224)), torch.tensor(label, dtype=torch.float32)
            
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']

        return image, torch.tensor(label, dtype=torch.float32)

def get_transforms(img_size=224, train=True):
    if train:
        return A.Compose([
            A.Resize(img_size, img_size),
            A.HorizontalFlip(p=0.5),
            A.ImageCompression(quality_lower=60, quality_upper=100, p=0.5),
            A.GaussianBlur(blur_limit=(3, 7), p=0.2),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
    else:
        return A.Compose([
            A.Resize(img_size, img_size),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])