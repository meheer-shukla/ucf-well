"""AffectNet loader — 3-class valence subset.

The dataset on disk uses a YOLO-format directory:
    YOLO_format/{train,valid,test}/images/ffhq_N.png
    YOLO_format/{train,valid,test}/labels/ffhq_N.txt

Each label file contains one row: <class_id> <x> <y> <w> <h>
where class_id follows the AffectNet 8-class order declared in data.yaml.

We collapse the 8 classes into 3 valence groups:
    positive : Happy
    neutral  : Neutral
    negative : Anger, Contempt, Disgust, Fear, Sad, Surprise
"""
from __future__ import annotations
import os, glob
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T


# AffectNet 8-class index -> valence group index
AFFECTNET_CLASSES = ["Anger", "Contempt", "Disgust", "Fear",
                     "Happy", "Neutral", "Sad", "Surprise"]
CLASS_TO_VALENCE = {0: 2, 1: 2, 2: 2, 3: 2, 4: 0, 5: 1, 6: 2, 7: 2}
VALENCE_NAMES = ["positive", "neutral", "negative"]


class AffectNetValence(Dataset):
    """3-class valence subset of AffectNet."""

    def __init__(self, root: str, split: str = "train",
                 img_size: int = 224, augment: bool = True):
        yolo = os.path.join(root, "YOLO_format")
        img_dir = os.path.join(yolo, split, "images")
        lbl_dir = os.path.join(yolo, split, "labels")

        self.samples = []
        for img_path in sorted(glob.glob(os.path.join(img_dir, "*.png"))):
            base = os.path.splitext(os.path.basename(img_path))[0]
            lbl_path = os.path.join(lbl_dir, base + ".txt")
            if not os.path.exists(lbl_path):
                continue
            with open(lbl_path, "r") as f:
                line = f.readline().strip()
            if not line:
                continue
            cls_id = int(line.split()[0])
            if cls_id not in CLASS_TO_VALENCE:
                continue
            self.samples.append((img_path, CLASS_TO_VALENCE[cls_id]))

        if not self.samples:
            raise RuntimeError(f"no samples found in {img_dir}")

        if augment:
            self.tf = T.Compose([
                T.Resize((img_size, img_size)),
                T.RandomHorizontalFlip(),
                T.RandomAffine(degrees=10, translate=(0.05, 0.05),
                               scale=(0.95, 1.05)),
                T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                T.RandomErasing(p=0.25),
            ])
        else:
            self.tf = T.Compose([
                T.Resize((img_size, img_size)),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        path, label = self.samples[i]
        img = Image.open(path).convert("RGB")
        return self.tf(img), label

    def class_counts(self):
        import numpy as np
        counts = [0, 0, 0]
        for _, y in self.samples:
            counts[y] += 1
        return counts


if __name__ == "__main__":
    for split in ["train", "valid", "test"]:
        try:
            d = AffectNetValence("datasets/affectnet", split=split)
            print(f"{split:6s}: {len(d):5d} images  "
                  f"class counts (pos/neu/neg): {d.class_counts()}")
        except Exception as e:
            print(f"{split:6s}: {e}")