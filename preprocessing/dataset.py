# scripts/dataset.py

import os
import hashlib
import cv2
import numpy as np
import torch
from PIL import Image
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


def apply_clahe(img):
    """Applies CLAHE contrast enhancement to the L channel of an RGB image."""
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    lab[:, :, 0] = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)


def crop_zone_bbox(image, mask, zone_num, pad=10):
    """Crops the bounding box of a zone, zeroing pixels outside it."""
    ys, xs = np.where(mask == zone_num)
    if len(xs) == 0:
        return None

    h, w = mask.shape
    y1, y2 = max(ys.min() - pad, 0), min(ys.max() + pad, h - 1)
    x1, x2 = max(xs.min() - pad, 0), min(xs.max() + pad, w - 1)

    crop = image[y1:y2+1, x1:x2+1].copy()
    crop[~(mask == zone_num)[y1:y2+1, x1:x2+1]] = 0
    return crop


def pad_to_square(img, fill=0):
    """Centers an image on a square canvas."""
    h, w = img.shape[:2]
    size = max(h, w)
    canvas = np.full((size, size, 3), fill, dtype=img.dtype)
    y, x = (size - h) // 2, (size - w) // 2
    canvas[y:y+h, x:x+w] = img
    return canvas


def normalize_path(image_path):
    """Strips Windows drive/prefix, keeping from 'Patient...' onward."""
    path = str(image_path).replace("\\", "/")
    start = path.find("Patient")
    return path[start:] if start != -1 else path


class FundusDataset(Dataset):
    DEFAULT_TRANSFORM = transforms.Compose([
        transforms.Resize((384, 384)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    def __init__(
        self,
        df,
        img_dir,
        transform=None,
        use_clahe=True,
        zone_nums=None,
        zone_pad=10,
        use_cache=True,
        cache_dir=None,
        cache_version="v1",
        label_dtype=torch.float32,
    ):
        self.image_dir    = Path(img_dir)
        self.transform    = transform or self.DEFAULT_TRANSFORM
        self.use_clahe    = use_clahe
        self.zone_nums    = list(zone_nums) if zone_nums else list(range(1, 11))
        self.zone_pad     = zone_pad
        self.use_cache    = use_cache
        self.cache_dir    = Path(cache_dir) if cache_dir else self.image_dir / "_tensor_cache"
        self.cache_version = cache_version
        self.label_dtype  = label_dtype

        if self.use_cache:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Drop rows where image or mask file is missing
        def has_files(image_path):
            base = str((self.image_dir / normalize_path(image_path)).with_suffix(""))
            has_mask = Path(base + "_masks.npy").exists()
            has_img  = any(Path(base + ext).exists() for ext in (".jpg", ".png", ".jpeg"))
            return has_mask and has_img

        keep = df["UWFFA"].apply(has_files)
        if (~keep).sum():
            print(f"[dataset] Dropping {(~keep).sum()} rows with missing files.")
        self.df = df[keep].reset_index(drop=True)
        print(f"[dataset] {len(self.df)} samples | clahe={use_clahe} | zones={self.zone_nums}")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row     = self.df.iloc[idx]
        img_rel = normalize_path(row["UWFFA"])

        if self.use_cache:
            cache_path = self.cache_dir / f"{hashlib.md5((img_rel + self.cache_version).encode()).hexdigest()}.pt"
            if cache_path.exists():
                s = torch.load(cache_path, map_location="cpu", weights_only=False)
                return s["full_image"], s["zone_imgs"], s["zone_labels"], s["img_path"]

        # Resolve paths
        base = str((self.image_dir / img_rel).with_suffix(""))
        img_path = next((base + ext for ext in (".jpg", ".png", ".jpeg") if os.path.exists(base + ext)), None)
        if img_path is None:
            raise FileNotFoundError(f"No image found: {base}")
        mask_path = base + "_masks.npy"

        # Load image + mask
        image = cv2.imread(img_path)
        if image is None:
            raise FileNotFoundError(f"Could not read: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.use_clahe:
            image = apply_clahe(image)
        labeled_mask = np.load(mask_path)

        # Build full image tensor and per-zone tensors
        full_image = self.transform(Image.fromarray(image))                           # (C, H, W)
        zone_imgs = []
        for z in self.zone_nums:
            crop = crop_zone_bbox(image, labeled_mask, z, self.zone_pad) or np.zeros((32, 32, 3), dtype=np.uint8)
            crop = pad_to_square(crop)
            zone_imgs.append(self.transform(Image.fromarray(crop)))
        zone_imgs   = torch.stack(zone_imgs)                                          # (num_zones, C, H, W)
        zone_labels = torch.tensor(
            [int(row[f"Zone{z}_label"] > 0) for z in self.zone_nums],
            dtype=self.label_dtype,
        )

        sample = {"full_image": full_image, "zone_imgs": zone_imgs, "zone_labels": zone_labels, "img_path": img_path}
        if self.use_cache:
            tmp = cache_path.with_suffix(f".tmp.{os.getpid()}.pt")
            torch.save(sample, tmp)
            os.replace(tmp, cache_path)

        return full_image, zone_imgs, zone_labels, img_path


def make_loader(dataset, batch_size, num_workers, shuffle):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        **({"persistent_workers": True, "prefetch_factor": 2} if num_workers > 0 else {}),
    )