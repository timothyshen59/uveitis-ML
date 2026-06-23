# scripts/dataset.py

import os
import re
import hashlib
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


def apply_clahe(img):
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)


def crop_zone_bbox(image, labeled_mask, zone_num, pad=10, outside_scale=0.0):
    zone_mask = labeled_mask == zone_num
    ys, xs = np.where(zone_mask)

    if len(xs) == 0 or len(ys) == 0:
        return None

    y1, y2 = ys.min(), ys.max()
    x1, x2 = xs.min(), xs.max()

    h, w = labeled_mask.shape[:2]

    y1 = max(y1 - pad, 0)
    y2 = min(y2 + pad, h - 1)
    x1 = max(x1 - pad, 0)
    x2 = min(x2 + pad, w - 1)

    crop = image[y1:y2 + 1, x1:x2 + 1].copy()
    crop_mask = zone_mask[y1:y2 + 1, x1:x2 + 1]

    if outside_scale <= 0.0:
        crop[~crop_mask] = 0
    elif outside_scale < 1.0:
        crop[~crop_mask] = (crop[~crop_mask] * outside_scale).astype(np.uint8)

    return crop


def pad_to_square(img, fill=0):
    h, w = img.shape[:2]
    size = max(h, w)

    canvas = np.full((size, size, 3), fill, dtype=img.dtype)

    y = (size - h) // 2
    x = (size - w) // 2

    canvas[y:y + h, x:x + w] = img
    return canvas


class FundusDataset(Dataset):
    DEFAULT_TRANSFORM = transforms.Compose([
        transforms.Resize((384, 384)),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225],
        ),
    ])

    def __init__(
        self,
        df,
        img_dir,
        transform=None,
        include_zone_crops=True,
        cache_dir=None,
        use_cache=True,
        cache_version="v1",
        use_clahe=True,
        zone_nums=None,
        zone_pad=10,
        outside_scale=0.0,
        pad_zone_to_square=True,
        label_dtype=torch.float32,
    ):
        self.df = df.reset_index(drop=True)
        self.image_dir = Path(img_dir)
        self.transform = transform or self.DEFAULT_TRANSFORM
        self.include_zone_crops = include_zone_crops

        self.use_cache = use_cache
        self.cache_dir = Path(cache_dir) if cache_dir is not None else None
        self.cache_version = cache_version

        self.use_clahe = use_clahe
        self.zone_nums = list(zone_nums) if zone_nums is not None else list(range(1, 11))
        self.zone_pad = zone_pad
        self.outside_scale = outside_scale
        self.pad_zone_to_square = pad_zone_to_square
        self.label_dtype = label_dtype

        if self.use_cache and self.cache_dir is None:
            self.cache_dir = self.image_dir / "_tensor_cache"

        if self.use_cache:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

        def find_image(image_path):
            img_rel = self._clean_rel_path(image_path)
            base = (self.image_dir / img_rel).with_suffix("")
            has_mask = Path(str(base) + "_masks.npy").exists()

            for ext in (".jpg", ".png", ".jpeg"):
                if Path(str(base) + ext).exists() and has_mask:
                    return True

            return False

        keep_mask = self.df["UWFFA"].apply(find_image)
        dropped = (~keep_mask).sum()

        if dropped:
            print(f"[dataset] Dropping {dropped} rows with missing images/masks.")

        print(
            f"[dataset] use_clahe={self.use_clahe}, "
            f"zones={self.zone_nums}, "
            f"zone_pad={self.zone_pad}, "
            f"outside_scale={self.outside_scale}, "
            f"use_cache={self.use_cache}"
        )

        self.df = self.df[keep_mask].reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def _clean_rel_path(self, image_path):
        img_rel = str(image_path).replace("\\", "/")
        img_rel = re.sub(r"^.*?(Patient)", r"\1", img_rel)
        return img_rel

    def _resolve_paths(self, img_rel):
        base = str((self.image_dir / img_rel).with_suffix(""))

        img_path = None
        for ext in (".jpg", ".png", ".jpeg"):
            candidate = base + ext
            if os.path.exists(candidate):
                img_path = candidate
                break

        if img_path is None:
            raise FileNotFoundError(f"No image found for: {base}")

        mask_path = base + "_masks.npy"

        if not os.path.exists(mask_path):
            raise FileNotFoundError(f"No mask found for: {mask_path}")

        return img_path, mask_path

    def _cache_path(self, img_rel):
        key = {
            "img_rel": img_rel,
            "cache_version": self.cache_version,
            "include_zone_crops": self.include_zone_crops,
            "use_clahe": self.use_clahe,
            "zone_nums": self.zone_nums,
            "zone_pad": self.zone_pad,
            "outside_scale": self.outside_scale,
            "pad_zone_to_square": self.pad_zone_to_square,
        }

        key_str = repr(key).encode("utf-8")
        digest = hashlib.md5(key_str).hexdigest()

        return self.cache_dir / f"{digest}.pt"

    def _torch_load(self, path):
        try:
            return torch.load(path, map_location="cpu", weights_only=False)
        except TypeError:
            return torch.load(path, map_location="cpu")

    def _atomic_save(self, obj, path):
        tmp_path = path.with_suffix(f".tmp.{os.getpid()}.pt")
        torch.save(obj, tmp_path)
        os.replace(tmp_path, path)

    def _load_raw_image_and_mask(self, img_path, mask_path):
        image = cv2.imread(img_path)

        if image is None:
            raise FileNotFoundError(f"Could not read image: {img_path}")

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        labeled_mask = np.load(mask_path)

        if image.shape[:2] != labeled_mask.shape[:2]:
            raise ValueError(
                f"Image/mask shape mismatch: image={image.shape[:2]}, "
                f"mask={labeled_mask.shape[:2]}, path={img_path}"
            )

        if self.use_clahe:
            image = apply_clahe(image)

        return image, labeled_mask

    def _make_full_image_tensor(self, image):
        full_image = Image.fromarray(image)

        if self.transform:
            full_image = self.transform(full_image)

        return full_image

    def _make_zone_tensors(self, image, labeled_mask):
        zone_imgs = []

        for zone_num in self.zone_nums:
            zone_crop = crop_zone_bbox(
                image=image,
                labeled_mask=labeled_mask,
                zone_num=zone_num,
                pad=self.zone_pad,
                outside_scale=self.outside_scale,
            )

            if zone_crop is None:
                zone_crop = np.zeros((32, 32, 3), dtype=np.uint8)

            if self.pad_zone_to_square:
                zone_crop = pad_to_square(zone_crop, fill=0)

            zone_crop = Image.fromarray(zone_crop)

            if self.transform:
                zone_crop = self.transform(zone_crop)

            zone_imgs.append(zone_crop)

        return torch.stack(zone_imgs, dim=0)

    def _make_labels(self, row):
        return torch.tensor(
            [int(row[f"Zone{z}_label"] > 0) for z in self.zone_nums],
            dtype=self.label_dtype,
        )

    def _build_sample(self, row, img_rel, img_path, mask_path):
        image, labeled_mask = self._load_raw_image_and_mask(img_path, mask_path)

        full_image = self._make_full_image_tensor(image)
        zone_labels = self._make_labels(row)

        if not self.include_zone_crops:
            return {
                "full_image": full_image.cpu(),
                "zone_labels": zone_labels.cpu(),
                "img_path": img_path,
            }

        zone_imgs = self._make_zone_tensors(image, labeled_mask)

        return {
            "full_image": full_image.cpu(),
            "zone_imgs": zone_imgs.cpu(),
            "zone_labels": zone_labels.cpu(),
            "img_path": img_path,
        }

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        img_rel = self._clean_rel_path(row["UWFFA"])
        img_path, mask_path = self._resolve_paths(img_rel)

        if self.use_cache:
            cache_path = self._cache_path(img_rel)

            if cache_path.exists():
                sample = self._torch_load(cache_path)
            else:
                sample = self._build_sample(row, img_rel, img_path, mask_path)
                self._atomic_save(sample, cache_path)
        else:
            sample = self._build_sample(row, img_rel, img_path, mask_path)

        if not self.include_zone_crops:
            return sample["full_image"], sample["zone_labels"]

        return (
            sample["full_image"],
            sample["zone_imgs"],
            sample["zone_labels"],
            sample["img_path"],
        )


def make_loader(dataset, batch_size, num_workers, shuffle):
    kwargs = {
        "dataset": dataset,
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": num_workers,
        "pin_memory": True,
    }

    if num_workers > 0:
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = 2

    return DataLoader(**kwargs)