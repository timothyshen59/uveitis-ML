#scripts/dataset.py 
import os
import torch
import cv2 
import numpy as np

from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
from PIL import Image, ImageOps
from pathlib import Path 

class FundusDataset(Dataset):
    DEFAULT_TRANSFORM = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])

    def __init__(self, df, img_dir, transform=None):
        self.df        = df.reset_index(drop=True)
        self.image_dir   = Path(img_dir)
        self.transform = transform or self.DEFAULT_TRANSFORM
        
        self._image_cache = {}
        self._mask_cache = {}
        ########
 
            
        def find_image(image_path):
            base = (Path(img_dir) / image_path.replace("\\", "/")).with_suffix("")
            has_mask = (Path(str(base) + "_masks.npy")).exists()
            for ext in (".jpg", ".png", ".jpeg"):
                if (Path(str(base) + ext)).exists() and has_mask:
                    return True
            return False

        mask = df["Image_File(FA)"].apply(find_image)
        dropped = (~mask).sum()
        if dropped:
            print(f"[dataset] Dropping {dropped} rows with missing images.")
        self.df = df[mask].reset_index(drop=True)
        ########
        
        
    def __len__(self):
        return len(self.df) * 10

    def __getitem__(self, idx):
        row_idx  = idx // 10
        zone_num = idx % 10 + 1

        row     = self.df.iloc[row_idx]
        img_rel = row["Image_File(FA)"].replace("\\", "/")
        img_path  = str(self.image_dir / img_rel)
        mask_path = img_path.replace(".png", "_masks.npy")

        # Load from cache or disk on first access
        if img_rel not in self._image_cache:
            img = cv2.imread(img_path)
            if img is None:
                raise FileNotFoundError(f"Could not read image: {img_path}")
            self._image_cache[img_rel] = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if img_rel not in self._mask_cache:
            self._mask_cache[img_rel] = np.load(mask_path)

        image        = self._image_cache[img_rel].copy()
        labeled_mask = self._mask_cache[img_rel]

        mask  = (labeled_mask == zone_num).astype(np.uint8)
        image = image * mask[..., np.newaxis]

        image = Image.fromarray(image)
        if self.transform:
            image = self.transform(image)

        label = torch.tensor(row[f"Zone{zone_num}_label"], dtype=torch.long)
        return image, label, zone_num


def make_loader(dataset, batch_size, num_workers, shuffle):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
    )