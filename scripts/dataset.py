#scripts/dataset.py 
import os
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
from PIL import Image, ImageOps


class FundusDataset(Dataset):
    DEFAULT_TRANSFORM = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])

    def __init__(self, df, img_dir, transform=None):
        self.df        = df.reset_index(drop=True)
        self.img_dir   = img_dir
        self.transform = transform or self.DEFAULT_TRANSFORM
        ########
        def find_image(img_rel):
            base = os.path.splitext(os.path.normpath(os.path.join(img_dir, img_rel.replace("\\", "/"))))[0]
            for ext in (".jpg", ".png", ".jpeg"):
                if os.path.exists(base + ext):
                    return True
            return False

        mask = df["Image_File(FA)"].apply(find_image)
        dropped = (~mask).sum()
        if dropped:
            print(f"[dataset] Dropping {dropped} rows with missing images.")
        self.df = df[mask].reset_index(drop=True)
        ########
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
      
        row     = self.df.iloc[idx]
        img_rel = row["Image_File(FA)"].replace("\\", "/")
        is_OD   = "OD" in row["Image_File(FA)"]

        # Try both extensions
        base     = os.path.splitext(os.path.normpath(os.path.join(self.img_dir, img_rel)))[0]
        img_path = None
        for ext in (".jpg", ".png", ".jpeg"):
            candidate = base + ext
            if os.path.exists(candidate):
                img_path = candidate
                break
        if img_path is None:
            raise FileNotFoundError(f"No image found for {base} (.jpg / .png / .jpeg)")

        image = Image.open(img_path).convert("RGB")

        if is_OD:
            image = ImageOps.mirror(image)
        image = self.transform(image)

        labels = torch.tensor(
            [row[f"Zone{i}_label"] for i in range(1, 11)],
            dtype=torch.float32,
        )
        return image, labels


def make_loader(dataset, batch_size, num_workers, shuffle):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
    )