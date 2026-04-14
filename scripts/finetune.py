import argparse
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import timm
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
from sklearn.model_selection import KFold
from PIL import Image, ImageOps
from tqdm import tqdm


# ── Dataset ──────────────────────────────────────────────────────────────────
class FundusDataset(Dataset):
    def __init__(self, df, img_dir, transform=None):
        self.df = df
        self.img_dir = img_dir
        self.transform = transform or transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        img_rel  = row["Image_File(FA)"].replace("\\", "/")
        img_path = os.path.normpath(os.path.join(self.img_dir, img_rel))
        is_OD    = "OD" in row["Image_File(FA)"]

        image = Image.open(img_path).convert('RGB')
        if is_OD:
            image = ImageOps.mirror(image)
        if self.transform:
            image = self.transform(image)

        labels = torch.tensor(
            [row[f"Zone{i}_label"] for i in range(1, 11)],
            dtype=torch.float32
        )
        return image, labels


# ── Model ─────────────────────────────────────────────────────────────────────
class ViTBaseModel(nn.Module):
    def __init__(self, num_zones=10, hidden_dim=256):
        super().__init__()
        self.num_thresholds = 2

        self.backbone = timm.create_model(
            "vit_small_patch16_224",
            pretrained=True,
            num_classes=0
        )

        feat_dim = self.backbone.num_features

        self.mlp = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.3),
        )

        self.ordinal_heads = nn.ModuleList([
            nn.Linear(hidden_dim, self.num_thresholds)
            for _ in range(num_zones)
        ])

    def forward(self, x):
        feats = self.backbone(x)
        feats = self.mlp(feats)
        return torch.stack([head(feats) for head in self.ordinal_heads], dim=1)


# ── Ordinal helpers ───────────────────────────────────────────────────────────
def encode_ordinal(labels, num_classes=3):
    B, Z = labels.shape
    thresholds = num_classes - 1
    out = torch.zeros(B, Z, thresholds, device=labels.device)
    for k in range(thresholds):
        out[:, :, k] = (labels > k).float()
    return out


def ordinal_to_class(logits):
    return torch.sum(torch.sigmoid(logits) > 0.5, dim=-1)


# ── Train / validate ──────────────────────────────────────────────────────────
def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for images, labels in tqdm(loader, desc="  train"):
        images, labels = images.to(device), labels.to(device)
        loss = criterion(model(images), encode_ordinal(labels))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, total_correct, total_samples = 0, 0, 0
    with torch.no_grad():
        for images, labels in tqdm(loader, desc="  eval "):
            images, labels = images.to(device), labels.to(device)
            preds = model(images)
            total_loss    += criterion(preds, encode_ordinal(labels)).item()
            total_correct += (ordinal_to_class(preds) == labels).sum().item()
            total_samples += labels.numel()
    return total_loss / len(loader), total_correct / total_samples


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--xlsx',       required=True,        help='Path to labels.xlsx')
    parser.add_argument('--img_dir',    required=True,        help='Path to images folder')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--workers',    type=int, default=4)
    parser.add_argument('--epochs',     type=int, default=10)
    parser.add_argument('--n_folds',    type=int, default=10, help='K for KFold (10 → 80/10/10)')
    parser.add_argument('--seed',       type=int, default=42)
    args = parser.parse_args()

    # Load & clean data
    df = pd.read_excel(args.xlsx)
    label_cols = [f"Zone{i}_label" for i in range(1, 11)]
    df = df.dropna(subset=label_cols).reset_index(drop=True)

    dataset   = FundusDataset(df, args.img_dir)
    device    = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    criterion = nn.BCEWithLogitsLoss()

    # KFold: n_folds=10 → each fold holds out 10% as test.
    # The remaining 90% is split ~89/11 → train/val, giving ~80/10/10 overall.
    kf          = KFold(n_splits=args.n_folds, shuffle=True, random_state=args.seed)
    all_indices = np.arange(len(dataset))
    fold_results = []

    for fold, (trainval_idx, test_idx) in enumerate(kf.split(all_indices)):
        print(f"\n{'='*60}")
        print(f" FOLD {fold+1}/{args.n_folds}  "
              f"(train+val={len(trainval_idx)}, test={len(test_idx)})")
        print(f"{'='*60}")

        # Split trainval → train / val
        rng       = np.random.default_rng(args.seed + fold)
        shuffled  = rng.permutation(trainval_idx)
        val_size  = max(1, len(trainval_idx) // args.n_folds)
        train_idx = shuffled[val_size:]
        val_idx   = shuffled[:val_size]

        print(f" train={len(train_idx)}  val={len(val_idx)}  test={len(test_idx)}\n")

        def make_loader(idx, shuffle):
            return DataLoader(
                Subset(dataset, idx),
                batch_size=args.batch_size,
                shuffle=shuffle,
                num_workers=args.workers,
                pin_memory=True
            )

        train_loader = make_loader(train_idx, shuffle=True)
        val_loader   = make_loader(val_idx,   shuffle=False)
        test_loader  = make_loader(test_idx,  shuffle=False)

        # Fresh model + optimizer per fold
        model = ViTBaseModel().to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-5, weight_decay=1e-4)

        best_val_loss = float("inf")
        best_state    = None

        for epoch in range(args.epochs):
            train_loss           = train_epoch(model, train_loader, optimizer, criterion, device)
            val_loss, val_acc    = evaluate(model, val_loader, criterion, device)

            print(f" Epoch {epoch+1:02d}/{args.epochs}  "
                  f"train={train_loss:.4f}  "
                  f"val={val_loss:.4f}  acc={val_acc*100:.2f}%")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state    = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        # Evaluate best checkpoint on held-out test set
        model.load_state_dict(best_state)
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)
        print(f"\n Fold {fold+1} TEST → loss={test_loss:.4f}  acc={test_acc*100:.2f}%")
        fold_results.append((test_loss, test_acc))

    # Summary
    losses, accs = zip(*fold_results)
    print(f"\n{'='*60}")
    print(" CROSS-VALIDATION RESULTS")
    print(f"{'='*60}")
    for i, (l, a) in enumerate(fold_results):
        print(f" Fold {i+1:2d}:  loss={l:.4f}  acc={a*100:.2f}%")
    print(f"\n Mean loss : {np.mean(losses):.4f} ± {np.std(losses):.4f}")
    print(f" Mean acc  : {np.mean(accs)*100:.2f}% ± {np.std(accs)*100:.2f}%")