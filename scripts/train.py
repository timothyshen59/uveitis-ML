#!/usr/bin/env python3
"""
train.py

Args:
  --epochs      # of training epochs
  --model_path  Path to a pretrained checkpoint
"""

import sys
import os
sys.path.append('/home/tim/uveitis-research')

import argparse
import wandb
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm

from config.config import DATA, MODEL, OPTIM 
from preprocessing.dataset import FundusDataset, make_loader
from models.Base_CNN import Base_CNN, load_model #Change for training ViTs alone


from sklearn.utils.class_weight import compute_class_weight
import numpy as np

#Bottlenecks: Training single MLP head to classify the same for 10 different zones? 
# ViT function well on glboal level, but no local or spatial representaiton (combine hybrid architecture + SSL Pretraining)
# Next Steps: Hybrid architecture and pretraining etc. 


def compute_cf_counts(logits, labels):
    preds = (logits > 0).long()
    labels = labels.long()
    tp = ((preds == 1) & (labels == 1)).sum()
    fp = ((preds == 1) & (labels == 0)).sum()
    fn = ((preds == 0) & (labels == 1)).sum()
    return tp, fp, fn

def train_epoch(model, loader, optimizer, criterion, device, scheduler):
    model.train()
    total_loss = 0.0
    for full_imgs, zone_labels in tqdm(loader, desc="  train", leave=False):
        full_imgs   = full_imgs.to(device)
        zone_labels = zone_labels.to(device)          # (B, num_zones)

        logits = model(full_imgs)                     # (B, num_zones)

        loss = criterion(logits, zone_labels.float())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # scheduler.step() 
        total_loss += loss.item()
    return total_loss / len(loader)

def evaluate(model, loader, criterion, device, num_zones):
    model.eval()
    total_loss, total_correct, total_samples = 0.0, 0, 0
    tp_total, fp_total, fn_total = 0, 0, 0
    zone_cnt     = torch.zeros(num_zones, device=device)
    zone_correct = torch.zeros(num_zones, device=device)

    with torch.no_grad():
        for full_imgs, zone_labels in tqdm(loader, desc="  eval ", leave=False):
            full_imgs   = full_imgs.to(device)
            zone_labels = zone_labels.to(device)           # (B, num_zones)

            logits = model(full_imgs)                      # (B, num_zones)
            B, num_zones = logits.shape

            loss = criterion(logits, zone_labels.float())
            total_loss += loss.item()

            preds = (logits > 0).long()                    # (B, num_zones)
            total_correct += (preds == zone_labels).sum().item()
            total_samples += zone_labels.numel()

            for z in range(num_zones):
                zone_cnt[z]     += B
                zone_correct[z] += (preds[:, z] == zone_labels[:, z]).sum().item()

            tp, fp, fn = compute_cf_counts(logits.view(-1), zone_labels.view(-1))
            tp_total += tp
            fp_total += fp
            fn_total += fn

    precision = tp_total / (tp_total + fp_total).clamp(min=1)
    recall    = tp_total / (tp_total + fn_total).clamp(min=1)
    f1        = 2 * (precision * recall) / (precision + recall).clamp(min=1)
    zone_accs = zone_correct / zone_cnt.clamp(min=1)

    return total_loss / len(loader), total_correct / total_samples, f1, zone_accs

def load_split(csv_path, label_cols):
    """Read CSV and drop rows with missing labels."""
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=label_cols).reset_index(drop=True)
    return df


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs",     type=int, required=True, help="Number of training epochs")
    parser.add_argument("--model_path", type=str, default=None,  help="Path to checkpoint for continued training")
    parser.add_argument("--run_name",   type=str, default=None,  help="Label for training run")
    return parser.parse_args()


def make_wandb_config(args, dcfg, mcfg, ocfg):
    return {
        "epochs":       args.epochs,
        "model_path":   args.model_path,
        "train_csv":    dcfg["train_csv"],
        "val_csv":      dcfg["val_csv"],
        "test_csv":     dcfg["test_csv"],
        "batch_size":   dcfg["batch_size"],
        "lr":           ocfg["lr"],
        "weight_decay": ocfg["weight_decay"],
        "num_classes":  mcfg["num_classes"],
        "seed":         dcfg["seed"],
        "vit_backbone": mcfg["vit_backbone"],
        "num_zones":    mcfg["num_zones"],
    }


def main():
    args = parse_args()

    dcfg = DATA 
    mcfg = MODEL
    ocfg = OPTIM

    wb_config = make_wandb_config(args, dcfg, mcfg, ocfg)
    device    = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    criterion = nn.CrossEntropyLoss()

    print(f"\n[config] epochs={args.epochs}  model_path={args.model_path or 'None (pretrained backbone)'}")
    print(f"[config] device={device}  batch={dcfg['batch_size']}")
    print(f"[config] train={dcfg['train_csv']}  val={dcfg['val_csv']}  test={dcfg['test_csv']}\n")

    label_cols = [f"Zone{i}_label" for i in range(1, 11)]

    train_df = load_split(dcfg["train_csv"], label_cols)
    print(train_df[[f"Zone{i}_label" for i in range(1, 11)]].mean())

    val_df   = load_split(dcfg["val_csv"],   label_cols)
    test_df  = load_split(dcfg["test_csv"],  label_cols)
    
    #####
    pos_weights = []
    for col in label_cols:
        labels = (train_df[col] > 0).astype(int).values
        classes = np.array([0, 1])
        weights = compute_class_weight('balanced', classes=classes, y=labels)
        pos_weights.append(weights[1] / weights[0])  # ratio of pos to neg weight

    pos_weight_tensor = torch.tensor(pos_weights, dtype=torch.float32).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
    ########## 
    print(f"[data] train={len(train_df)}  val={len(val_df)}  test={len(test_df)}\n")

    train_loader = make_loader(FundusDataset(train_df, dcfg["img_dir"], include_zone_crops=False), dcfg["batch_size"], dcfg["workers"], shuffle=True)
    val_loader   = make_loader(FundusDataset(val_df,   dcfg["img_dir"], include_zone_crops=False), dcfg["batch_size"], dcfg["workers"], shuffle=False)
    test_loader  = make_loader(FundusDataset(test_df,  dcfg["img_dir"], include_zone_crops=False), dcfg["batch_size"], dcfg["workers"], shuffle=False)

    model     = load_model(args.model_path, mcfg, device)
    # optimizer = torch.optim.AdamW(
    #     filter(lambda p: p.requires_grad, model.parameters()),
    #     lr=ocfg["lr"],
    #     weight_decay=ocfg["weight_decay"]
    # )
    optimizer = torch.optim.AdamW([
    {"params": model.backbone.parameters(), "lr": 1e-5},  # slow — preserve ImageNet features
    {"params": model.head.parameters(),     "lr": 1e-3},  # fast — head learns quickly
    ], weight_decay=0.1)
    
    #Scheduler 
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR( 
        optimizer, 
        T_max = args.epochs + 20, 
        eta_min = 1e-6, 
    )
    

    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"{name:<50} shape={list(param.shape)}")

    wandb.init(
        project="fundus-ordinal-regression",
        name=args.run_name,
        config=wb_config,
    )

    best_val_loss = float("inf")
    best_state    = None
    os.makedirs("checkpoints", exist_ok=True)

    for epoch in range(args.epochs):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device, scheduler)
        val_loss, val_acc, val_f1, zone_acc = evaluate(model, val_loader, criterion,
                                                        device, mcfg["num_zones"])

        print(f" Epoch {epoch+1:02d}/{args.epochs}  "
              f"train={train_loss:.4f}  "
              f"val={val_loss:.4f}  acc={val_acc*100:.2f}%")

        zone_acc_dict = {f"val-zone-acc/zone_{i+1}": zone_acc[i].item() for i in range(len(zone_acc))}

        wandb.log({"epoch": epoch + 1, "train-loss": train_loss,
                   "val-loss": val_loss, "val-acc": val_acc, "val-f1": val_f1, **zone_acc_dict}) # Log entire cnt arrays + F1 Score

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state    = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    ckpt_path = "checkpoints/best.pt"
    torch.save(best_state, ckpt_path)
    print(f"\n Saved best checkpoint → {ckpt_path}")

    model.load_state_dict(best_state)
    test_loss, test_acc, test_f1, test_zone_acc = evaluate(model, test_loader, criterion,
                                                            device, mcfg["num_zones"])

    # Log Zone Accuracy Results
    print(f"\n TEST → loss={test_loss:.4f}  acc={test_acc*100:.2f}%")

    test_zone_acc_dict = {f"test-zone-acc/zone_{i+1}": test_zone_acc[i].item() for i in range(len(test_zone_acc))}
    wandb.log({"test/loss": test_loss, "test/acc": test_acc, "test/f1": test_f1, **test_zone_acc_dict})
    wandb.summary["test/loss"] = test_loss
    wandb.summary["test/acc"]  = test_acc
    wandb.finish()


if __name__ == "__main__":
    main()