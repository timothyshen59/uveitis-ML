#!/usr/bin/env python3
"""
train_with_crops.py

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
from sklearn.utils.class_weight import compute_class_weight
from tqdm import tqdm
from pathlib import Path

from config.config import DATA, MODEL, OPTIM 
from preprocessing.dataset import FundusDataset, make_loader
from models.hybrid_cross import HybridModel, load_model


def compute_cf_counts(logits, labels): 
    preds = (logits > 0).long()
    labels = labels.long()
    tp = ((preds == 1) & (labels == 1)).sum()
    fp = ((preds == 1) & (labels == 0)).sum()
    fn = ((preds == 0) & (labels == 1)).sum()
    return tp, fp, fn  


def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    for full_imgs, zone_imgs, zone_labels, _ in tqdm(loader, desc="  train", leave=False):
        full_imgs   = full_imgs.to(device)
        zone_imgs   = zone_imgs.to(device)
        zone_labels = zone_labels.to(device)

        logits = model(full_imgs, zone_imgs)                      # (B, num_zones, num_classes)
        B, num_zones, num_classes = logits.shape

        # use positive class logit for BCE
        loss = criterion(
            logits[:, :, 1],                                      # (B, num_zones)
            zone_labels.float()                                   # (B, num_zones)
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


def evaluate(model, loader, criterion, device, num_zones):
    model.eval()
    total_loss, total_correct, total_samples = 0.0, 0, 0
    tp_total, fp_total, fn_total = 0, 0, 0
    zone_cnt     = torch.zeros(num_zones, device=device)
    zone_correct = torch.zeros(num_zones, device=device)

    misclassified = []  # collect misclassified samples

    with torch.no_grad():
        for full_imgs, zone_imgs, zone_labels, img_paths in tqdm(loader, desc="  eval ", leave=False):
            full_imgs   = full_imgs.to(device)
            zone_imgs   = zone_imgs.to(device)
            zone_labels = zone_labels.to(device)

            logits = model(full_imgs, zone_imgs)                  # (B, num_zones, num_classes)
            B, num_zones, num_classes = logits.shape

            loss = criterion(
                logits[:, :, 1],                                  # (B, num_zones)
                zone_labels.float()                               # (B, num_zones)
            )
            total_loss += loss.item()

            preds = (logits[:, :, 1] > 0).long()                  # (B, num_zones)
            total_correct += (preds == zone_labels).sum().item()
            total_samples += zone_labels.numel()

            for z in range(num_zones):
                zone_cnt[z]     += B
                zone_correct[z] += (preds[:, z] == zone_labels[:, z]).sum().item()

            for i in range(B):
                wrong_zones = (preds[i] != zone_labels[i]).nonzero(as_tuple=True)[0].tolist()
                if wrong_zones:
                    misclassified.append({
                        "image":       img_paths[i],
                        "wrong_zones": [z + 1 for z in wrong_zones],
                        "preds":       preds[i].cpu().tolist(),
                        "labels":      zone_labels[i].cpu().tolist(),
                    })

            tp, fp, fn = compute_cf_counts(
                logits[:, :, 1].view(-1),
                zone_labels.view(-1)
            )
            tp_total += tp
            fp_total += fp
            fn_total += fn

    pd.DataFrame(misclassified).to_csv("misclassified.csv", index=False)
    print(f"[eval] {len(misclassified)} misclassified samples saved to misclassified.csv")

    precision = tp_total / (tp_total + fp_total).clamp(min=1)
    recall    = tp_total / (tp_total + fn_total).clamp(min=1)
    f1        = 2 * (precision * recall) / (precision + recall).clamp(min=1)
    zone_accs = zone_correct / zone_cnt.clamp(min=1)

    return total_loss / len(loader), total_correct / total_samples, f1, zone_accs


def load_split(csv_path, label_cols):
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
        # "test_csv":     dcfg["test_csv"],
        "batch_size":   dcfg["batch_size"],
        "lr":           ocfg["lr"],
        "weight_decay": ocfg["weight_decay"],
        "num_classes":  mcfg["num_classes"],
        "seed":         dcfg["seed"],
        "vit_backbone": mcfg["vit_backbone"],
        "cnn_backbone": mcfg["cnn_backbone"],
        "num_zones":    mcfg["num_zones"],
        "proj_dim":     mcfg["proj_dim"],
        "embed_dim":    mcfg["embed_dim"],
    }


def main():
    args = parse_args()
    
    dcfg = DATA 
    mcfg = MODEL
    ocfg = OPTIM
    
    #Set Seed 
    seed = dcfg["seed"]
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = False 
    torch.backends.cudnn.benchmark = True 

    wb_config = make_wandb_config(args, dcfg, mcfg, ocfg)
    device    = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print(f"\n[config] epochs={args.epochs}  model_path={args.model_path or 'None (pretrained backbone)'}")
    print(f"[config] device={device}  batch={dcfg['batch_size']}")
    # print(f"[config] train={dcfg['train_csv']}  val={dcfg['val_csv']}  test={dcfg['test_csv']}\n")
    print(f"[config] train={dcfg['train_csv']}  val={dcfg['val_csv']}\n")

    label_cols = [f"Zone{i}_label" for i in range(1, 11)]

    train_df = load_split(dcfg["train_csv"], label_cols)
    val_df   = load_split(dcfg["val_csv"],   label_cols)
    # test_df  = load_split(dcfg["test_csv"],  label_cols)

    # print(f"[data] train={len(train_df)}  val={len(val_df)}  test={len(test_df)}\n")
    print(f"[data] train={len(train_df)}  val={len(val_df)}\n")

    criterion = nn.BCEWithLogitsLoss()

    cache_dir = Path(dcfg["img_dir"]) / dcfg["cache_dir"]

    dataset_kwargs = dict(
        img_dir=dcfg["img_dir"],
        include_zone_crops=True,
        cache_dir=cache_dir,
        use_cache=True,
        cache_version=dcfg["cache_version"],       
        use_clahe=True,
        zone_nums=range(1, 10),   # Zones 1–9 only, removes Zone 10
        zone_pad=10,
        outside_scale=0.5,        # 50% dimmed context outside the zone
        pad_zone_to_square=True,
    )

    train_dataset = FundusDataset(
        train_df,
        **dataset_kwargs,
    )

    val_dataset = FundusDataset(
        val_df,
        **dataset_kwargs,
    )
    train_loader = make_loader(
        train_dataset,
        dcfg["batch_size"],
        dcfg["workers"],
        shuffle=True,
    )

    val_loader = make_loader(
        val_dataset,
        dcfg["batch_size"],
        dcfg["workers"],
        shuffle=False,
    )
    # train_loader = make_loader(FundusDataset(train_df, dcfg["img_dir"], include_zone_crops=True), dcfg["batch_size"], dcfg["workers"], shuffle=True)
    # val_loader   = make_loader(FundusDataset(val_df,   dcfg["img_dir"], include_zone_crops=True), dcfg["batch_size"], dcfg["workers"], shuffle=False)
    # test_loader  = make_loader(FundusDataset(test_df,  dcfg["img_dir"], include_zone_crops=True), dcfg["batch_size"], dcfg["workers"], shuffle=False)

    model     = load_model(args.model_path, mcfg, device)
    optimizer = torch.optim.AdamW([
        {"params": model.vit.parameters(),        "lr": ocfg.get("backbone_lr", 1e-5)},
        {"params": model.cnn.parameters(),        "lr": ocfg.get("backbone_lr", 1e-5)},
        {"params": model.cnn_to_shared.parameters(),   "lr": ocfg["lr"]},
        {"params": model.patch_to_shared.parameters(), "lr": ocfg["lr"]},
        {"params": model.cls_to_shared.parameters(),   "lr": ocfg["lr"]},
        {"params": model.mha.parameters(),             "lr": ocfg["lr"]},
        {"params": model.mlp.parameters(),       "lr": ocfg["lr"]},
        {"params": model.head.parameters(),      "lr": ocfg["lr"]},],
        weight_decay=ocfg["weight_decay"]
    )


    wandb.init(
        project="fundus-ordinal-regression",
        name=args.run_name,
        config=wb_config,
    )

    best_val_loss = float("inf")
    best_state    = None
    os.makedirs("checkpoints", exist_ok=True)

    for epoch in range(args.epochs):
        
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc, val_f1, zone_acc = evaluate(model, val_loader, criterion,
                                                        device, mcfg["num_zones"])


        print(f" Epoch {epoch+1:02d}/{args.epochs}  "
              f"train={train_loss:.4f}  "
              f"val={val_loss:.4f}  acc={val_acc*100:.2f}%")

        zone_acc_dict = {f"val-zone-acc/zone_{i+1}": zone_acc[i].item() for i in range(len(zone_acc))}
        wandb.log({"epoch": epoch + 1, "train-loss": train_loss,
                   "val-loss": val_loss, "val-acc": val_acc, "val-f1": val_f1, **zone_acc_dict})

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state    = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    ckpt_path = "checkpoints/best.pt"
    torch.save(best_state, ckpt_path)
    print(f"\n Saved best checkpoint → {ckpt_path}")


    # model.load_state_dict(best_state)
    # test_loss, test_acc, test_f1, test_zone_acc = evaluate(model, test_loader, criterion,
    #                                                         device, mcfg["num_zones"])
    # print(f"\n TEST → loss={test_loss:.4f}  acc={test_acc*100:.2f}%")
    # test_zone_acc_dict = {f"test-zone-acc/zone_{i+1}": test_zone_acc[i].item() for i in range(len(test_zone_acc))}
    # wandb.log({"test/loss": test_loss, "test/acc": test_acc, "test/f1": test_f1, **test_zone_acc_dict})
    # wandb.summary["test/loss"] = test_loss
    # wandb.summary["test/acc"]  = test_acc

    wandb.finish()


if __name__ == "__main__":
    main()