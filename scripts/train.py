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

import config
from dataset import FundusDataset, make_loader
from models.VitS import ViTBaseModel, load_model, encode_ordinal, ordinal_to_class


def train_epoch(model, loader, optimizer, criterion, device, num_classes):
    model.train()
    total_loss = 0.0
    for images, labels in tqdm(loader, desc="  train", leave=False):
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        loss   = criterion(logits, encode_ordinal(labels, num_classes))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


def evaluate(model, loader, criterion, device, num_classes):
    model.eval()
    total_loss, total_correct, total_samples = 0.0, 0, 0
    with torch.no_grad():
        for images, labels in tqdm(loader, desc="  eval ", leave=False):
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            total_loss    += criterion(logits, encode_ordinal(labels, num_classes)).item()
            total_correct += (ordinal_to_class(logits) == labels).sum().item()
            total_samples += labels.numel()
    return total_loss / len(loader), total_correct / total_samples


def load_split(csv_path, label_cols):
    """Read CSV  and drop rows with missing labels."""
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
    }


def main():
    args = parse_args()

    dcfg = config.DATA
    mcfg = config.MODEL
    ocfg = config.OPTIM

    wb_config = make_wandb_config(args, dcfg, mcfg, ocfg)
    device    = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    criterion = nn.BCEWithLogitsLoss()

    print(f"\n[config] epochs={args.epochs}  model_path={args.model_path or 'None (pretrained backbone)'}")
    print(f"[config] device={device}  batch={dcfg['batch_size']}")
    print(f"[config] train={dcfg['train_csv']}  val={dcfg['val_csv']}  test={dcfg['test_csv']}\n")

    label_cols = [f"Zone{i}_label" for i in range(1, 11)]

    train_df = load_split(dcfg["train_csv"], label_cols)
    val_df   = load_split(dcfg["val_csv"],   label_cols)
    test_df  = load_split(dcfg["test_csv"],  label_cols)
    
    print(f"[data] train={len(train_df)}  val={len(val_df)}  test={len(test_df)}\n")

    train_loader = make_loader(FundusDataset(train_df, dcfg["img_dir"]), dcfg["batch_size"], dcfg["workers"], shuffle=True)
    val_loader   = make_loader(FundusDataset(val_df,   dcfg["img_dir"]), dcfg["batch_size"], dcfg["workers"], shuffle=False)
    test_loader  = make_loader(FundusDataset(test_df,  dcfg["img_dir"]), dcfg["batch_size"], dcfg["workers"], shuffle=False)
    
    model     = load_model(args.model_path, mcfg, device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=ocfg["lr"], weight_decay=ocfg["weight_decay"]
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
        train_loss        = train_epoch(model, train_loader, optimizer,
                                        criterion, device, mcfg["num_classes"])
        val_loss, val_acc = evaluate(model, val_loader, criterion,
                                     device, mcfg["num_classes"])

        print(f" Epoch {epoch+1:02d}/{args.epochs}  "
              f"train={train_loss:.4f}  "
              f"val={val_loss:.4f}  acc={val_acc*100:.2f}%")

        wandb.log({"epoch": epoch + 1, "train-loss": train_loss,
                   "val-loss": val_loss, "val-acc": val_acc})

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state    = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    ckpt_path = "checkpoints/best.pt"
    torch.save(best_state, ckpt_path)
    print(f"\n Saved best checkpoint → {ckpt_path}")

    model.load_state_dict(best_state)
    test_loss, test_acc = evaluate(model, test_loader, criterion,
                                   device, mcfg["num_classes"])

    print(f"\n TEST → loss={test_loss:.4f}  acc={test_acc*100:.2f}%")

    wandb.log({"test/loss": test_loss, "test/acc": test_acc})
    wandb.summary["test/loss"] = test_loss
    wandb.summary["test/acc"]  = test_acc
    wandb.finish()


if __name__ == "__main__":
    main()