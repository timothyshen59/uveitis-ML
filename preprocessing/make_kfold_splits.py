#!/usr/bin/env python3
"""
make_kfold_splits.py

Patient-level stratified k-fold splits with held-out test set.
- Test set held out first, never used during training
- No patient appears in more than one fold's val set
- 0/1/2 severity distribution kept similar across all splits
- Severity = max zone label per image (image-level stratum)

Usage:
    python make_kfold_splits.py \
        --csv  path/to/data.xlsx \
        --out  folds/ \
        --k    5 \
        --val_size  0.15 \
        --test_size 0.15
"""

import argparse
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, StratifiedShuffleSplit

LABEL_COLS = [f"Zone{i}_label" for i in range(1, 11)]


def load_data(path, sep="\t"):
    if path.endswith(".xlsx") or path.endswith(".xls"):
        df = pd.read_excel(path)
    else:
        df = pd.read_csv(path, sep=sep)
    return df


def add_stratum(df):
    """Image-level severity = max zone label across all zones."""
    df = df.copy()
    df["stratum"] = df[LABEL_COLS].max(axis=1)
    return df


def split_test(df, test_size, random_state):
    """
    Hold out test set at patient level, stratified by severity.
    Returns (trainval_df, test_df).
    """
    patient_stratum = (
        df.groupby("Patient_ID")["stratum"]
        .max()
        .reset_index()
    )
    patients = patient_stratum["Patient_ID"].values
    strata   = patient_stratum["stratum"].values

    sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    trainval_idx, test_idx = next(sss.split(patients, strata))

    trainval_patients = set(patients[trainval_idx])
    test_patients     = set(patients[test_idx])

    trainval_df = df[df["Patient_ID"].isin(trainval_patients)].reset_index(drop=True)
    test_df     = df[df["Patient_ID"].isin(test_patients)].reset_index(drop=True)

    return trainval_df, test_df


def verify_no_leakage(train_df, val_df, test_df, fold):
    tp = set(train_df["Patient_ID"])
    vp = set(val_df["Patient_ID"])
    sp = set(test_df["Patient_ID"])
    assert tp.isdisjoint(vp), f"[fold {fold}] Patient leakage: train/val"
    assert tp.isdisjoint(sp), f"[fold {fold}] Patient leakage: train/test"
    assert vp.isdisjoint(sp), f"[fold {fold}] Patient leakage: val/test"


def print_distribution(name, df):
    counts = df["stratum"].value_counts().sort_index().to_dict()
    n = len(df)
    print(f"  [{name:<6}] images={n:>4} patients={df['Patient_ID'].nunique():>3}  "
          f"sev_0={counts.get(0,0):>3}({counts.get(0,0)/n*100:.0f}%)  "
          f"sev_1={counts.get(1,0):>3}({counts.get(1,0)/n*100:.0f}%)  "
          f"sev_2={counts.get(2,0):>3}({counts.get(2,0)/n*100:.0f}%)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv",          type=str,   required=True, help="Path to master CSV or XLSX")
    parser.add_argument("--out",          type=str,   default="folds", help="Output directory")
    parser.add_argument("--k",            type=int,   default=5,     help="Number of folds (default: 5)")
    parser.add_argument("--val_size",     type=float, default=0.15,  help="Val fraction (default: 0.15)")
    parser.add_argument("--test_size",    type=float, default=0.15,  help="Test fraction (default: 0.15)")
    parser.add_argument("--random_state", type=int,   default=42)
    parser.add_argument("--sep",          type=str,   default="\t",  help="CSV separator (default: tab)")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    # load
    df = load_data(args.csv, sep=args.sep)
    print(f"[data] loaded {len(df)} images, {df['Patient_ID'].nunique()} unique patients")
    df = df.dropna(subset=LABEL_COLS).reset_index(drop=True)
    print(f"[data] after dropna: {len(df)} images")
    df = add_stratum(df)

    print(f"\n[data] overall severity distribution:")
    for sev, cnt in df["stratum"].value_counts().sort_index().items():
        print(f"  severity_{int(sev)}: {cnt} ({cnt/len(df)*100:.1f}%)")

    # hold out test set first
    trainval_df, test_df = split_test(df, test_size=args.test_size, random_state=args.random_state)
    print(f"\n[split] held-out test set: {len(test_df)} images, {test_df['Patient_ID'].nunique()} patients")
    print(f"[split] trainval pool:     {len(trainval_df)} images, {trainval_df['Patient_ID'].nunique()} patients")

    # save test set once — shared across all folds
    test_path = os.path.join(args.out, "test.csv")
    test_df.drop(columns=["stratum"]).to_csv(test_path, index=False)
    print(f"[saved] {test_path}")

    # k-fold on trainval
    sgkf = StratifiedGroupKFold(n_splits=args.k, shuffle=True, random_state=args.random_state)

    print(f"\n[kfold] {args.k} folds on trainval pool")
    for fold, (train_idx, val_idx) in enumerate(
        sgkf.split(trainval_df, y=trainval_df["stratum"], groups=trainval_df["Patient_ID"]), start=1
    ):
        train_df = trainval_df.iloc[train_idx].reset_index(drop=True)
        val_df   = trainval_df.iloc[val_idx].reset_index(drop=True)

        verify_no_leakage(train_df, val_df, test_df, fold)

        print(f"\n[fold {fold}]")
        print_distribution("train", train_df)
        print_distribution("val",   val_df)
        print_distribution("test",  test_df)
        print(f"  [verify] No patient leakage ✓")

        fold_dir = os.path.join(args.out, f"fold_{fold}")
        os.makedirs(fold_dir, exist_ok=True)

        train_df.drop(columns=["stratum"]).to_csv(os.path.join(fold_dir, "train.csv"), index=False)
        val_df.drop(columns=["stratum"]).to_csv(os.path.join(fold_dir, "val.csv"),     index=False)

        print(f"  [saved] {fold_dir}/train.csv")
        print(f"  [saved] {fold_dir}/val.csv")

    print(f"\n[done] folder structure:")
    print(f"  {args.out}/test.csv          ← held out, same for all folds")
    for fold in range(1, args.k + 1):
        print(f"  {args.out}/fold_{fold}/train.csv")
        print(f"  {args.out}/fold_{fold}/val.csv")


if __name__ == "__main__":
    main()