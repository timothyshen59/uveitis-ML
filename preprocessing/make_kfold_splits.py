#!/usr/bin/env python3
"""
make_kfold_splits.py  —  patient-level stratified k-fold splits with a held-out test set.

Usage:
    python make_kfold_splits.py --csv path/to/data.xlsx --out folds/ --k 5
"""

import argparse
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, StratifiedShuffleSplit

LABEL_COLS = [f"Zone{i}_label" for i in range(1, 11)]


def load_data(path, sep="\t"):
    if path.endswith(".xlsx") or path.endswith(".xls"):
        return pd.read_excel(path)
    return pd.read_csv(path, sep=sep)


def add_stratum(df):
    df = df.copy()
    df["stratum"] = df[LABEL_COLS].max(axis=1)
    return df


def split_test(df, test_size, random_state):
    """Holds out a patient-level stratified test set. Returns (trainval_df, test_df)."""
    patient_strata = df.groupby("Patient_ID")["stratum"].max().reset_index()
    patients = patient_strata["Patient_ID"].values
    strata   = patient_strata["stratum"].values

    sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    trainval_idx, test_idx = next(sss.split(patients, strata))

    trainval_df = df[df["Patient_ID"].isin(patients[trainval_idx])].reset_index(drop=True)
    test_df     = df[df["Patient_ID"].isin(patients[test_idx])].reset_index(drop=True)
    return trainval_df, test_df


def verify_no_leakage(train_df, val_df, test_df, fold):
    tp, vp, sp = set(train_df["Patient_ID"]), set(val_df["Patient_ID"]), set(test_df["Patient_ID"])
    assert tp.isdisjoint(vp), f"[fold {fold}] leakage: train/val"
    assert tp.isdisjoint(sp), f"[fold {fold}] leakage: train/test"
    assert vp.isdisjoint(sp), f"[fold {fold}] leakage: val/test"


def print_distribution(name, df):
    counts = df["stratum"].value_counts().sort_index().to_dict()
    n = len(df)
    print(f"  [{name:<6}] images={n:>4} patients={df['Patient_ID'].nunique():>3}  "
          f"sev_0={counts.get(0,0):>3}({counts.get(0,0)/n*100:.0f}%)  "
          f"sev_1={counts.get(1,0):>3}({counts.get(1,0)/n*100:.0f}%)  "
          f"sev_2={counts.get(2,0):>3}({counts.get(2,0)/n*100:.0f}%)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv",          type=str,   required=True)
    parser.add_argument("--out",          type=str,   default="folds")
    parser.add_argument("--k",            type=int,   default=5)
    parser.add_argument("--val_size",     type=float, default=0.15)
    parser.add_argument("--test_size",    type=float, default=0.15)
    parser.add_argument("--random_state", type=int,   default=42)
    parser.add_argument("--sep",          type=str,   default="\t")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    df = load_data(args.csv, sep=args.sep)
    df = df.dropna(subset=LABEL_COLS).reset_index(drop=True)
    df = add_stratum(df)
    print(f"[data] {len(df)} images, {df['Patient_ID'].nunique()} patients")
    for sev, cnt in df["stratum"].value_counts().sort_index().items():
        print(f"  severity_{int(sev)}: {cnt} ({cnt/len(df)*100:.1f}%)")

    trainval_df, test_df = split_test(df, args.test_size, args.random_state)
    print(f"\n[split] test={len(test_df)} images / {test_df['Patient_ID'].nunique()} patients  "
          f"| trainval={len(trainval_df)} images / {trainval_df['Patient_ID'].nunique()} patients")

    test_path = os.path.join(args.out, "test.csv")
    test_df.drop(columns=["stratum"]).to_csv(test_path, index=False)
    print(f"[saved] {test_path}")

    sgkf = StratifiedGroupKFold(n_splits=args.k, shuffle=True, random_state=args.random_state)

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

        fold_dir = os.path.join(args.out, f"fold_{fold}")
        os.makedirs(fold_dir, exist_ok=True)
        train_df.drop(columns=["stratum"]).to_csv(os.path.join(fold_dir, "train.csv"), index=False)
        val_df.drop(columns=["stratum"]).to_csv(os.path.join(fold_dir, "val.csv"),     index=False)
        print(f"  [saved] {fold_dir}/")

    print(f"\n[done]")


if __name__ == "__main__":
    main()