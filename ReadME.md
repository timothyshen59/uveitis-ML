# Uveitis Disease Classification

ML pipeline for zone-level uveitis severity classification from Ultra-Widefield Fluorescein Angiography (UWFFA) retinal images.

---


## Project Structure

```
config/
    config.py               # DATA, MODEL, OPTIM hyperparameter dicts

models/
    hybrid_model.py         # HybridModel definition + load_model()

preprocessing/
    mask.py                 # Generates per-zone segmentation masks from UWFFA images
    dataset.py              # FundusDataset, make_loader

scripts/
    train.py                # Training + validation loop
    make_kfold_splits.py    # Patient-level stratified k-fold split generation
```

---

## Setup

```bash
pip install -e .
```

---

## Preprocessing

**Step 1 — Generate zone masks**

Detects yellow zone boundary lines in UWFFA images via HSV thresholding and saves per-image masks as `.npy` files:

```bash
python preprocessing/mask.py \
    --image_dir /path/to/images \
    --csv_path  /path/to/data.csv
```

**Step 2 — Generate k-fold splits**

Creates patient-level stratified k-fold splits with a held-out test set:

```bash
python scripts/make_kfold_splits.py \
    --csv  /path/to/data.xlsx \
    --out  folds/ \
    --k    5 \
    --test_size 0.15
```

Output structure:
```
folds/
    test.csv
    fold_1/train.csv
    fold_1/val.csv
    ...
```

---

## Training

```bash
python scripts/train.py \
    --epochs 50 \
    --run_name my_run
```

To resume from a checkpoint:
```bash
python scripts/train.py \
    --epochs 50 \
    --model_path checkpoints/best.pt
```

Training logs to [Weights & Biases](https://wandb.ai) and saves the best checkpoint (by val loss) to `checkpoints/best.pt`.

---

## Configuration

All hyperparameters live in `config/config.py`:

```python
DATA = {
    "img_dir":       "/path/to/images",
    "train_csv":     "folds/fold_1/train.csv",
    "val_csv":       "folds/fold_1/val.csv",
    "batch_size":    16,
    "workers":       4,
    "seed":          42,
    "cache_dir":     "_tensor_cache_v1",
    "cache_version": "v1",
}

MODEL = {
    "cnn_backbone":   "resnet50",
    "proj_dim":       64,
    "num_classes":    2,
    "num_zones":      9,
    "freeze_backbone": True,
}

OPTIM = {
    "lr":          1e-4,
    "backbone_lr": 1e-5,
    "weight_decay": 1e-2,
}
```

---

## Dataset

`FundusDataset` expects a CSV with:
- `UWFFA` — path to the image file
- `Zone1_label` … `Zone10_label` — per-zone severity labels

Zone crops are cached as `.pt` tensors on first load to speed up subsequent epochs.
