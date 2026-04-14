# config.py

DATA = {
    "train_csv":  "/home/tim/UVEITIS_OCT_classidication/fold_4/train.csv",
    "val_csv":    "/home/tim/UVEITIS_OCT_classidication/fold_4/val.csv",
    "test_csv":   "/home/tim/UVEITIS_OCT_classidication/fold_4/test.csv",
    "img_dir":    "/home/tim/uveitis-research/data/updated_samples",
    "batch_size": 32,
    "workers":    4,
    "seed":       84,
}

MODEL = {
    "backbone":    "swin_small_patch4_window7_224",
    "pretrained":  True,
    "num_zones":   10,
    "hidden_dim":  256,
    "dropout":     0.3,
    "num_classes": 3,
}

OPTIM = {
    "lr":           3e-5,
    "weight_decay": 1e-4,
}