# config.py

DATA = {
    "train_csv":  "/home/tim/uveitis-research/data/folds/fold_2/train.csv",
    "val_csv":    "/home/tim/uveitis-research/data/folds/fold_2/val.csv",
    # "test_csv":   "/home/tim/UVEITIS_OCT_classidication/fold_3/test.csv",
    "img_dir":    "/home/tim/uveitis-research/data/most_recent_samples",
    "batch_size": 32,
    "workers":    4,
    "seed":       84,
    "cache_dir":     "_tensor_cache_nozone10_pad10_scale05_384",
    "cache_version": "v2",

}

MODEL = {
    "vit_backbone": "vit_small_patch16_224",
    "cnn_backbone": "resnet18",
    "hidden_dim":   256,
    "proj_dim":     128,
    "num_classes":  2,
    "num_zones":    10,
    "embed_dim":    64,
    "freeze_backbone": True, 
}

OPTIM = {
    "lr":           1e-3,   # head lr
    "backbone_lr":  1e-5,   # vit + cnn (100x lower)
    "weight_decay": 1e-2,
}