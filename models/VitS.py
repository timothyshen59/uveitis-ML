import torch
import torch.nn as nn
import timm

class ViTBaseModel(nn.Module):
    def __init__(self, backbone, hidden_dim=256, num_classes=3):
        super().__init__()

        self.backbone = timm.create_model(
            backbone, 
            pretrained=True,
            num_classes=0
        )

        feat_dim = self.backbone.num_features

        self.mlp = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.3),
        )

        self.head = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        feats = self.backbone(x)
        feats = self.mlp(feats)
        return self.head(feats)  # (B, num_classes)


def load_model(model_path, model_cfg, device):
    model = ViTBaseModel(
        backbone=model_cfg["backbone"],
        hidden_dim=model_cfg["hidden_dim"],
        num_classes=model_cfg["num_classes"],
    ).to(device)

    if model_path:
        state = torch.load(model_path, map_location=device)
        if isinstance(state, dict) and "model" in state:
            state = state["model"]
        model.load_state_dict(state)
        print(f"[model] Loaded weights from: {model_path}")
    else:
        print("[model] Starting from ImageNet pretrained backbone.")
        print(f"[model] backbone={model.backbone.default_cfg['architecture']}")

    return model