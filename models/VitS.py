# Vit-Base Model: Pretrained on ImageNet 
import torch
import torch.nn as nn
import timm

#Model
class ViTBaseModel(nn.Module):
    def __init__(self, backbone, num_zones=10, hidden_dim=256, num_thresholds=2):
        super().__init__()
        self.num_thresholds = num_thresholds

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

        self.ordinal_heads = nn.ModuleList([
            nn.Linear(hidden_dim, self.num_thresholds)
            for _ in range(num_zones)
        ])

    def forward(self, x):
        feats = self.backbone(x)
        feats = self.mlp(feats)
        return torch.stack([head(feats) for head in self.ordinal_heads], dim=1)

def load_model(model_path, model_cfg, device):
    model = ViTBaseModel(
        backbone=model_cfg["backbone"],
        num_zones=model_cfg["num_zones"],
        hidden_dim=model_cfg["hidden_dim"],
        num_thresholds=model_cfg["num_classes"] - 1,  
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

#Ordinal Helpers 
def encode_ordinal(labels: torch.Tensor, num_classes: int = 3) -> torch.Tensor:
    B, Z = labels.shape
    thresholds = num_classes - 1
    out = torch.zeros(B, Z, thresholds, device=labels.device)
    
    for k in range(thresholds):
        out[:, :, k] = (labels > k).float()
        
    return out


def ordinal_to_class(logits: torch.Tensor) -> torch.Tensor:
    return torch.sum(torch.sigmoid(logits) > 0.5, dim=-1)




