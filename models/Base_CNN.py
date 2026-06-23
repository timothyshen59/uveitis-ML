import torch
import torch.nn as nn
import torchvision.models as models


class Base_CNN(nn.Module):
    def __init__(self, num_zones=10, freeze_backbone=False):
        super().__init__()
        backbone = models.resnet50(pretrained=True)
        self.backbone = nn.Sequential(*list(backbone.children())[:-1])  # (B, 2048, 1, 1)

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(2048, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_zones)
        )

    def forward(self, x):
        feats = self.backbone(x)    # (B, 2048, 1, 1)
        return self.head(feats)     # (B, num_zones)


def load_model(model_path, mcfg, device):
    model = Base_CNN(
        num_zones=mcfg["num_zones"],
        freeze_backbone=mcfg.get("freeze_backbone", False),
    ).to(device)

    if model_path:
        state = torch.load(model_path, map_location=device)
        if isinstance(state, dict) and "model" in state:
            state = state["model"]
        model.load_state_dict(state)
        print(f"[model] Loaded weights from: {model_path}")
    else:
        print("[model] Starting from ImageNet pretrained ResNet50 backbone.")

    return model