# models/hybrid_model.py

import torch
import torch.nn as nn
import timm


class HybridModel(nn.Module):
    def __init__(self, cnn_backbone, proj_dim=64, num_classes=2, num_zones=10, freeze_backbones=True):
        super().__init__()

        self.vit = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=0, global_pool='')
        self.cnn = timm.create_model(cnn_backbone, pretrained=True, num_classes=0, global_pool='')

        if freeze_backbones:
            for param in self.vit.parameters():
                param.requires_grad = False
            for param in self.cnn.parameters():
                param.requires_grad = False
            print("[model] backbones frozen")

        vit_dim = self.vit.num_features                 
        cnn_dim = self.cnn.num_features

        self.cnn_to_shared   = nn.Linear(cnn_dim, proj_dim)
        self.patch_to_shared = nn.Linear(vit_dim, proj_dim)
        self.cls_to_shared   = nn.Linear(vit_dim, proj_dim)

        self.zone_embed = nn.Embedding(num_zones, proj_dim)
        self.zone_norm  = nn.LayerNorm(proj_dim)
        self.mha        = nn.MultiheadAttention(embed_dim=proj_dim, num_heads=4, batch_first=True)

        self.vit_dropout = nn.Dropout(0.4)
        self.cnn_dropout = nn.Dropout(0.4)

        self.mlp = nn.Sequential(
            nn.Linear(proj_dim * 2, 128),
            nn.GELU(),
            nn.Linear(128, 256),
            nn.GELU(),
            nn.Linear(256, 64),
            nn.GELU(),
            nn.Dropout(0.4),
        )
        self.head = nn.Linear(64, num_classes)

        self._init_weights()

    def _init_weights(self):
        for proj in [self.cnn_to_shared, self.patch_to_shared, self.cls_to_shared]:
            nn.init.kaiming_normal_(proj.weight, mode='fan_out', nonlinearity='relu')
            nn.init.zeros_(proj.bias)
        for layer in self.mlp:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='relu')
                nn.init.zeros_(layer.bias)
        nn.init.normal_(self.head.weight, std=0.01)
        nn.init.zeros_(self.head.bias)

    def forward(self, full_image, zone_imgs):
        # CLS + Patch Tokens
        vit_out      = self.vit(full_image)                         # (B, 197, 768)
        cls_token    = vit_out[:, 0, :]                             # (B, 768)
        patch_tokens = vit_out[:, 1:, :]                           # (B, 196, 768)

        global_proj = self.vit_dropout(self.cls_to_shared(cls_token))   # (B, proj_dim)
        patch_proj  = self.patch_to_shared(patch_tokens)                # (B, 196, proj_dim)


        B, num_zones, C, H, W = zone_imgs.shape
        zone_imgs      = zone_imgs.view(B * num_zones, C, H, W)
        feature_maps   = self.cnn(zone_imgs)
        local_features = feature_maps.mean(dim=(-2, -1))                # (B*num_zones, cnn_dim)
        local_features = self.cnn_to_shared(local_features)             # (B*num_zones, proj_dim)
        local_features = local_features.view(B, num_zones, -1)          # (B, num_zones, proj_dim)

        zone_ids       = torch.arange(num_zones, device=local_features.device)
        local_features = local_features + self.zone_embed(zone_ids).unsqueeze(0).expand(B, -1, -1)
        attn_out, _    = self.mha(local_features, patch_proj, patch_proj)
        local_features = self.zone_norm(local_features + attn_out)      # (B, num_zones, proj_dim)

        predictions = []
        for i in range(num_zones):
            zone_input = torch.cat([
                self.cnn_dropout(local_features[:, i, :]),  
                global_proj,                               
            ], dim=1)                                       # (B, proj_dim * 2)
            predictions.append(self.head(self.mlp(zone_input)))

        return torch.stack(predictions, dim=1)              # (B, num_zones, num_classes)


def load_model(model_path, model_cfg, device):
    model = HybridModel(
        cnn_backbone=model_cfg["cnn_backbone"],
        proj_dim=model_cfg["proj_dim"],
        num_classes=model_cfg["num_classes"],
        num_zones=model_cfg["num_zones"],
        freeze_backbones=model_cfg["freeze_backbone"],
    ).to(device)

    if model_path:
        state = torch.load(model_path, map_location=device)
        if isinstance(state, dict) and "model" in state:
            state = state["model"]
        model.load_state_dict(state)
        print(f"[model] loaded weights from {model_path}")
    else:
        print(f"[model] fresh init | cnn={model_cfg['cnn_backbone']} proj_dim={model_cfg['proj_dim']}")

    return model