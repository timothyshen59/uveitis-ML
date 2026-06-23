import torch 
import torch.nn as nn 
import timm 
from huggingface_hub import hf_hub_download

class HybridModel(nn.Module): 
    def __init__(self, cnn_backbone, proj_dim=64, num_classes=2, num_zones=10, freeze_backbones=True): 
        super().__init__() 
        self.vit = timm.create_model(
            'vit_base_patch16_224', 
            pretrained=True,       
            num_classes=0,
            global_pool='', 
        )

        self.cnn = timm.create_model(
            cnn_backbone, 
            pretrained=True, 
            num_classes=0, 
            global_pool=''
        ) 
        
        #Rewrite to helper function and load weghts optionall for CNN and ViT
          # # Load RETFound weights
        # checkpoint_path = "/home/tim/uveitis-research/model_weights/RETFound_mae_meh.pth"
        # checkpoint       = torch.load(checkpoint_path, map_location='cpu')
        # checkpoint_model = checkpoint['model']
        # state_dict       = self.vit.state_dict()
        # for k in ['head.weight', 'head.bias']:
        #     if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
        #         del checkpoint_model[k]
        # msg = self.vit.load_state_dict(checkpoint_model, strict=False)
        # print(f"[model] Loaded RETFound weights: {msg}")

        self.zone_embed = nn.Embedding(num_zones, proj_dim)
        self.zone_norm = nn.LayerNorm(proj_dim)
        self.attn_gate = nn.Parameter(torch.tensor(0.0))


    
        self.vit_dropout = nn.Dropout(0.4) 
        self.cnn_dropout = nn.Dropout(0.4) 
        
        if freeze_backbones: 
            print("FREEZING BACKBONES")
            for param in self.vit.parameters():
                param.requires_grad = False
            for param in self.cnn.parameters():
                param.requires_grad = False
                
        vit_dim = 768                  # RETFound ViT-L CLS token dim
        cnn_dim = self.cnn.num_features 

        self.mha = nn.MultiheadAttention(embed_dim=proj_dim, num_heads=4, batch_first=True)
        
        self.cnn_to_shared = nn.Linear(cnn_dim, proj_dim) 
        self.patch_to_shared = nn.Linear(vit_dim, proj_dim) 
        self.cls_to_shared = nn.Linear(vit_dim, proj_dim)
        
        fused_dim = proj_dim    # CLS + zone projection
        
        self.mlp = nn.Sequential(
            nn.Linear(fused_dim, 128), 
            nn.GELU(), 
            nn.Linear(128, 256 ), 
            nn.GELU(),
            nn.Linear(256, 64), 
            nn.GELU(),
            nn.Dropout(0.4), 
        )
        
        self.head = nn.Linear(64, num_classes)
    
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
        # Extract CLS token only — index 0 is CLS, rest are patch tokens
        # vit_out        = self.vit(full_image)          # (B, 197, 1024)
        # cls_token = vit_out[:,0,:]
        # patch_tokens = vit_out[:,1:]
        
 
        
        # patch_proj = self.patch_to_shared(patch_tokens)
        # global_proj = self.cls_to_shared(cls_token) 
        # global_proj = self.vit_dropout(global_proj)
    
        B, num_zones, C, H, W = zone_imgs.shape 
        zone_imgs    = zone_imgs.view(B * num_zones, C, H, W)
        feature_maps = self.cnn(zone_imgs) 
        local_features = feature_maps.mean(dim=(-2, -1))  # GAP → (B*num_zones, cnn_dim)
        local_features = self.cnn_to_shared(local_features)    # (B*num_zones, proj_dim)
        local_features = local_features.view(B, num_zones, -1) 
        zone_ids = torch.arange(num_zones, device=local_features.device)
        zone_emb = self.zone_embed(zone_ids).unsqueeze(0).expand(B, -1, -1)

        local_features = local_features + zone_emb
        
        # local_features , _ = self.mha(local_features, patch_proj, patch_proj) 
        # local_features = self.zone_norm(local_features + attn_out)
        
        predictions = [] 
        for i in range(num_zones): 
            
            # zone_input = torch.cat([
            #     self.cnn_dropout(local_features[:, i, :]),               # (B, proj_dim) local zone features
            #     global_proj                     # (B, 1024)     global CLS context
            # ], dim=1)
            
            zone_input = self.cnn_dropout(local_features[:, i, :])
            zone_input = self.mlp(zone_input)
            predictions.append(self.head(zone_input))
            
        return torch.stack(predictions, dim=1)         # (B, num_zones, num_classes)
             

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
        print(f"[model] Loaded weights from: {model_path}")
    else:
        print("[model] Starting from ViT-B + pretrained CNN backbones (cross-attention, no-MHA,MLP, 64)")
        print(f"[model] cnn_backbone={model_cfg['cnn_backbone']}")

    return model