import torch
import torch.nn as nn
import timm
from transformers import AutoModel

from huggingface_hub import login, hf_hub_download
login(token="")


class RetViT(nn.Module): 
    def __init__(self, vit_backbone, num_classes = 2, num_zones = 10, freeze_backbone = True):
        super().__init__()
        self.vit = timm.create_model(
            'vit_large_patch16_224', 
            pretrained=False, 
            num_classes=0, 
            global_pool='avg',
        )
        
        checkpoint_path = "/home/tim/uveitis-research/model_weights/RETFound_mae_meh.pth"

        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        checkpoint_model = checkpoint['model']
        state_dict = self.vit.state_dict() 

         # strip head keys that won't match
        for k in ['head.weight', 'head.bias']:
            if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                print(f"[model] Removing key {k} from pretrained checkpoint")
                del checkpoint_model[k]

        msg = self.vit.load_state_dict(checkpoint_model, strict=False)
        print(f"[model] Loaded RETFound weights: {msg}")

        
        if freeze_backbone: 
            for param in self.vit.parameters(): 
                param.requires_grad = False 
    
        self.num_classes = num_classes 
        self.num_zones = num_zones
        
        self.head = nn.Sequential(
            nn.Linear(1024, 256),
            nn.GELU(),
            # nn.Dropout(0.3),
            nn.Linear(256, num_zones)   # 10 * 2 = 20
        )

        
    def forward(self, x): 
        feats = self.vit(x) 
        output = self.head(feats)                                  # (B, 20)

        return output 
       
        
    
    

def load_model(model_path, mcfg, device): 
    model = RetViT(
        vit_backbone = mcfg["vit_backbone"], #Hardcoded and need to make sure path matches absolutely 
        num_classes=mcfg["num_classes"],
        num_zones=mcfg["num_zones"],
    ).to(device)
    
    if model_path:
        state = torch.load(model_path, map_location=device)
        if isinstance(state, dict) and "model" in state:
            state = state["model"]
        model.load_state_dict(state)
        print(f"[model] Loaded weights from: {model_path}")
    else:
        print("[model] Starting from RETFound pretrained backbone.")

    return model
    
        