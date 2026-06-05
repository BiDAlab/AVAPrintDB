import torch
import clip

import torchvision.transforms.v2 as T

def get_clip_backbone( model_name="ViT-L/14", freeze=True, device="cpu", return_preprocess=False, **kwargs):
    model, preprocess = clip.load(model_name, device=device)

    if freeze:
        for p in model.parameters():
            p.requires_grad = False
        model = model.eval()
    model = model.to(device)

    if return_preprocess:
        return model, preprocess
    return model

def get_clip_transform():
    transforms = T.Compose([
        T.ToImage(),           # Converts NumPy or other formats to image tensor
        T.Resize((224, 224)),  # Works on tensors, PIL, or NumPy
        T.ToDtype(torch.float32, scale=True),  # Converts [0, 255] → [0.0, 1.0]
        T.Normalize(mean=(0.48145466, 0.4578275, 0.40821073),
                    std=(0.26862954, 0.26130258, 0.27577711)),
    ])
    return transforms

def run_inference_clip(model, x):
    BS, T, C, H, W = x.shape
    x = x.view(BS * T, C, H, W)
    with torch.no_grad():
        feats = model.encode_image(x)
    return feats.view(BS, T, -1)

def get_dinov2_backbone(model_name="dinov2_vitl14_reg", freeze=True, device="cpu", **kwargs):
    model = torch.hub.load('facebookresearch/dinov2', model_name)

    if freeze:
        for p in model.parameters():
            p.requires_grad = False
        model = model.eval()
    model = model.to(device)

    return model

def get_dinov2_transform():
    transforms = T.Compose([
        T.ToImage(),           # Converts NumPy or other formats to image tensor
        T.Resize((224, 224)),  # Works on tensors, PIL, or NumPy
        T.ToDtype(torch.float32, scale=True),  # Converts [0, 255] → [0.0, 1.0]
        T.Normalize(mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225)),
    ])
    return transforms

def run_inference_dinov2(model, x):
    BS, T, C, H, W = x.shape
    x = x.view(BS * T, C, H, W)
    with torch.no_grad():
        feats = model(x)  # [BS*T, D]
    return feats.view(BS, T, -1)



def get_backbone_and_transforms(model_name="CLIP", device="cpu", **kwargs):
    if model_name == "CLIP":
        return get_clip_backbone(freeze=True, device=device, **kwargs), get_clip_transform(), run_inference_clip
    elif model_name == "DINO":
        return get_dinov2_backbone(freeze=True, device=device, **kwargs), get_dinov2_transform(), run_inference_dinov2
    else:
        raise ValueError(f"Unknown backbone: {model_name}")