"""DepthPro — Monocular Depth Estimation (DPT + MiDaS-style)"""
import torch, torch.nn as nn, numpy as np, cv2
import torchvision.transforms as T
from PIL import Image
import argparse

class DPTHead(nn.Module):
    def __init__(self, in_ch=768, features=256):
        super().__init__()
        self.proj = nn.Conv2d(in_ch, features, 1)
        self.refine = nn.Sequential(
            nn.Conv2d(features, features//2, 3, padding=1), nn.ReLU(),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            nn.Conv2d(features//2, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 1, 1), nn.ReLU()
        )
    def forward(self, x): return self.refine(self.proj(x))

transform = T.Compose([T.Resize((384,384)), T.ToTensor(), T.Normalize([0.5],[0.5])])

@torch.inference_mode()
def estimate_depth(img_path: str, output_path: str = None, device: str = "cpu"):
    import timm
    encoder = timm.create_model("vit_base_patch16_384", pretrained=True, features_only=False)
    head = DPTHead(768, 256)
    encoder.eval().to(device); head.eval().to(device)
    img = Image.open(img_path).convert("RGB")
    orig_w, orig_h = img.size
    tensor = transform(img).unsqueeze(0).to(device)
    feats = encoder.forward_features(tensor)
    if feats.dim() == 3:
        b, n, c = feats.shape; h = w = int(n**0.5)
        feats = feats.permute(0,2,1).view(b, c, h, w)
    depth = head(feats)
    depth = torch.nn.functional.interpolate(depth, (orig_h, orig_w), mode="bilinear", align_corners=True)
    depth_np = depth.squeeze().cpu().numpy()
    depth_norm = ((depth_np - depth_np.min()) / (depth_np.max() - depth_np.min() + 1e-8) * 255).astype(np.uint8)
    colored = cv2.applyColorMap(depth_norm, cv2.COLORMAP_MAGMA)
    if output_path: cv2.imwrite(output_path, colored); print(f"Saved: {output_path}")
    return depth_np, colored

if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("--input", required=True); p.add_argument("--output", default="depth.png")
    a = p.parse_args(); estimate_depth(a.input, a.output)
