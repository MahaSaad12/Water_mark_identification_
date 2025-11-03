# predicat_masks.py — Tiny-UNet inference for reference Briquet engravings
# Run: python predicat_masks.py  ✅ No command arguments needed

import os
from pathlib import Path
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

# ========= DEFAULT SETTINGS (EDIT IF NEEDED) =========
IMAGES_DIR = r"Datasets\watermark\briquet_engraving"  # Reference images location
OUT_DIR    = r"pred_masks\briquet"                    # Output folder for masks
WEIGHTS    = r"runs\best_model.pt"                    # Trained model checkpoint
IMG_SIZE   = 512
THRESH     = 0.5
DEVICE     = "cpu"
CLEAR_OUT  = True
# ====================================================

# ---------- Tiny UNet (must match training model) ----------
def conv_bn_relu(cin, cout):
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1, bias=False),
        nn.BatchNorm2d(cout),
        nn.ReLU(inplace=True),
    )

class TinyUNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.c1 = conv_bn_relu(1, 16); self.p1 = nn.MaxPool2d(2)
        self.c2 = conv_bn_relu(16, 32); self.p2 = nn.MaxPool2d(2)
        self.c3 = conv_bn_relu(32, 64); self.p3 = nn.MaxPool2d(2)
        self.c4 = conv_bn_relu(64, 64)

        self.u3 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.dc3 = conv_bn_relu(64+64, 64)
        self.u2 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.dc2 = conv_bn_relu(64+32, 32)
        self.u1 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.dc1 = conv_bn_relu(32+16, 16)
        self.head = nn.Conv2d(16, 1, 1)

    def forward(self, x):
        e1 = self.c1(x)
        e2 = self.c2(self.p1(e1))
        e3 = self.c3(self.p2(e2))
        b  = self.c4(self.p3(e3))

        d3 = self.u3(b)
        if d3.shape[-2:] != e3.shape[-2:]:
            d3 = F.interpolate(d3, size=e3.shape[-2:], mode="bilinear", align_corners=False)
        d3 = self.dc3(torch.cat([d3, e3], 1))

        d2 = self.u2(d3)
        if d2.shape[-2:] != e2.shape[-2:]:
            d2 = F.interpolate(d2, size=e2.shape[-2:], mode="bilinear", align_corners=False)
        d2 = self.dc2(torch.cat([d2, e2], 1))

        d1 = self.u1(d2)
        if d1.shape[-2:] != e1.shape[-2:]:
            d1 = F.interpolate(d1, size=e1.shape[-2:], mode="bilinear", align_corners=False)
        d1 = self.dc1(torch.cat([d1, e1], 1))

        return self.head(d1)  # logits

# ----------------------------------------------------------
def load_model(weights):
    model = TinyUNet().to(DEVICE)
    state = torch.load(weights, map_location=DEVICE)
    if "state_dict" in state: state = state["state_dict"]
    model.load_state_dict(state)
    model.eval()
    return model

@torch.no_grad()
def infer_mask(model, path):
    orig = Image.open(path).convert("L")
    w0, h0 = orig.size
    inp = orig.resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)

    t = torch.from_numpy(np.array(inp)/255.0).float().unsqueeze(0).unsqueeze(0).to(DEVICE)
    prob = torch.sigmoid(model(t))[0,0].cpu().numpy()
    mask_small = (prob > THRESH).astype(np.uint8)*255
    mask = Image.fromarray(mask_small).resize((w0, h0), Image.NEAREST)

    return orig.convert("RGB"), mask

# ------------------ MAIN ------------------------------------------------
def main():
    img_dir = Path(IMAGES_DIR)
    out_img = Path(OUT_DIR) / "images"
    out_msk = Path(OUT_DIR) / "masks"
    out_img.mkdir(parents=True, exist_ok=True)
    out_msk.mkdir(parents=True, exist_ok=True)

    if CLEAR_OUT:
        for f in out_img.glob("*.png"): f.unlink()
        for f in out_msk.glob("*.png"): f.unlink()

    images = list(path for path in img_dir.rglob("*") if path.suffix.lower() in [".png",".jpg",".jpeg",".bmp",".tiff"])
    if not images:
        print(f"❌ No images found under: {img_dir}")
        return

    print(f"✅ Found {len(images)} reference images")
    print(f"📌 Loading model: {WEIGHTS}")
    model = load_model(WEIGHTS)

    for ip in tqdm(images, desc="Generating masks"):
        rgb, msk = infer_mask(model, ip)
        name = ip.stem + ".png"
        rgb.save(out_img / name)
        msk.save(out_msk / name)

    print("\n✅ Briquet mask prediction complete!")
    print(f"📍 Output: {OUT_DIR}")

if __name__ == "__main__":
    main()
