import os, re, glob
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
from torchvision import transforms

# ===== CONFIG =====
IMAGES_DIR = "paired/images"
MASKS_DIR  = "paired/masks"
IMG_SIZE   = 256
BATCH_SIZE = 6
EPOCHS     = 20
LR         = 1e-4
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"

# ---------- pairing helpers ----------
def all_files(root):
    files = []
    for ext in ("*.png","*.jpg","*.jpeg","*.tif","*.tiff"):
        files.extend(glob.glob(os.path.join(root, "**", ext), recursive=True))
    return sorted(files)

def norm_base(path):
    base = os.path.splitext(os.path.basename(path))[0].lower()
    base = re.sub(r"(_mask|-mask|\.mask|_label|-label|\.label)$", "", base)
    base = re.sub(r"^(briquet_|img_|image_|scan_|page_)", "", base)
    return base

def build_pairs(img_dir, msk_dir):
    imgs  = all_files(img_dir)
    msks  = all_files(msk_dir)
    mask_by_key = {}
    for m in msks:
        mask_by_key.setdefault(norm_base(m), m)
    pairs = []
    for i in imgs:
        key = norm_base(i)
        m = mask_by_key.get(key)
        if m: pairs.append((i, m))
    return pairs

pairs = build_pairs(IMAGES_DIR, MASKS_DIR)
if not pairs:
    raise RuntimeError("No image–mask pairs found in paired/ .")

print(f"Found {len(pairs)} matched pairs for training.")

# ---------- dataset ----------
class WMDS(Dataset):
    def __init__(self, pairs, size=256):
        self.pairs = pairs
        self.tf_img  = transforms.Compose([
            transforms.Resize((size, size)),
            transforms.ToTensor()
        ])
        self.tf_mask = transforms.Compose([
            transforms.Resize((size, size), interpolation=Image.NEAREST),
            transforms.ToTensor()
        ])
    def __len__(self): return len(self.pairs)
    def __getitem__(self, idx):
        img_p, msk_p = self.pairs[idx]
        img  = Image.open(img_p).convert("RGB")
        msk  = Image.open(msk_p).convert("L")
        img  = self.tf_img(img)
        msk  = self.tf_mask(msk)
        # binarize mask (keep shape [1,H,W])
        msk  = (msk > 0.5).float()
        return img, msk

dataset = WMDS(pairs, IMG_SIZE)
loader  = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

# ---------- U-Net (fixed channels) ----------
def CBR(in_c, out_c):
    return nn.Sequential(
        nn.Conv2d(in_c, out_c, 3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_c, out_c, 3, padding=1),
        nn.ReLU(inplace=True)
    )

class UNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc1 = CBR(3, 64)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = CBR(64, 128)
        self.pool2 = nn.MaxPool2d(2)
        self.bott = CBR(128, 256)
        self.up2  = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2 = CBR(128 + 128, 128)   # <-- fixed: 256 in
        self.up1  = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1 = CBR(64 + 64, 64)      # <-- fixed: 128 in
        self.out  = nn.Conv2d(64, 1, 1)
    def forward(self, x):
        e1 = self.enc1(x)          # 64
        e2 = self.enc2(self.pool1(e1))  # 128
        b  = self.bott(self.pool2(e2))  # 256
        d2 = self.up2(b)                 # 128
        d2 = torch.cat([d2, e2], dim=1)  # 256
        d2 = self.dec2(d2)               # 128
        d1 = self.up1(d2)                # 64
        d1 = torch.cat([d1, e1], dim=1)  # 128
        d1 = self.dec1(d1)               # 64
        return torch.sigmoid(self.out(d1))

model = UNet().to(DEVICE)
criterion = nn.BCELoss()
optimzr   = optim.Adam(model.parameters(), lr=LR)

# ---------- train ----------
for epoch in range(EPOCHS):
    model.train()
    running = 0.0
    for imgs, msks in loader:
        imgs, msks = imgs.to(DEVICE), msks.to(DEVICE)
        preds = model(imgs)
        loss  = criterion(preds, msks)
        optimzr.zero_grad(); loss.backward(); optimzr.step()
        running += loss.item()
    print(f"Epoch {epoch+1}/{EPOCHS}  Loss: {running/len(loader):.4f}")

torch.save(model.state_dict(), "../unet_watermark.pth")
print("✅ Training complete. Saved unet_watermark.pth")
