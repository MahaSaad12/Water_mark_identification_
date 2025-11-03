# build_index.py
# Build a retrieval index from REFERENCE MASKS.
# Run (no args):  python build_index.py
# Or override:    python build_index.py --refs "pred_masks\\briquet\\masks" --out "index\\briquet_features.npz"

from __future__ import annotations
import argparse, csv, json, os
from pathlib import Path
from typing import List, Tuple

import numpy as np
import cv2

# ----------------------- Config (light) -----------------------
IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}

# Feature settings
DS_SIZE  = 128   # downsampled mask edge size (DS_SIZE^2 dims)
RAD_BINS = 64    # FFT radial profile bins

# ----------------------- Utils -----------------------
def list_images(root: Path) -> List[Path]:
    return sorted([p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMG_EXTS])

def load_mask_gray(path: Path) -> np.ndarray:
    m = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if m is None:
        raise RuntimeError(f"Failed to read: {path}")
    # binarize to {0,255}
    if m.max() > 1:
        _, m = cv2.threshold(m, 127, 255, cv2.THRESH_BINARY)
    return m

def feature_hu_moments(mask: np.ndarray) -> np.ndarray:
    """7 Hu moments (log(+abs), sign preserved)."""
    mom = cv2.moments((mask > 0).astype(np.uint8))
    hu  = cv2.HuMoments(mom).flatten()
    hu  = np.sign(hu) * np.log1p(np.abs(hu))
    return hu.astype(np.float32)  # [7]

def feature_fft_radial(mask: np.ndarray, bins: int = RAD_BINS) -> np.ndarray:
    """Rotation-tolerant-ish radial profile of FFT magnitude."""
    f   = np.fft.fft2(mask.astype(np.float32))
    mag = np.abs(np.fft.fftshift(f))
    h, w = mag.shape
    cy, cx = h // 2, w // 2
    y, x = np.ogrid[:h, :w]
    r = np.sqrt((y - cy) ** 2 + (x - cx) ** 2)
    r_norm = r / (r.max() + 1e-6)

    prof = np.zeros(bins, dtype=np.float32)
    idx  = np.minimum((r_norm * bins).astype(np.int32), bins - 1)
    for b in range(bins):
        sel = (idx == b)
        if sel.any():
            prof[b] = mag[sel].mean()
    if prof.max() > 0:
        prof /= (prof.max() + 1e-6)
    return prof.astype(np.float32)  # [bins]

def feature_downsample(mask: np.ndarray, size: int = DS_SIZE) -> np.ndarray:
    """Coarse normalized shape descriptor."""
    dm = cv2.resize(mask, (size, size), interpolation=cv2.INTER_AREA)
    v  = (dm > 127).astype(np.float32).ravel()
    n  = np.linalg.norm(v)
    if n > 0:
        v /= n
    return v.astype(np.float32)  # [size*size]

def extract_feature(mask: np.ndarray) -> np.ndarray:
    hu  = feature_hu_moments(mask)           # 7
    rad = feature_fft_radial(mask, RAD_BINS) # RAD_BINS
    dsm = feature_downsample(mask, DS_SIZE)  # DS_SIZE^2
    return np.concatenate([hu, rad, dsm], axis=0).astype(np.float32)

def build_index(ref_masks_dir: Path) -> Tuple[np.ndarray, List[str]]:
    paths = list_images(ref_masks_dir)
    print(f"Found {len(paths)} mask files under: {ref_masks_dir}")
    if not paths:
        raise RuntimeError(f"No reference masks found in: {ref_masks_dir}")
    feats, names = [], []
    for p in paths:
        m = load_mask_gray(p)
        feats.append(extract_feature(m))
        names.append(str(p.resolve()))
    feats = np.stack(feats, axis=0).astype(np.float32)
    return feats, names

def save_index(
    feats: np.ndarray,
    names: List[str],
    out_npz: Path,
    manifest_csv: Path,
    meta_json: Path,
    refs_dir: Path
) -> None:
    out_npz.parent.mkdir(parents=True, exist_ok=True)
    manifest_csv.parent.mkdir(parents=True, exist_ok=True)
    meta_json.parent.mkdir(parents=True, exist_ok=True)

    # features + paths
    np.savez_compressed(out_npz, features=feats, paths=np.array(names, dtype=object))

    # simple manifest
    with open(manifest_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "path", "basename"])
        for i, p in enumerate(names):
            w.writerow([i, p, os.path.basename(p)])

    # meta for reproducibility
    meta = {
        "refs_dir": str(refs_dir),
        "npz_path": str(out_npz),
        "csv_path": str(manifest_csv),
        "ds_size": DS_SIZE,
        "rad_bins": RAD_BINS,
        "feature_dim": int(feats.shape[1]),
        "num_items": int(feats.shape[0]),
        "notes": "Feature = [Hu(7), FFT radial profile(rad_bins), downsampled mask(ds_size^2)]"
    }
    with open(meta_json, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

# ----------------------- Main -----------------------
def main():
    # Resolve defaults relative to project root
    proj_root = Path(__file__).resolve().parent  # folder containing this file
    # If your structure is pythonProject2/<this file>, and pred_masks is at the same level:
    default_refs_abs = (proj_root.parent / "pred_masks" / "briquet" / "masks").resolve()
    default_out_abs  = (proj_root.parent / "index" / "briquet_features.npz").resolve()
    default_csv_abs  = (proj_root.parent / "index" / "briquet_manifest.csv").resolve()
    default_meta_abs = (proj_root.parent / "index" / "briquet_meta.json").resolve()

    ap = argparse.ArgumentParser("Build retrieval index from reference MASKS")
    ap.add_argument("--refs", type=str, default=str(default_refs_abs),
                    help="Folder with reference mask images")
    ap.add_argument("--out",  type=str, default=str(default_out_abs),
                    help="Output NPZ (features+paths)")
    args = ap.parse_args()

    refs_dir = Path(args.refs).resolve()
    out_npz  = Path(args.out).resolve()
    manifest = Path(str(out_npz).replace(".npz", "_manifest.csv"))
    meta     = Path(str(out_npz).replace(".npz", "_meta.json"))

    print(f"Refs (resolved): {refs_dir}")
    print(f"Out  (resolved): {out_npz}")

    feats, names = build_index(refs_dir)
    print(f"Built features: {feats.shape} for {len(names)} masks")

    save_index(feats, names, out_npz, manifest, meta, refs_dir)
    print("\n✅ Index saved:")
    print(f"- Features: {out_npz}")
    print(f"- Manifest: {manifest}")
    print(f"- Meta    : {meta}")

if __name__ == "__main__":
    main()
