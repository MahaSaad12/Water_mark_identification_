# make_pseudomasks_full.py
# Build pseudo-masks for watermark dataset parts:
# - A_classification/{train,test}
# - B_cross_domain_plus
# - (optional) briquet_synthetic

import argparse
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

# -----------------------
# Utilities
# -----------------------
def list_images(root: Path):
    return [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMG_EXTS]

def load_gray(path: Path):
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise RuntimeError(f"Failed to read: {path}")
    return img

def best_match_ncc(img, templ, scales=(0.4,0.5,0.6,0.7,0.8,0.9,1.0)):
    """Multi-scale TM_CCOEFF_NORMED; returns (score, top_left, bottom_right, scaled_template)"""
    h, w = img.shape[:2]
    best = (-1.0, (0,0), (0,0), None)
    for s in scales:
        th = max(8, int(templ.shape[0] * s))
        tw = max(8, int(templ.shape[1] * s))
        if th >= h or tw >= w:  # skip too large
            continue
        t = cv2.resize(templ, (tw, th), interpolation=cv2.INTER_AREA)
        res = cv2.matchTemplate(img, t, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(res)
        if max_val > best[0]:
            tl = max_loc
            br = (max_loc[0] + tw, max_loc[1] + th)
            best = (max_val, tl, br, t)
    return best

def refine_mask(img_gray, tl, br):
    """Edge-based refinement inside the matched box."""
    (x1, y1), (x2, y2) = tl, br
    h, w = img_gray.shape
    x1 = max(0, x1); y1 = max(0, y1); x2 = min(w, x2); y2 = min(h, y2)
    if x2 <= x1 or y2 <= y1:
        return np.zeros_like(img_gray, dtype=np.uint8)

    roi = img_gray[y1:y2, x1:x2]

    edges = cv2.Canny(roi, 50, 150)
    edges = cv2.dilate(edges, np.ones((3,3), np.uint8), iterations=1)

    mask = np.zeros_like(roi, dtype=np.uint8)
    mask[edges > 0] = 255
    mask = cv2.medianBlur(mask, 3)

    full = np.zeros_like(img_gray, dtype=np.uint8)
    full[y1:y2, x1:x2] = mask

    k = np.ones((3,3), np.uint8)
    full = cv2.morphologyEx(full, cv2.MORPH_OPEN, k, iterations=1)
    full = cv2.morphologyEx(full, cv2.MORPH_CLOSE, k, iterations=1)
    return full

def ensure_dirs(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def save_pair(ip: Path, mask: np.ndarray, out_img: Path, out_msk: Path):
    out_name = ip.stem + ".png"
    # store images as PNG to guarantee 1:1 pairing by stem
    Image.open(ip).convert("RGB").save(out_img/out_name)
    Image.fromarray(mask).save(out_msk/out_name)

def find_reference(ref_root: Path, cls: str) -> Path | None:
    # expected: briquet_<class>.<ext>
    cand = list(ref_root.glob(f"briquet_{cls}.*"))
    return cand[0] if cand else None

# -----------------------
# Builders
# -----------------------
def build_pairs_from_Aclassification(base: Path, out_base: Path, ref_root: Path,
                                     splits=("train","test"), ncc_thresh=0.25,
                                     scales=(0.4,0.5,0.6,0.7,0.8,0.9,1.0)):
    for split in splits:
        in_root = base / "A_classification" / split
        out_img = out_base / split / "images"
        out_msk = out_base / split / "masks"
        ensure_dirs(out_img); ensure_dirs(out_msk)

        imgs = list_images(in_root)
        if not imgs:
            print(f"[WARN] No images found under {in_root}")
            continue

        for ip in tqdm(imgs, desc=f"A_classification/{split}"):
            # class is first directory under split
            try:
                cls = ip.relative_to(in_root).parts[0]
            except Exception:
                cls = ip.parent.name

            ref_path = find_reference(ref_root, cls)
            if ref_path is None:
                print(f"[MISS REF] briquet_{cls}.* -> skipping {ip.name}")
                continue

            img = load_gray(ip)
            templ = load_gray(ref_path)
            score, tl, br, _ = best_match_ncc(img, templ, scales=scales)

            if score < ncc_thresh:
                # fallback rectangle so the model still gets a positive signal
                mask = np.zeros_like(img, dtype=np.uint8)
                x1, y1 = tl; x2, y2 = br
                cv2.rectangle(mask, (x1,y1), (x2,y2), 255, -1)
            else:
                mask = refine_mask(img, tl, br)

            save_pair(ip, mask, out_img, out_msk)

def build_pairs_from_B(base: Path, out_base: Path, ref_root: Path,
                       ncc_thresh=0.25, scales=(0.4,0.5,0.6,0.7,0.8,0.9,1.0)):
    in_root = base / "B_cross_domain_plus"
    out_img = out_base / "B_cross_domain_plus" / "images"
    out_msk = out_base / "B_cross_domain_plus" / "masks"
    ensure_dirs(out_img); ensure_dirs(out_msk)

    imgs = list_images(in_root)
    if not imgs:
        print(f"[WARN] No images found under {in_root}")
        return

    for ip in tqdm(imgs, desc="B_cross_domain_plus"):
        cls = ip.parent.name  # class folder name
        ref_path = find_reference(ref_root, cls)
        if ref_path is None:
            print(f"[MISS REF] briquet_{cls}.* -> skipping {ip.name}")
            continue

        img = load_gray(ip)
        templ = load_gray(ref_path)
        score, tl, br, _ = best_match_ncc(img, templ, scales=scales)

        if score < ncc_thresh:
            mask = np.zeros_like(img, dtype=np.uint8)
            x1, y1 = tl; x2, y2 = br
            cv2.rectangle(mask, (x1,y1), (x2,y2), 255, -1)
        else:
            mask = refine_mask(img, tl, br)

        save_pair(ip, mask, out_img, out_msk)

def build_pairs_from_synthetic(base: Path, out_base: Path, ref_root: Path,
                               ncc_thresh=0.25, scales=(0.4,0.5,0.6,0.7,0.8,0.9,1.0)):
    in_root = base / "briquet_synthetic"
    out_img = out_base / "briquet_synthetic" / "images"
    out_msk = out_base / "briquet_synthetic" / "masks"
    ensure_dirs(out_img); ensure_dirs(out_msk)

    imgs = list_images(in_root)
    if not imgs:
        print(f"[WARN] No images found under {in_root}")
        return

    for ip in tqdm(imgs, desc="briquet_synthetic"):
        # synthetic files are usually named like briquet_<class>_SOMETHING.png
        stem = ip.stem.lower()
        cls = stem.replace("briquet_", "").split("_")[0] if "briquet_" in stem else ip.parent.name
        ref_path = find_reference(ref_root, cls)
        if ref_path is None:
            print(f"[MISS REF] briquet_{cls}.* -> skipping {ip.name}")
            continue

        img = load_gray(ip)
        templ = load_gray(ref_path)
        score, tl, br, _ = best_match_ncc(img, templ, scales=scales)

        if score < ncc_thresh:
            mask = np.zeros_like(img, dtype=np.uint8)
            x1, y1 = tl; x2, y2 = br
            cv2.rectangle(mask, (x1,y1), (x2,y2), 255, -1)
        else:
            mask = refine_mask(img, tl, br)

        save_pair(ip, mask, out_img, out_msk)

# -----------------------
# CLI
# -----------------------
def main():
    ap = argparse.ArgumentParser("Build pseudo-masks for watermark dataset")
    ap.add_argument("--base", default=r"C:\Users\maha_\Job_search\pythonProject2\Datasets\watermark",
                    help="Dataset base folder (contains A_classification, B_cross_domain_plus, ...)")
    ap.add_argument("--refs", default=None,
                    help="Folder with reference engravings (default: <base>/briquet_engraving)")
    ap.add_argument("--out",  default=r"C:\Users\maha_\Job_search\pythonProject2\paired",
                    help="Output base folder for paired {images,masks}")
    ap.add_argument("--include", nargs="+",
                    choices=["A", "B", "S"], default=["A", "B"],
                    help="Which parts to process: A=A_classification, B=B_cross_domain_plus, S=briquet_synthetic")
    ap.add_argument("--ncc-thresh", type=float, default=0.25,
                    help="Min NCC score to accept refined mask (otherwise rectangle fallback)")
    ap.add_argument("--scales", type=float, nargs="+",
                    default=[0.4,0.5,0.6,0.7,0.8,0.9,1.0],
                    help="Template scales to search")
    args = ap.parse_args()

    base = Path(args.base)
    refs = Path(args.refs) if args.refs else (base / "briquet_engraving")
    outb = Path(args.out)

    if "A" in args.include:
        build_pairs_from_Aclassification(base, outb, refs,
                                         splits=("train","test"),
                                         ncc_thresh=args.ncc_thresh,
                                         scales=tuple(args.scales))
    if "B" in args.include:
        build_pairs_from_B(base, outb, refs,
                           ncc_thresh=args.ncc_thresh,
                           scales=tuple(args.scales))
    if "S" in args.include:
        build_pairs_from_synthetic(base, outb, refs,
                                   ncc_thresh=args.ncc_thresh,
                                   scales=tuple(args.scales))

    print(f"\n✅ Done. Paired data written to: {outb}")

if __name__ == "__main__":
    main()
