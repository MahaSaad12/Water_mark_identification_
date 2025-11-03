# compare_watermark.py
# Top-K retrieval of query masks against a reference index (NPZ)
# Run directly (paths are set in DEFAULTS below), or pass CLI args.

import argparse
from pathlib import Path
import csv
import json
import numpy as np
from tqdm import tqdm

# ---------- DEFAULTS (edit or override via CLI) ----------
DEFAULT_QUERIES = r"pred_masks\AllDatasets\masks"
DEFAULT_INDEX   = r"index\briquet_features.npz"

DEFAULT_OUT     = r"runs_match"
DEFAULT_TOPK    = 10
# ---------------------------------------------------------

def _choose_key(dct, candidates):
    """Return the first present key from candidates, else None."""
    for k in candidates:
        if k in dct:
            return k
    return None

def load_index(npz_path: Path):
    """Load index NPZ and return (feats [N,D], names [N], meta: dict|None)."""
    with np.load(npz_path, allow_pickle=True) as data:
        keys = set(data.keys())

        # Try robust key selection for features
        feat_key = _choose_key(data, ["feats", "features", "X", "embeddings"])
        name_key = _choose_key(data, ["names", "paths", "files", "stems", "ids"])

        if feat_key is None or name_key is None:
            raise KeyError(
                f"Could not find expected arrays in {npz_path}.\n"
                f"Available keys: {sorted(keys)}\n"
                f"Expected one of features in ['feats','features','X','embeddings'] "
                f"and names in ['names','paths','files','stems','ids']."
            )

        feats = np.asarray(data[feat_key]).astype(np.float32)
        names = np.asarray(data[name_key])

        # Optional: try meta keys saved separately or inside the npz
        meta = None
        meta_key = _choose_key(data, ["meta", "manifest"])
        if meta_key is not None:
            # some writers store a pickled dict/bytes; make a best effort
            try:
                maybe = data[meta_key].item()
                if isinstance(maybe, dict):
                    meta = maybe
            except Exception:
                pass

    # If there is a sidecar json manifest with the same stem, attach it
    json_sidecar = npz_path.with_suffix("").with_name(npz_path.stem + "_meta.json")
    if json_sidecar.exists():
        try:
            meta = json.loads(Path(json_sidecar).read_text(encoding="utf-8"))
        except Exception:
            pass

    return feats, names, meta

def list_masks(root: Path):
    exts = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp")
    return sorted([p for p in root.rglob("*") if p.suffix.lower() in exts])

def load_mask_binary(path: Path, target_size=None):
    from PIL import Image
    img = Image.open(path).convert("L")
    if target_size is not None:
        img = img.resize(target_size, Image.NEAREST)
    arr = (np.array(img) > 127).astype(np.float32)
    return arr

def coarse_mask_feature(mask: np.ndarray, ds=128, rad_bins=64):
    """
    Simple, fast, rotation-robust-ish descriptor:
      1) downsample mask to ds x ds
      2) compute radial profile (mean along concentric rings)
    """
    from scipy.ndimage import zoom
    # downsample to fixed size
    h, w = mask.shape
    z = zoom(mask, (ds / h, ds / w), order=0)
    # radial bins
    yy, xx = np.mgrid[0:ds, 0:ds]
    cy = (ds - 1) / 2.0
    cx = (ds - 1) / 2.0
    rr = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    rmax = rr.max()
    bins = np.linspace(0, rmax, rad_bins + 1)
    prof = np.zeros(rad_bins, dtype=np.float32)
    for b in range(rad_bins):
        sel = (rr >= bins[b]) & (rr < bins[b + 1])
        if sel.any():
            prof[b] = z[sel].mean()
    # L2 normalize
    n = np.linalg.norm(prof) + 1e-8
    return (prof / n).astype(np.float32)

def normalize_rows(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x, axis=1, keepdims=True) + 1e-8
    return x / n

def main():
    ap = argparse.ArgumentParser("Top-K retrieval of query masks against briquet index")
    ap.add_argument("--queries", default=DEFAULT_QUERIES, help="Folder with query masks")
    ap.add_argument("--index",   default=DEFAULT_INDEX,   help="NPZ file with reference index")
    ap.add_argument("--out",     default=DEFAULT_OUT,     help="Output folder")
    ap.add_argument("--topk",    default=DEFAULT_TOPK, type=int, help="How many matches to keep")
    args = ap.parse_args()

    queries_dir = Path(args.queries)
    index_file  = Path(args.index)
    out_dir     = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Queries: {queries_dir.resolve()}")
    print(f"Index  : {index_file.resolve()}")
    print(f"Output : {out_dir.resolve()}")
    print(f"Top-K  : {args.topk}")

    if not index_file.exists():
        raise FileNotFoundError(f"Index file not found: {index_file}")

    # --- Load index robustly
    feats, names, meta = load_index(index_file)
    if feats.ndim != 2:
        raise ValueError(f"Index features must be 2D, got shape {feats.shape}")
    N, D = feats.shape
    print(f"Loaded index: {N} references with dim={D}")

    # Normalize reference features for cosine similarity
    ref_feats = normalize_rows(feats)

    # Collect queries
    q_paths = list_masks(queries_dir)
    if not q_paths:
        raise RuntimeError(f"No query masks found in {queries_dir}")

    rows = []
    for i, qp in enumerate(tqdm(q_paths, desc="Retrieving", unit="mask")):
        # feature for this query
        q_mask = load_mask_binary(qp)
        q_feat = coarse_mask_feature(q_mask)        # shape [Dq]
        # If dims mismatch (different ds/rad_bins), pad/trim to match D
        if q_feat.shape[0] != D:
            if q_feat.shape[0] < D:
                q_feat = np.pad(q_feat, (0, D - q_feat.shape[0]), constant_values=0)
            else:
                q_feat = q_feat[:D]
        q_feat = q_feat.astype(np.float32)[None, :]  # [1, D]
        q_feat = normalize_rows(q_feat)

        # cosine similarity = dot since both are normalized
        scores = (ref_feats @ q_feat.T).ravel()     # [N]
        top = np.argsort(-scores)[:args.topk]

        for rank, idx in enumerate(top, 1):
            rows.append([str(qp), rank, float(scores[idx]), str(names[idx])])

    # write CSV
    csv_path = out_dir / "retrieval_results.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["query", "rank", "score", "ref_mask"])
        w.writerows(rows)

    print(f"\n✅ Saved Top-{args.topk} results to: {csv_path.resolve()}")

if __name__ == "__main__":
    main()

