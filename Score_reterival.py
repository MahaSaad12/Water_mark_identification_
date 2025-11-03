# score_retrieval.py
from pathlib import Path
import csv, re

RESULTS_CSV = Path("runs_match/retrieval_results.csv")
# Optional: mapping CSV with columns: query_stem,briquet_id
# Leave as None if you don't have it.
MAPPING_CSV = Path("aclass_to_briquet.csv")  # or None

TOPKS = (1, 5, 10)

# -------- helpers --------
def load_mapping(mapping_csv: Path | None):
    m = {}
    if mapping_csv and mapping_csv.exists():
        with open(mapping_csv, newline="", encoding="utf-8") as f:
            r = csv.DictReader(f)
            for row in r:
                qstem = row["query_stem"].strip()
                bid = str(row["briquet_id"]).strip()
                if qstem and bid:
                    m[qstem] = bid
    return m

RE_BRIQ = re.compile(r"briquet[_-]?(\d+)", re.IGNORECASE)

def parse_briquet_id_from_path(p: str) -> str | None:
    """Try to extract a briquet numeric id from a path/stem."""
    # Look for the *last* briquet_#### in the string
    found = None
    for m in RE_BRIQ.finditer(p):
        found = m.group(1)
    return found

# -------- main --------
def main():
    # load mapping
    mapping = load_mapping(MAPPING_CSV)

    # group rows per query
    per_q = {}
    with open(RESULTS_CSV, newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            q = row["query"]
            per_q.setdefault(q, []).append(row)

    # sort by rank
    for q in per_q:
        per_q[q] = sorted(per_q[q], key=lambda x: int(x["rank"]))

    total, skipped = 0, 0
    hits = {k: 0 for k in TOPKS}

    for q, rows in per_q.items():
        qstem = Path(q).stem

        # 1) mapping overrides all
        gt = mapping.get(qstem)
        # 2) else, try to parse GT directly from query path (for synthetic/cross-domain that carry briquet id)
        if gt is None:
            gt = parse_briquet_id_from_path(q)

        if gt is None:
            skipped += 1
            continue

        # predicted order → extract briquet ids from ref names
        pred_ids = []
        for row in rows:
            rid = parse_briquet_id_from_path(row["ref_mask"])
            pred_ids.append(rid)

        total += 1
        for K in TOPKS:
            topk = pred_ids[:K]
            if gt in topk:
                hits[K] += 1

    denom = max(1, total)
    for K in TOPKS:
        acc = hits[K] / denom
        print(f"Top-{K} accuracy: {hits[K]}/{total} = {acc:.3f}")
    print(f"(Skipped {skipped} queries without a GT briquet id.)")

if __name__ == "__main__":
    main()
