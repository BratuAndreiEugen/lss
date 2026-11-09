import os
import sys
import json
import argparse
import numpy as np


def find_result_pairs(folder):
    """Finds all (*_summary.json, *.tsv) pairs in the given folder or its subdirectories."""
    pairs = []
    for root, _, files in os.walk(folder):
        for f in sorted(files):
            if f.endswith("_summary.json"):
                json_path = os.path.join(root, f)
                tsv_name = f.replace("_summary.json", ".tsv")
                tsv_path = os.path.join(root, tsv_name)
                if os.path.isfile(tsv_path):
                    pairs.append((json_path, tsv_path))
    return pairs


def load_tsv_scores(tsv_path):
    pos_scores = []
    neg_scores = []
    with open(tsv_path, "r", encoding="utf-8") as f:
        header = f.readline()
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) < 3:
                continue
            label, idx, score = parts[0], int(parts[1]), float(parts[2])
            if label == "pos":
                pos_scores.append((idx, score))
            elif label == "neg":
                neg_scores.append((idx, score))
    return pos_scores, neg_scores


def print_report(json_path, tsv_path, custom_threshold=None):
    with open(json_path, "r", encoding="utf-8") as f:
        summary = json.load(f)

    pos_items, neg_items = load_tsv_scores(tsv_path)
    pos_scores = np.array([s for _, s in pos_items], dtype=float)
    neg_scores = np.array([s for _, s in neg_items], dtype=float)

    scenario = summary.get("scenario", os.path.basename(os.path.dirname(json_path)))
    eer_score = summary.get("eer_score", 0.0)
    eer_threshold = summary.get("eer_threshold", 0.0)
    threshold = custom_threshold if custom_threshold is not None else eer_threshold
    thresh_label = "Custom Threshold" if custom_threshold is not None else "EER Threshold (τ)"

    n_pos = len(pos_scores)
    n_neg = len(neg_scores)

    tp = int(np.sum(pos_scores >= threshold)) if n_pos else 0
    fn = n_pos - tp
    tpr = (tp / n_pos * 100.0) if n_pos else 0.0
    fnr = (fn / n_pos * 100.0) if n_pos else 0.0

    tn = int(np.sum(neg_scores < threshold)) if n_neg else 0
    fp = n_neg - tn
    tnr = (tn / n_neg * 100.0) if n_neg else 0.0
    fpr = (fp / n_neg * 100.0) if n_neg else 0.0

    total = n_pos + n_neg
    acc = ((tp + tn) / total * 100.0) if total else 0.0

    print("=" * 72)
    print(f" DETECTION REPORT: {scenario}")
    print("=" * 72)
    print(f" Directory         : {os.path.dirname(json_path)}")
    print(f" Summary JSON      : {os.path.basename(json_path)}")
    print(f" Scores TSV        : {os.path.basename(tsv_path)}")
    print("-" * 72)
    print(f" Equal Error Rate  : {eer_score * 100:.2f}% (at EER τ = {eer_threshold:.6f})")
    print(f" Active Threshold  : {threshold:.6f} ({thresh_label})")
    print("-" * 72)

    if n_pos > 0:
        pos_gt_zero = int(np.sum(pos_scores > 0))
        print(f" [TARGET DATASET IN '-i' ({n_pos} files)]")
        print(f"   Classified WATERMARKED  : {tp:>4d} / {n_pos} ({tpr:.2f}%)")
        print(f"   Classified CLEAN        : {fn:>4d} / {n_pos} ({fnr:.2f}%)")
        print(f"   Scores > 0              : {pos_gt_zero:>4d} / {n_pos} ({pos_gt_zero / n_pos * 100:.2f}%)")
        print(f"   Score Stats             : mean={np.mean(pos_scores):.6f}, std={np.std(pos_scores):.6f}, "
              f"median={np.median(pos_scores):.6f}, min={np.min(pos_scores):.6f}, max={np.max(pos_scores):.6f}")
        missed_indices = [idx for idx, s in pos_items if s < threshold]
        if missed_indices:
            preview = missed_indices[:15]
            more = f" ... (+{len(missed_indices) - 15} more)" if len(missed_indices) > 15 else ""
            print(f"   CLEAN File Indices      : {preview}{more}")
        print("-" * 72)

    if n_neg > 0:
        neg_gt_zero = int(np.sum(neg_scores > 0))
        print(f" [CLEAN REFERENCE BASELINE IN '-c' ({n_neg} files)]")
        print(f"   Classified CLEAN        : {tn:>4d} / {n_neg} ({tnr:.2f}%)")
        print(f"   Classified WATERMARKED  : {fp:>4d} / {n_neg} ({fpr:.2f}%)")
        print(f"   Scores > 0              : {neg_gt_zero:>4d} / {n_neg} ({neg_gt_zero / n_neg * 100:.2f}%)")
        print(f"   Score Stats             : mean={np.mean(neg_scores):.6f}, std={np.std(neg_scores):.6f}, "
              f"median={np.median(neg_scores):.6f}, min={np.min(neg_scores):.6f}, max={np.max(neg_scores):.6f}")
    print("=" * 72 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Generate a watermark detection report from a results folder.")
    parser.add_argument("folder", help="Path to the detection results folder (containing TSV and JSON files)")
    parser.add_argument(
        "-t", "--threshold",
        type=float,
        default=None,
        help="Optional decision threshold override (defaults to eer_threshold from the summary JSON)"
    )
    args = parser.parse_args()

    if not os.path.isdir(args.folder):
        print(f"[ERROR] Folder not found: {args.folder}")
        sys.exit(1)

    pairs = find_result_pairs(args.folder)
    if not pairs:
        print(f"[ERROR] No (*_summary.json, *.tsv) pairs found inside '{args.folder}'.")
        sys.exit(1)

    for json_path, tsv_path in pairs:
        print_report(json_path, tsv_path, custom_threshold=args.threshold)


if __name__ == "__main__":
    main()

