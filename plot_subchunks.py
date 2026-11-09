import os
import sys
import json
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm


def load_splice_metadata(metadata_path):
    """Loads splicer metadata.json and maps both tampered_file_id and original_file to tampered_region."""
    if not metadata_path or not os.path.isfile(metadata_path):
        return {}
    with open(metadata_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    region_map = {}
    for rec in data.get("tampered_recordings", []):
        region = rec.get("tampered_region")
        if region:
            if "tampered_file_id" in rec:
                region_map[rec["tampered_file_id"]] = region
            if "original_file" in rec:
                region_map[rec["original_file"]] = region
    return region_map


def moving_average(values, window):
    if window <= 1 or len(values) < window:
        return values
    kernel = np.ones(window) / window
    return np.convolve(values, kernel, mode="same")


def main():
    parser = argparse.ArgumentParser(
        description="Plot subchunk watermark scores across time for input (pos) and reference (neg) files."
    )
    parser.add_argument(
        "json_input",
        help="Path to subchunk_scores.json (or the directory containing subchunk_scores.json)"
    )
    parser.add_argument(
        "-o", "--output_dir",
        default=None,
        help="Directory to save output PNG plots (defaults to '<json_dir>/plots')"
    )
    parser.add_argument(
        "-m", "--metadata",
        default=None,
        help="Optional path to splicer metadata.json to highlight the 1s spliced region on each plot"
    )
    parser.add_argument(
        "-w", "--smooth_window",
        type=int,
        default=8,
        help="Moving average window in subchunks to overlay smoothed curve (8 subchunks = 1 chunk ≈ 0.43s; 1 to disable)"
    )
    parser.add_argument(
        "-t", "--threshold",
        type=float,
        default=None,
        help="Optional threshold override (defaults to threshold saved in subchunk_scores.json)"
    )

    args = parser.parse_args()

    json_path = args.json_input
    if os.path.isdir(json_path):
        json_path = os.path.join(json_path, "subchunk_scores.json")

    if not os.path.isfile(json_path):
        print(f"[ERROR] JSON file not found: {json_path}")
        sys.exit(1)

    out_dir = args.output_dir or os.path.join(os.path.dirname(json_path), "plots")
    os.makedirs(out_dir, exist_ok=True)

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    threshold = args.threshold if args.threshold is not None else data.get("threshold", 0.0)
    files = data.get("files", [])
    splice_regions = load_splice_metadata(args.metadata)

    print(f"[INFO] Loaded {len(files)} files from '{json_path}'.")
    print(f"[INFO] Saving plots to '{out_dir}'...")

    for entry in tqdm(files, desc="Generating Plots"):
        file_id = entry["file_id"]
        stem = os.path.splitext(file_id)[0]
        clean_stem = stem.removeprefix("tampered_")

        sub_pos = entry.get("subchunks_pos") or []
        sub_neg = entry.get("subchunks_neg") or []

        if not sub_pos:
            continue

        # Midpoint time (seconds) for each subchunk
        t_pos = np.array([(sc["start_sec"] + sc["end_sec"]) / 2.0 for sc in sub_pos])
        y_pos = np.array([sc["score"] for sc in sub_pos])

        fig, ax = plt.subplots(figsize=(12, 5))

        # Plot raw subchunk scores (pos)
        if args.smooth_window > 1:
            ax.plot(t_pos, y_pos, color="#1f77b4", alpha=0.30, linewidth=1.0, label="Input Subchunks (53ms raw)")
            y_pos_smooth = moving_average(y_pos, args.smooth_window)
            ax.plot(
                t_pos, y_pos_smooth, color="#1f77b4", linewidth=2.2,
                label=f"Input Smoothed ({args.smooth_window}-subchunk avg, global={entry.get('pos_global_score', 0):.4f})"
            )
        else:
            ax.plot(
                t_pos, y_pos, color="#1f77b4", linewidth=1.5,
                label=f"Input Score (pos, global={entry.get('pos_global_score', 0):.4f})"
            )

        # Plot reference subchunk scores (neg) if available
        if sub_neg:
            t_neg = np.array([(sc["start_sec"] + sc["end_sec"]) / 2.0 for sc in sub_neg])
            y_neg = np.array([sc["score"] for sc in sub_neg])
            if args.smooth_window > 1:
                ax.plot(t_neg, y_neg, color="#7f7f7f", alpha=0.25, linewidth=1.0, label="Reference Subchunks (53ms raw)")
                y_neg_smooth = moving_average(y_neg, args.smooth_window)
                ax.plot(
                    t_neg, y_neg_smooth, color="#555555", linewidth=1.8, linestyle="-.",
                    label=f"Reference Smoothed (neg, global={entry.get('neg_global_score', 0):.4f})"
                )
            else:
                ax.plot(
                    t_neg, y_neg, color="#555555", linewidth=1.3, alpha=0.8,
                    label=f"Reference Score (neg, global={entry.get('neg_global_score', 0):.4f})"
                )

        # Plot threshold line and zero line
        ax.axhline(threshold, color="#d62728", linestyle="--", linewidth=1.5, label=f"Threshold τ = {threshold:.4f}")
        ax.axhline(0.0, color="black", linestyle=":", linewidth=0.8, alpha=0.5)

        # Highlight spliced/tampered region if metadata.json was provided
        region = splice_regions.get(file_id) or splice_regions.get(f"{clean_stem}.wav")
        if region:
            s_sec, e_sec = region["start_sec"], region["end_sec"]
            ax.axvspan(s_sec, e_sec, color="#ff9896", alpha=0.35, label=f"Spliced Region ({s_sec:.2f}s - {e_sec:.2f}s)")

        ax.set_title(f"Watermark Score Across Time — {file_id}", fontsize=11, fontweight="bold")
        ax.set_xlabel("Time (seconds)", fontsize=10)
        ax.set_ylabel("Subchunk Watermark Score", fontsize=10)
        ax.set_xlim(0, max(t_pos[-1] + 0.1, 10.0))
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.legend(loc="upper right", fontsize=8.5)

        fig.tight_layout()
        out_file = os.path.join(out_dir, f"{stem}.png")
        fig.savefig(out_file, dpi=150)
        plt.close(fig)

    print(f"[DONE] Saved {len(files)} plots to '{out_dir}'.")


if __name__ == "__main__":
    main()
