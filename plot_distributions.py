import os
import sys
import json
import argparse
import numpy as np
from scipy.stats import gaussian_kde
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm

from export_distribution_stats import (
    load_theta_jsons_from_config,
    load_files_by_id,
    export_stats_csv_from_config,
)


def plot_four_distributions(ax, series_list, bins=30, show_hist=True):
    """
    Plots multiple score distributions (KDE curve + optional light histogram + mean line) on a single Axes.
    series_list: list of dicts with keys: 'scores', 'label', 'color', 'linestyle', 'fill_alpha'
    """
    valid_series = [s for s in series_list if s["scores"] is not None and len(s["scores"]) > 1]
    if not valid_series:
        return

    all_vals = np.concatenate([s["scores"] for s in valid_series])
    x_min, x_max = np.percentile(all_vals, 0.5), np.percentile(all_vals, 99.5)
    margin = 0.15 * (x_max - x_min + 1e-6)
    x_grid = np.linspace(x_min - margin, x_max + margin, 500)

    for s in valid_series:
        scores = s["scores"]
        color = s["color"]
        ls = s["linestyle"]
        mean_val = float(np.mean(scores))
        std_val = float(np.std(scores))
        full_label = f"{s['label']} (μ={mean_val:.4f}, σ={std_val:.4f})"

        if show_hist:
            ax.hist(
                scores, bins=bins, range=(x_min - margin, x_max + margin),
                density=True, color=color, alpha=0.10, histtype="stepfilled"
            )

        try:
            kde = gaussian_kde(scores)
            y_kde = kde(x_grid)
            ax.plot(x_grid, y_kde, color=color, linestyle=ls, linewidth=2.2, label=full_label)
            ax.fill_between(x_grid, 0, y_kde, color=color, alpha=s.get("fill_alpha", 0.12))
        except Exception:
            ax.hist(
                scores, bins=bins, density=True, color=color, linestyle=ls,
                histtype="step", linewidth=2.0, label=full_label
            )

        ax.axvline(mean_val, color=color, linestyle=ls, linewidth=1.2, alpha=0.8)

    ax.axvline(0.0, color="black", linestyle=":", linewidth=1.0, alpha=0.6, label="Zero Score (0.0)")
    ax.set_xlim(x_min - margin, x_max + margin)


def main():
    parser = argparse.ArgumentParser(
        description="Plot 4 subchunk score distributions (theta=0.18 & 0.36, tampered vs. not-tampered) using YAML config."
    )
    parser.add_argument(
        "--config", "-c",
        default="config/config_LSS.yaml",
        help="Path to YAML configuration file (default: config/config_LSS.yaml)"
    )
    parser.add_argument(
        "-o", "--output_dir",
        default=None,
        help="Directory to save output distribution plots and stats CSV"
    )
    parser.add_argument(
        "--bins",
        type=int,
        default=30,
        help="Number of histogram bins (default: 30)"
    )
    parser.add_argument(
        "--no_hist",
        action="store_true",
        help="Plot smooth KDE distribution curves only without background histogram bars"
    )

    args = parser.parse_args()

    if not os.path.isfile(args.config):
        print(f"[ERROR] Configuration file not found: {args.config}")
        sys.exit(1)

    theta_jsons, config = load_theta_jsons_from_config(args.config)
    print(f"[INFO] Loaded {len(theta_jsons)} detection paths from '{args.config}':")

    datasets = {}
    all_file_ids = []
    for theta_label, jpath in theta_jsons:
        print(f"  - θ = {theta_label}: {jpath}")
        datasets[theta_label] = load_files_by_id(jpath)
        for fid in datasets[theta_label]:
            if fid not in all_file_ids:
                all_file_ids.append(fid)

    if args.output_dir:
        out_dir = args.output_dir
    else:
        first_json_dir = os.path.dirname(theta_jsons[0][1])
        parent_dir = os.path.dirname(first_json_dir) or "."
        out_dir = os.path.join(parent_dir, "distribution_plots")

    os.makedirs(out_dir, exist_ok=True)

    palette = {
        ("0.18", "not_tampered"): {"color": "#1f77b4", "linestyle": "-",  "fill_alpha": 0.14, "label": "θ=0.18 | Watermarked (Not Tampered)"},
        ("0.18", "tampered"):     {"color": "#ff7f0e", "linestyle": "--", "fill_alpha": 0.14, "label": "θ=0.18 | Watermarked + Tampered (Spliced)"},
        ("0.36", "not_tampered"): {"color": "#2ca02c", "linestyle": "-",  "fill_alpha": 0.14, "label": "θ=0.36 | Watermarked (Not Tampered)"},
        ("0.36", "tampered"):     {"color": "#d62728", "linestyle": "--", "fill_alpha": 0.14, "label": "θ=0.36 | Watermarked + Tampered (Spliced)"},
    }
    fallback_colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]

    for fid in tqdm(all_file_ids, desc="Plotting Distributions"):
        stem = os.path.splitext(fid)[0]
        fig, ax = plt.subplots(figsize=(11, 6))

        series_list = []
        c_idx = 0
        for theta_label, _ in theta_jsons:
            file_data = datasets[theta_label].get(fid)
            if not file_data:
                continue

            for variant_key, default_desc in [
                ("not_tampered", f"θ={theta_label} | Watermarked (Not Tampered)"),
                ("tampered",     f"θ={theta_label} | Watermarked + Tampered (Spliced)")
            ]:
                style = palette.get((theta_label, variant_key), {
                    "color": fallback_colors[c_idx % len(fallback_colors)],
                    "linestyle": "-" if variant_key == "not_tampered" else "--",
                    "fill_alpha": 0.12,
                    "label": default_desc
                })
                series_list.append({
                    "scores": file_data[variant_key],
                    **style
                })
                c_idx += 1

        plot_four_distributions(ax, series_list, bins=args.bins, show_hist=not args.no_hist)

        ax.set_title(f"Subchunk Watermark Score Distributions — {fid}", fontsize=11, fontweight="bold")
        ax.set_xlabel("Subchunk Watermark Score", fontsize=10)
        ax.set_ylabel("Probability Density", fontsize=10)
        ax.grid(True, linestyle="--", alpha=0.35)
        ax.legend(loc="upper right", fontsize=9, framealpha=0.9)

        fig.tight_layout()
        out_path = os.path.join(out_dir, f"dist_{stem}.png")
        fig.savefig(out_path, dpi=150)
        plt.close(fig)

    print(f"[DONE] Saved {len(all_file_ids)} distribution plots to '{out_dir}'.")

    csv_path = os.path.join(out_dir, "distribution_stats.csv")
    export_stats_csv_from_config(args.config, csv_path)
    print(f"[DONE] Saved summary statistics CSV to '{csv_path}'.")


if __name__ == "__main__":
    main()
