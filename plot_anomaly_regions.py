import os
import sys
import json
import yaml
import argparse
from collections import defaultdict
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm


CONFIG_NAMES = [
    "theta_0_18_not_tampered",
    "theta_0_18_tampered",
    "theta_0_36_not_tampered",
    "theta_0_36_tampered",
]


def moving_average(values, window):
    if window <= 1 or len(values) < window:
        return values
    kernel = np.ones(window) / window
    return np.convolve(values, kernel, mode="same")


def region_duration_ms(region):
    if "start_ms" in region and "end_ms" in region:
        return round(float(region["end_ms"]) - float(region["start_ms"]), 3)
    if "start_sec" in region and "end_sec" in region:
        return round((float(region["end_sec"]) - float(region["start_sec"])) * 1000.0, 3)
    return 0.0


def plot_configuration_files(anomaly_json_path, out_base_dir, smooth_window=8, min_duration_ms=0.0):
    with open(anomaly_json_path, "r", encoding="utf-8") as f:
        anom_data = json.load(f)

    config_name = anom_data.get("configuration", os.path.splitext(os.path.basename(anomaly_json_path))[0])
    variant = anom_data.get("variant", "not_tampered" if "not_tampered" in config_name else "tampered")
    subchunk_scores_path = anom_data.get("subchunk_scores_path")
    lower_limit = anom_data.get("lower_limit_percentile", 5)
    upper_limit = anom_data.get("upper_limit_percentile", 95)
    all_regions = anom_data.get("regions", [])

    if not subchunk_scores_path or not os.path.isfile(subchunk_scores_path):
        print(f"[ERROR] subchunk_scores.json not found for '{config_name}': {subchunk_scores_path}")
        return

    with open(subchunk_scores_path, "r", encoding="utf-8") as f:
        det_data = json.load(f)

    # Group detected anomaly regions by original_file
    regions_by_file = defaultdict(list)
    for reg in all_regions:
        if min_duration_ms > 0.0 and region_duration_ms(reg) < min_duration_ms:
            continue
        regions_by_file[reg["original_file"]].append(reg)

    subchunk_field = "subchunks_neg" if variant == "not_tampered" else "subchunks_pos"
    global_score_field = "neg_global_score" if variant == "not_tampered" else "pos_global_score"

    config_out_dir = os.path.join(out_base_dir, config_name)
    os.makedirs(config_out_dir, exist_ok=True)

    lower_type = f"<p{lower_limit}"
    upper_type = f">p{upper_limit}"

    files = det_data.get("files", [])
    plotted_count = 0
    for entry in tqdm(files, desc=f"Plotting {config_name}"):
        file_id = entry["file_id"]
        file_regions = regions_by_file.get(file_id, [])
        if not file_regions:
            continue

        stem = os.path.splitext(file_id)[0]
        subchunks = entry.get(subchunk_field) or []
        if not subchunks:
            continue

        t_sec = np.array([(sc["start_sec"] + sc["end_sec"]) / 2.0 for sc in subchunks], dtype=float)
        y_scores = np.array([sc["score"] for sc in subchunks], dtype=float)
        global_score = entry.get(global_score_field)
        global_str = f", global={global_score:.4f}" if global_score is not None else ""

        fig, ax = plt.subplots(figsize=(12, 5))

        # 1. Plot raw scores and smoothed scores (just like plot_subchunks.py)
        if smooth_window > 1 and len(y_scores) >= smooth_window:
            ax.plot(t_sec, y_scores, color="#1f77b4", alpha=0.30, linewidth=1.0, label="Subchunk Scores (53ms raw)")
            y_smooth = moving_average(y_scores, smooth_window)
            ax.plot(
                t_sec, y_smooth, color="#1f77b4", linewidth=2.2,
                label=f"Smoothed ({smooth_window}-subchunk avg{global_str})"
            )
        else:
            ax.plot(
                t_sec, y_scores, color="#1f77b4", linewidth=1.5,
                label=f"Subchunk Scores ({config_name}{global_str})"
            )

        ax.axhline(0.0, color="black", linestyle=":", linewidth=0.8, alpha=0.5)

        # 2. Highlight anomaly regions detected by parse_anomaly_regions.py
        low_regions = [r for r in file_regions if r.get("type") == lower_type]
        high_regions = [r for r in file_regions if r.get("type") == upper_type]

        for idx, reg in enumerate(low_regions):
            label = f"Anomaly {lower_type} ({len(low_regions)} regions)" if idx == 0 else None
            ax.axvspan(
                reg["start_sec"], reg["end_sec"],
                color="#d62728", alpha=0.28, label=label
            )

        for idx, reg in enumerate(high_regions):
            label = f"Anomaly {upper_type} ({len(high_regions)} regions)" if idx == 0 else None
            ax.axvspan(
                reg["start_sec"], reg["end_sec"],
                color="#2ca02c", alpha=0.25, label=label
            )

        ax.set_title(f"{file_id} — [{config_name}]", fontsize=11, fontweight="bold")
        ax.set_xlabel("Time (seconds)", fontsize=10)
        ax.set_ylabel("Subchunk Watermark Score", fontsize=10)
        ax.set_xlim(0, max(float(t_sec[-1]) + 0.1, 10.0))
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.legend(loc="upper right", fontsize=8.5)

        fig.tight_layout()
        out_png = os.path.join(config_out_dir, f"{stem}.png")
        fig.savefig(out_png, dpi=150)
        plt.close(fig)
        plotted_count += 1

    print(f"[DONE] Saved {plotted_count} / {len(files)} plots (files with anomalies) for '{config_name}' -> {config_out_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Plot subchunk scores + smoothed scores + highlighted anomaly regions for every file into 4 configuration folders."
    )
    parser.add_argument(
        "--config", "-c",
        default="config/anomaly_pipeline_config.yaml",
        help="Path to YAML configuration file (default: config/anomaly_pipeline_config.yaml)"
    )
    parser.add_argument(
        "--json_dir", "-j",
        default=None,
        help="Directory containing the 4 'anomalies_<config>.json' files (defaults to directory of parser.csv_path)"
    )
    parser.add_argument(
        "--output_dir", "-o",
        default=None,
        help="Base output directory to create the 4 plot folders in (defaults to '<json_dir>/anomaly_region_plots')"
    )
    parser.add_argument(
        "-w", "--smooth_window",
        type=int,
        default=8,
        help="Moving average window in subchunks to overlay smoothed curve (8 subchunks ≈ 0.43s; 1 to disable)"
    )
    parser.add_argument(
        "--min_duration_ms",
        type=float,
        default=0.0,
        help="Optional minimum anomaly region duration in ms to highlight on the plots (default: 0.0 = highlight all)"
    )

    args = parser.parse_args()

    json_dir = args.json_dir
    if json_dir is None:
        if not os.path.isfile(args.config):
            print(f"[ERROR] Config file not found: {args.config}")
            sys.exit(1)
        with open(args.config, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        csv_path = cfg.get("parser", {}).get("csv_path")
        if not csv_path:
            print("[ERROR] Could not determine json_dir from config. Pass --json_dir explicitly.")
            sys.exit(1)
        json_dir = os.path.dirname(os.path.abspath(csv_path))

    out_base_dir = args.output_dir or os.path.join(json_dir, "anomaly_region_plots")
    os.makedirs(out_base_dir, exist_ok=True)

    found_any = False
    for cfg_name in CONFIG_NAMES:
        json_path = os.path.join(json_dir, f"anomalies_{cfg_name}.json")
        if os.path.isfile(json_path):
            found_any = True
            plot_configuration_files(
                anomaly_json_path=json_path,
                out_base_dir=out_base_dir,
                smooth_window=args.smooth_window,
                min_duration_ms=args.min_duration_ms,
            )
        else:
            print(f"[WARN] Anomaly JSON not found: {json_path}")

    if not found_any:
        print(f"[ERROR] No 'anomalies_*.json' files found in '{json_dir}'. Run parse_anomaly_regions.py first.")
        sys.exit(1)


if __name__ == "__main__":
    main()
