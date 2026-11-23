import os
import sys
import csv
import json
import yaml
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm


DEFAULT_MULTIPLIERS = [0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2]
DEFAULT_THETAS = [0.18, 0.36]
PERCENTILES = [5, 10, 15, 20, 25, 50, 75, 80, 85, 90, 95]


def moving_average(values, window):
    if window <= 1 or len(values) < window:
        return values
    kernel = np.ones(window) / window
    return np.convolve(values, kernel, mode="same")


def filter_subchunks(subchunks, ignore_last_chunk=False):
    if not subchunks:
        return []
    if ignore_last_chunk:
        last_chunk = max(sc["chunk_idx"] for sc in subchunks)
        return [sc for sc in subchunks if sc["chunk_idx"] < last_chunk]
    return subchunks


def compute_stats_dict(scores):
    if scores is None or len(scores) == 0:
        res = {f"p{p}": None for p in PERCENTILES}
        res["mean"] = None
        res["std"] = None
        return res
    pct_vals = np.percentile(scores, PERCENTILES)
    res = {f"p{p}": round(float(val), 6) for p, val in zip(PERCENTILES, pct_vals)}
    res["mean"] = round(float(np.mean(scores)), 6)
    res["std"] = round(float(np.std(scores)), 6)
    return res


def discover_detection_grid(detection_root, thetas=None, multipliers=None):
    """
    Discovers all available (theta, multiplier) subchunk_scores.json files under:
      <detection_root>/<theta>/<multiplier>/subchunk_scores.json
    """
    if thetas is None:
        thetas = DEFAULT_THETAS
    if multipliers is None:
        multipliers = DEFAULT_MULTIPLIERS

    grid = {}  # (theta_str, m_str) -> { "theta": float, "m": float, "json_path": str, "threshold": float, "files": dict }
    found_thetas = []
    found_multipliers = []

    for t in thetas:
        t_str = str(t)
        t_dir = os.path.join(detection_root, t_str)
        if not os.path.isdir(t_dir):
            continue
        if t_str not in found_thetas:
            found_thetas.append(t_str)

        for m in multipliers:
            # Check both exact representation (e.g. '1' or '1.0')
            candidates = [str(m)]
            if isinstance(m, float) and m.is_integer():
                candidates.append(str(int(m)))
            elif isinstance(m, int):
                candidates.append(str(float(m)))

            json_path = None
            m_label = str(m)
            for cand in candidates:
                p = os.path.join(t_dir, cand, "subchunk_scores.json")
                if os.path.isfile(p):
                    json_path = p
                    m_label = cand
                    break

            if json_path is None:
                continue

            if m_label not in found_multipliers:
                found_multipliers.append(m_label)

            grid[(t_str, m_label)] = {
                "theta": float(t),
                "theta_str": t_str,
                "m": float(m),
                "m_str": m_label,
                "json_path": json_path,
            }

    found_multipliers.sort(key=lambda x: float(x))
    found_thetas.sort(key=lambda x: float(x))
    return grid, found_thetas, found_multipliers


def load_all_data(grid, ignore_last_chunk=False):
    """
    Loads all subchunk_scores.json files in the grid and indexes them by file_id.
    """
    all_file_ids = []
    for key, info in grid.items():
        with open(info["json_path"], "r", encoding="utf-8") as f:
            raw = json.load(f)
        info["threshold"] = float(raw.get("threshold", 0.0))
        files_map = {}
        for entry in raw.get("files", []):
            fid = entry["file_id"]
            if fid not in all_file_ids:
                all_file_ids.append(fid)

            sub_pos = filter_subchunks(entry.get("subchunks_pos") or [], ignore_last_chunk=ignore_last_chunk)
            sub_neg = filter_subchunks(entry.get("subchunks_neg") or [], ignore_last_chunk=ignore_last_chunk)

            t_pos = np.array([(sc["start_sec"] + sc["end_sec"]) / 2.0 for sc in sub_pos], dtype=float)
            y_pos = np.array([sc["score"] for sc in sub_pos], dtype=float)
            t_neg = np.array([(sc["start_sec"] + sc["end_sec"]) / 2.0 for sc in sub_neg], dtype=float)
            y_neg = np.array([sc["score"] for sc in sub_neg], dtype=float)

            files_map[fid] = {
                "t_pos": t_pos,
                "y_pos": y_pos,
                "t_neg": t_neg,
                "y_neg": y_neg,
                "pos_mean": float(np.mean(y_pos)) if len(y_pos) > 0 else 0.0,
                "neg_mean": float(np.mean(y_neg)) if len(y_neg) > 0 else 0.0,
                "pos_stats": compute_stats_dict(y_pos),
                "neg_stats": compute_stats_dict(y_neg),
            }
        info["files"] = files_map

    return all_file_ids


def export_statistics_csvs(grid, thetas, multipliers, all_file_ids, out_dir):
    """
    Writes:
      1. A master CSV with all (theta, m) statistics for every file.
      2. A separate CSV per theta for convenient inspection.
    """
    stat_keys = [f"p{p}" for p in PERCENTILES] + ["mean", "std"]

    # 1. Per-theta CSVs + Master CSV
    master_header = ["file_id"]
    master_cases = []

    for t_str in thetas:
        t_clean = t_str.replace(".", "_")
        theta_header = ["file_id"]
        theta_cases = []

        for m_str in multipliers:
            if (t_str, m_str) not in grid:
                continue
            m_clean = m_str.replace(".", "_")
            for variant_key, variant_label in [("pos_stats", "watermarked"), ("neg_stats", "clean_ref")]:
                prefix = f"theta_{t_clean}_m_{m_clean}_{variant_label}"
                master_cases.append((t_str, m_str, variant_key, prefix))
                theta_cases.append((t_str, m_str, variant_key, prefix))
                for sk in stat_keys:
                    master_header.append(f"{prefix}_{sk}")
                    theta_header.append(f"{prefix}_{sk}")

        # Save per-theta CSV
        theta_csv_path = os.path.join(out_dir, f"distribution_stats_theta_{t_clean}.csv")
        with open(theta_csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(theta_header)
            for fid in all_file_ids:
                row = [fid]
                for (ts, ms, v_key, _) in theta_cases:
                    f_data = grid[(ts, ms)]["files"].get(fid)
                    stats = f_data[v_key] if f_data else {sk: None for sk in stat_keys}
                    row.extend([stats[sk] for sk in stat_keys])
                writer.writerow(row)
        print(f"[CSV] Saved θ={t_str} statistics CSV -> {theta_csv_path}")

    # Save master CSV
    master_csv_path = os.path.join(out_dir, "distribution_stats_all.csv")
    with open(master_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(master_header)
        for fid in all_file_ids:
            row = [fid]
            for (ts, ms, v_key, _) in master_cases:
                f_data = grid[(ts, ms)]["files"].get(fid)
                stats = f_data[v_key] if f_data else {sk: None for sk in stat_keys}
                row.extend([stats[sk] for sk in stat_keys])
            writer.writerow(row)
    print(f"[CSV] Saved master statistics CSV -> {master_csv_path}")


def plot_dataset_average_vs_multiplier(grid, thetas, multipliers, all_file_ids, out_dir):
    """
    Creates:
      - One separate plot per theta: Multiplier (X) vs. Dataset Average Score (Y)
      - One combined plot comparing all thetas together
    """
    theta_colors = {"0.18": "#1f77b4", "0.36": "#2ca02c"}

    combined_fig, combined_ax = plt.subplots(figsize=(9, 5.5))

    for idx_t, t_str in enumerate(thetas):
        m_vals = []
        pos_means = []
        pos_stds = []
        neg_means = []
        neg_stds = []

        for m_str in multipliers:
            if (t_str, m_str) not in grid:
                continue
            info = grid[(t_str, m_str)]
            file_pos_means = [info["files"][fid]["pos_mean"] for fid in all_file_ids if fid in info["files"]]
            file_neg_means = [info["files"][fid]["neg_mean"] for fid in all_file_ids if fid in info["files"]]

            if not file_pos_means:
                continue

            m_vals.append(float(m_str))
            pos_means.append(float(np.mean(file_pos_means)))
            pos_stds.append(float(np.std(file_pos_means)))
            neg_means.append(float(np.mean(file_neg_means)))
            neg_stds.append(float(np.std(file_neg_means)))

        if not m_vals:
            continue

        m_arr = np.array(m_vals)
        pos_m_arr = np.array(pos_means)
        pos_s_arr = np.array(pos_stds)
        neg_m_arr = np.array(neg_means)
        neg_s_arr = np.array(neg_stds)

        color = theta_colors.get(t_str, f"C{idx_t}")

        # Individual plot for this theta
        fig, ax = plt.subplots(figsize=(9, 5.5))
        ax.plot(m_arr, pos_m_arr, marker="o", color=color, linewidth=2.4, label=f"Watermarked Mean Score (θ={t_str})")
        ax.fill_between(m_arr, pos_m_arr - pos_s_arr, pos_m_arr + pos_s_arr, color=color, alpha=0.18, label="Watermarked ±1σ across dataset")

        ax.plot(m_arr, neg_m_arr, marker="s", color="#555555", linestyle="--", linewidth=1.8, label=f"Clean Reference Mean Score (θ={t_str})")
        ax.fill_between(m_arr, neg_m_arr - neg_s_arr, neg_m_arr + neg_s_arr, color="#7f7f7f", alpha=0.15)

        ax.axhline(0.0, color="black", linestyle=":", linewidth=0.9, alpha=0.6)
        ax.set_title(f"Whole-Dataset Average Watermark Score vs. Amplitude Multiplier (θ = {t_str})", fontsize=11.5, fontweight="bold")
        ax.set_xlabel("Amplitude Multiplier (m)", fontsize=10.5)
        ax.set_ylabel("Dataset Average Score", fontsize=10.5)
        ax.set_xticks(m_arr)
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.legend(loc="best", fontsize=9.5)
        fig.tight_layout()

        t_clean = t_str.replace(".", "_")
        out_path = os.path.join(out_dir, f"dataset_avg_score_vs_multiplier_theta_{t_clean}.png")
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"[PLOT] Saved dataset average vs. multiplier plot (θ={t_str}) -> {out_path}")

        # Also add to combined plot
        combined_ax.plot(m_arr, pos_m_arr, marker="o", color=color, linewidth=2.4, label=f"Watermarked (θ={t_str})")
        combined_ax.fill_between(m_arr, pos_m_arr - pos_s_arr, pos_m_arr + pos_s_arr, color=color, alpha=0.14)
        combined_ax.plot(m_arr, neg_m_arr, marker="s", color=color, linestyle="--", linewidth=1.4, alpha=0.7, label=f"Clean Ref (θ={t_str})")

    combined_ax.axhline(0.0, color="black", linestyle=":", linewidth=0.9, alpha=0.6)
    combined_ax.set_title("Whole-Dataset Average Watermark Score vs. Amplitude Multiplier (All θ)", fontsize=11.5, fontweight="bold")
    combined_ax.set_xlabel("Amplitude Multiplier (m)", fontsize=10.5)
    combined_ax.set_ylabel("Dataset Average Score", fontsize=10.5)
    combined_ax.grid(True, linestyle="--", alpha=0.4)
    combined_ax.legend(loc="best", fontsize=9.5)
    combined_fig.tight_layout()
    combined_out = os.path.join(out_dir, "dataset_avg_score_vs_multiplier_all_thetas.png")
    combined_fig.savefig(combined_out, dpi=150)
    plt.close(combined_fig)


def plot_per_file_stacked(grid, thetas, multipliers, all_file_ids, out_dir, smooth_window=8):
    """
    For each file in the dataset, creates a single tall vertically-stacked figure containing:
      - All (theta, m) subchunk score-over-time plots stacked one under the other (16 plots for 2 thetas x 8 multipliers)
      - 1 bottom plot with Multiplier (m) on the X-axis and the file's Average Score on the Y-axis (for each theta)
    """
    per_file_dir = os.path.join(out_dir, "per_file_plots")
    os.makedirs(per_file_dir, exist_ok=True)

    # Ordered list of (theta_str, m_str) panels: group by theta (or interleave by m)
    panels = []
    for t_str in thetas:
        for m_str in multipliers:
            if (t_str, m_str) in grid:
                panels.append((t_str, m_str))

    n_time_panels = len(panels)
    if n_time_panels == 0:
        return

    total_rows = n_time_panels + 1  # 16 time-series plots + 1 modifier vs. average score plot
    theta_colors = {"0.18": "#1f77b4", "0.36": "#2ca02c"}

    print(f"[PLOT] Generating {len(all_file_ids)} per-file stacked plots ({n_time_panels} time panels + 1 summary panel each)...")

    for fid in tqdm(all_file_ids, desc="Per-File Stacked Plots"):
        stem = os.path.splitext(fid)[0]

        # Height: 2.3 inches per time panel + 3.5 inches for the bottom summary plot
        fig_height = n_time_panels * 2.3 + 3.8
        fig, axes = plt.subplots(
            nrows=total_rows,
            ncols=1,
            figsize=(13, fig_height),
            gridspec_kw={"height_ratios": [1.0] * n_time_panels + [1.45]},
        )
        if total_rows == 1:
            axes = [axes]

        # 1. Plot each (theta, m) score-over-time graph one under the other
        for idx_p, (t_str, m_str) in enumerate(panels):
            ax = axes[idx_p]
            info = grid[(t_str, m_str)]
            threshold = info["threshold"]
            f_data = info["files"].get(fid)
            main_color = theta_colors.get(t_str, "#1f77b4")

            if not f_data or len(f_data["t_pos"]) == 0:
                ax.text(0.5, 0.5, f"No data for θ={t_str}, m={m_str}", ha="center", va="center")
                continue

            t_pos, y_pos = f_data["t_pos"], f_data["y_pos"]
            t_neg, y_neg = f_data["t_neg"], f_data["y_neg"]

            if smooth_window > 1 and len(y_pos) >= smooth_window:
                ax.plot(t_pos, y_pos, color=main_color, alpha=0.28, linewidth=0.9, label="Watermarked (53ms raw)")
                y_pos_smooth = moving_average(y_pos, smooth_window)
                ax.plot(
                    t_pos, y_pos_smooth, color=main_color, linewidth=2.0,
                    label=f"Watermarked Smoothed (mean={f_data['pos_mean']:.4f})"
                )
            else:
                ax.plot(t_pos, y_pos, color=main_color, linewidth=1.4, label=f"Watermarked (mean={f_data['pos_mean']:.4f})")

            if len(t_neg) > 0:
                if smooth_window > 1 and len(y_neg) >= smooth_window:
                    ax.plot(t_neg, y_neg, color="#7f7f7f", alpha=0.22, linewidth=0.9)
                    y_neg_smooth = moving_average(y_neg, smooth_window)
                    ax.plot(
                        t_neg, y_neg_smooth, color="#555555", linestyle="-.", linewidth=1.5,
                        label=f"Clean Ref Smoothed (mean={f_data['neg_mean']:.4f})"
                    )
                else:
                    ax.plot(t_neg, y_neg, color="#555555", linestyle="-.", linewidth=1.2, label=f"Clean Ref (mean={f_data['neg_mean']:.4f})")

            ax.axhline(threshold, color="#d62728", linestyle="--", linewidth=1.2, label=f"EER τ = {threshold:.4f}")
            ax.axhline(0.0, color="black", linestyle=":", linewidth=0.8, alpha=0.5)

            ax.set_title(f"[Panel {idx_p + 1}/{n_time_panels}]  θ = {t_str}  |  Multiplier m = {m_str}  —  {fid}", fontsize=10, fontweight="bold", loc="left")
            ax.set_ylabel("Score", fontsize=9)
            ax.set_xlim(0, max(float(t_pos[-1]) + 0.1, 10.0))
            ax.grid(True, linestyle="--", alpha=0.35)
            ax.legend(loc="upper right", fontsize=8, ncol=2)

        axes[n_time_panels - 1].set_xlabel("Time (seconds)", fontsize=9.5)

        # 2. Bottom panel (Row 17): Multiplier (m) on X-axis vs. File's Average Score on Y-axis
        ax_sum = axes[-1]
        all_m_ticks = []
        for idx_t, t_str in enumerate(thetas):
            m_vals = []
            pos_avgs = []
            neg_avgs = []
            for m_str in multipliers:
                if (t_str, m_str) not in grid:
                    continue
                f_data = grid[(t_str, m_str)]["files"].get(fid)
                if not f_data:
                    continue
                m_val = float(m_str)
                m_vals.append(m_val)
                if m_val not in all_m_ticks:
                    all_m_ticks.append(m_val)
                pos_avgs.append(f_data["pos_mean"])
                neg_avgs.append(f_data["neg_mean"])

            if m_vals:
                c = theta_colors.get(t_str, f"C{idx_t}")
                ax_sum.plot(m_vals, pos_avgs, marker="o", color=c, linewidth=2.3, label=f"Watermarked Avg Score (θ={t_str})")
                ax_sum.plot(m_vals, neg_avgs, marker="s", color=c, linestyle="--", linewidth=1.4, alpha=0.75, label=f"Clean Ref Avg Score (θ={t_str})")

        ax_sum.axhline(0.0, color="black", linestyle=":", linewidth=0.9, alpha=0.6)
        ax_sum.set_title(f"[Summary] File Average Watermark Score vs. Amplitude Multiplier (m) — {fid}", fontsize=10.5, fontweight="bold", loc="left")
        ax_sum.set_xlabel("Amplitude Multiplier (m)", fontsize=10)
        ax_sum.set_ylabel("File Average Score", fontsize=9.5)
        if all_m_ticks:
            ax_sum.set_xticks(sorted(all_m_ticks))
        ax_sum.grid(True, linestyle="--", alpha=0.4)
        ax_sum.legend(loc="best", fontsize=8.5, ncol=2)

        fig.suptitle(f"Iteration 4 Amplitude Scaling Analysis — {fid}", fontsize=13, fontweight="bold", y=0.999)
        fig.tight_layout(rect=[0, 0, 1, 0.997])

        out_png = os.path.join(per_file_dir, f"{stem}_all_multipliers.png")
        fig.savefig(out_png, dpi=130)
        plt.close(fig)

    print(f"[DONE] Saved {len(all_file_ids)} per-file stacked plots to '{per_file_dir}'.")


def main():
    parser = argparse.ArgumentParser(
        description="Aggregate Iteration 4 detection results across all multipliers (m) and thetas (θ)."
    )
    parser.add_argument(
        "--config", "-c",
        default="../config/iteration4_config.yaml",
        help="Path to iteration4_config.yaml (default: ../config/iteration4_config.yaml)"
    )
    parser.add_argument(
        "--detection_dir", "-d",
        default=None,
        help="Override path to '<general_output_path>/detection' directory containing '<theta>/<m>/subchunk_scores.json'"
    )
    parser.add_argument(
        "--output_dir", "-o",
        default=None,
        help="Directory to save aggregated plots and CSVs (defaults to '<general_output_path>/aggregated_results')"
    )
    parser.add_argument(
        "-w", "--smooth_window",
        type=int,
        default=8,
        help="Moving average window in subchunks for time-series plots (8 subchunks ≈ 0.43s; 1 to disable)"
    )
    parser.add_argument(
        "--ignore_last_chunk",
        action="store_true",
        help="Ignore all subchunks belonging to the last chunk of each file"
    )
    parser.add_argument(
        "--max_file_plots",
        type=int,
        default=None,
        help="Optional limit on the number of per-file 17-panel plots to generate (default: all files)"
    )

    args = parser.parse_args()

    general_out = None
    if os.path.isfile(args.config):
        with open(args.config, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        general_out = cfg.get("general", {}).get("output_path")

    detection_root = args.detection_dir
    if detection_root is None:
        if general_out is None:
            print("[ERROR] Could not resolve detection directory from config. Pass --detection_dir explicitly.")
            sys.exit(1)
        detection_root = os.path.join(general_out, "detection")

    if not os.path.isdir(detection_root):
        print(f"[ERROR] Detection root directory not found: {detection_root}")
        sys.exit(1)

    out_dir = args.output_dir or os.path.join(general_out or os.path.dirname(detection_root), "aggregated_results")
    os.makedirs(out_dir, exist_ok=True)

    grid, thetas, multipliers = discover_detection_grid(detection_root)
    if not grid:
        print(f"[ERROR] No '<theta>/<m>/subchunk_scores.json' files found inside '{detection_root}'.")
        sys.exit(1)

    print(f"[INFO] Found {len(grid)} (θ, m) detection runs across θ={thetas} and m={multipliers}.")
    if args.ignore_last_chunk:
        print("[INFO] --ignore_last_chunk enabled: excluding final chunk from all calculations and plots.")

    all_file_ids = load_all_data(grid, ignore_last_chunk=args.ignore_last_chunk)
    print(f"[INFO] Loaded {len(all_file_ids)} unique files across the dataset.")

    # 1. Export CSV statistics (p5..p95, mean, std for each file, theta, and m)
    export_statistics_csvs(grid, thetas, multipliers, all_file_ids, out_dir)

    # 2. Plot whole-dataset Average Score vs. Multiplier (m) separately for each theta (plus combined)
    plot_dataset_average_vs_multiplier(grid, thetas, multipliers, all_file_ids, out_dir)

    # 3. Plot per-file 16 stacked time-series graphs + 17th modifier-vs-average-score graph
    files_to_plot = all_file_ids[:args.max_file_plots] if args.max_file_plots else all_file_ids
    plot_per_file_stacked(grid, thetas, multipliers, files_to_plot, out_dir, smooth_window=args.smooth_window)


if __name__ == "__main__":
    main()
