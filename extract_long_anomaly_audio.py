import os
import sys
import json
import yaml
import argparse
import numpy as np
import torch
import torchaudio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


CONFIG_NAMES = [
    "theta_0_18_not_tampered",
    "theta_0_18_tampered",
    "theta_0_36_not_tampered",
    "theta_0_36_tampered",
]


def region_duration_ms(region):
    if "start_ms" in region and "end_ms" in region:
        return round(float(region["end_ms"]) - float(region["start_ms"]), 3)
    if "start_sec" in region and "end_sec" in region:
        return round((float(region["end_sec"]) - float(region["start_sec"])) * 1000.0, 3)
    return 0.0


def sanitize_type_for_filename(anom_type):
    """Converts '<p5' -> 'below_p5' and '>p95' -> 'above_p95' for valid Windows filenames."""
    return anom_type.replace("<", "below_").replace(">", "above_")


def moving_average(values, window):
    if window <= 1 or len(values) < window:
        return values
    kernel = np.ones(window) / window
    return np.convolve(values, kernel, mode="same")


def load_detection_subchunks(subchunk_scores_path, variant):
    """
    Loads the original detection subchunk_scores.json and returns a dict:
    file_id -> list of subchunk dicts (from 'subchunks_neg' if not_tampered, else 'subchunks_pos').
    """
    if not subchunk_scores_path or not os.path.isfile(subchunk_scores_path):
        return {}
    with open(subchunk_scores_path, "r", encoding="utf-8") as f:
        det_data = json.load(f)

    field = "subchunks_neg" if variant == "not_tampered" else "subchunks_pos"
    by_file = {}
    for item in det_data.get("files", []):
        by_file[item["file_id"]] = item.get(field) or []
    return by_file


def extract_region_subchunks(region, detection_by_file, frames_per_sec=75.0):
    """
    Returns a list of (duration_sec, score) for each subchunk inside 'region',
    either from the original detection output or from 'subchunk_scores' stored in the region.
    """
    orig_file = region["original_file"]
    s_frame = int(region["start_frame"])
    e_frame = int(region["end_frame"])

    file_subchunks = detection_by_file.get(orig_file, [])
    if file_subchunks:
        matched = []
        for sc in file_subchunks:
            sc_s = int(sc["start_frame"])
            sc_e = int(sc["end_frame"])
            if sc_s >= s_frame and sc_e <= e_frame:
                dur_sec = (sc_e - sc_s) / frames_per_sec
                matched.append((dur_sec, float(sc["score"])))
        if matched:
            return matched

    # Fallback if subchunk_scores list was embedded directly in the region
    scores = region.get("subchunk_scores", [])
    if scores:
        total_dur_sec = (e_frame - s_frame) / frames_per_sec
        step_sec = total_dur_sec / len(scores)
        return [(step_sec, float(s)) for s in scores]

    return []


def plot_concatenated_subchunks(t_sec, y_scores, title_str, out_png_path, smooth_window=8):
    """
    Plots the concatenated subchunk scores across time in the exact style of plot_subchunks.py,
    without reference, threshold, or spliced region shading.
    """
    if len(t_sec) == 0 or len(y_scores) == 0:
        return

    mean_score = float(np.mean(y_scores))
    fig, ax = plt.subplots(figsize=(12, 5))

    if smooth_window > 1 and len(y_scores) >= smooth_window:
        ax.plot(t_sec, y_scores, color="#1f77b4", alpha=0.30, linewidth=1.0, label="Concatenated Subchunks (53ms raw)")
        y_smooth = moving_average(y_scores, smooth_window)
        ax.plot(
            t_sec, y_smooth, color="#1f77b4", linewidth=2.2,
            label=f"Smoothed ({smooth_window}-subchunk avg, mean={mean_score:.4f})"
        )
    else:
        ax.plot(
            t_sec, y_scores, color="#1f77b4", linewidth=1.5,
            label=f"Subchunk Score (mean={mean_score:.4f})"
        )

    ax.axhline(0.0, color="black", linestyle=":", linewidth=0.8, alpha=0.5)

    ax.set_title(f"Watermark Score Across Time — {title_str}", fontsize=11, fontweight="bold")
    ax.set_xlabel("Time (seconds)", fontsize=10)
    ax.set_ylabel("Subchunk Watermark Score", fontsize=10)
    ax.set_xlim(0, max(float(t_sec[-1]) + 0.05, 0.5))
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(loc="upper right", fontsize=8.5)

    fig.tight_layout()
    fig.savefig(out_png_path, dpi=150)
    plt.close(fig)


def process_configuration(anomaly_json_path, out_base_dir, min_duration_ms=500.0, descending=True, smooth_window=8):
    with open(anomaly_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    config_name = data.get("configuration", os.path.splitext(os.path.basename(anomaly_json_path))[0])
    variant = data.get("variant", "not_tampered" if "not_tampered" in config_name else "tampered")
    audio_folder = data.get("audio_folder")
    subchunk_scores_path = data.get("subchunk_scores_path")
    total_files = data.get("total_files_processed", 0)
    lower_limit = data.get("lower_limit_percentile", 5)
    upper_limit = data.get("upper_limit_percentile", 95)
    all_regions = data.get("regions", [])

    if not audio_folder or not os.path.isdir(audio_folder):
        print(f"[ERROR] Audio folder not found for '{config_name}': {audio_folder}")
        return

    detection_by_file = load_detection_subchunks(subchunk_scores_path, variant)

    min_type = f"<p{lower_limit}"
    max_type = f">p{upper_limit}"

    print("=" * 80)
    print(f" CONFIGURATION: {config_name} (Source folder: {audio_folder})")
    print("=" * 80)

    wav_cache = {}
    manifest_summary = {
        "configuration": config_name,
        "min_duration_ms": min_duration_ms,
        "sort_order": "descending" if descending else "ascending",
        "total_dataset_files": total_files,
        "types": {},
    }

    for anom_type in [min_type, max_type]:
        # 1. Filter regions of this type with duration >= min_duration_ms
        qualifying = []
        for reg in all_regions:
            if reg.get("type") != anom_type:
                continue
            dur_ms = region_duration_ms(reg)
            if dur_ms >= min_duration_ms:
                reg_copy = dict(reg)
                reg_copy["duration_ms"] = dur_ms
                qualifying.append(reg_copy)

        # 2. Sort regions by length
        qualifying.sort(key=lambda r: r["duration_ms"], reverse=descending)

        # 3. Extract and concatenate audio slices + subchunk scores in length-sorted order
        segments = []
        used_regions = []
        matched_files = set()
        target_sr = None

        concat_t_midpoints = []
        concat_scores = []
        running_time_sec = 0.0

        for reg in qualifying:
            orig_filename = reg["original_file"]
            src_wav_path = os.path.join(audio_folder, orig_filename)
            if not os.path.isfile(src_wav_path):
                print(f"[WARN] Source WAV missing: {src_wav_path}")
                continue

            if orig_filename not in wav_cache:
                wav, sr = torchaudio.load(src_wav_path)
                if wav.shape[0] > 1:
                    wav = wav.mean(dim=0, keepdim=True)
                wav_cache[orig_filename] = (wav, sr)
            else:
                wav, sr = wav_cache[orig_filename]

            if target_sr is None:
                target_sr = sr
            elif sr != target_sr:
                wav = torchaudio.functional.resample(wav, sr, target_sr)

            total_samples = wav.shape[-1]
            s_idx = max(0, int(round(reg["start_sec"] * target_sr)))
            e_idx = min(total_samples, int(round(reg["end_sec"] * target_sr)))

            if e_idx > s_idx:
                segments.append(wav[:, s_idx:e_idx])
                used_regions.append(reg)
                matched_files.add(orig_filename)

                # Append this region's subchunk scores along the concatenated timeline
                reg_subchunks = extract_region_subchunks(reg, detection_by_file)
                for sc_dur_sec, sc_score in reg_subchunks:
                    midpoint = running_time_sec + (sc_dur_sec / 2.0)
                    concat_t_midpoints.append(midpoint)
                    concat_scores.append(sc_score)
                    running_time_sec += sc_dur_sec

        safe_type = sanitize_type_for_filename(anom_type)
        out_wav_name = f"{config_name}_{safe_type}.wav"
        out_wav_path = os.path.join(out_base_dir, out_wav_name)
        out_png_name = f"{config_name}_{safe_type}.png"
        out_png_path = os.path.join(out_base_dir, out_png_name)

        total_dur_sec = 0.0
        if segments and target_sr is not None:
            concat_wav = torch.cat(segments, dim=-1)
            torchaudio.save(out_wav_path, concat_wav, target_sr)
            total_dur_sec = round(concat_wav.shape[-1] / target_sr, 3)
            saved_status = out_wav_path

            if concat_scores:
                plot_concatenated_subchunks(
                    t_sec=np.array(concat_t_midpoints, dtype=float),
                    y_scores=np.array(concat_scores, dtype=float),
                    title_str=f"{out_wav_name} ({anom_type}, {len(used_regions)} regions)",
                    out_png_path=out_png_path,
                    smooth_window=smooth_window,
                )
                plot_status = out_png_path
            else:
                plot_status = "No subchunk scores available for plot"
        else:
            saved_status = "No matching regions (no WAV written)"
            plot_status = "No matching regions (no plot written)"

        manifest_summary["types"][anom_type] = {
            "output_wav": out_wav_path if segments else None,
            "output_plot": out_png_path if (segments and concat_scores) else None,
            "matching_files_count": len(matched_files),
            "matching_regions_count": len(used_regions),
            "total_concatenated_duration_sec": total_dur_sec,
            "matching_files": sorted(matched_files),
            "regions_sorted_by_length": used_regions,
        }

        print(f"  [Type {anom_type:<5}] (length >= {min_duration_ms:.1f} ms)")
        print(f"    Matching Files Found   : {len(matched_files):>4d} / {total_files} files")
        print(f"    Matching Regions Found : {len(used_regions):>4d} regions (total concatenated: {total_dur_sec:.2f}s)")
        print(f"    Output WAV             : {saved_status}")
        print(f"    Output Plot            : {plot_status}")
        print("-" * 80)

    manifest_path = os.path.join(out_base_dir, f"manifest_{config_name}.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest_summary, f, indent=4)
    print(f"  Manifest saved to: {manifest_path}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Concatenate anomaly regions (>= min_duration_ms) sorted by length into 8 single WAV files and 8 subchunk score plots."
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
        help="Directory to save the 8 concatenated WAV files and 8 plots (defaults to '<json_dir>/concatenated_anomaly_audio')"
    )
    parser.add_argument(
        "--min_duration_ms",
        type=float,
        default=500.0,
        help="Minimum region duration in milliseconds (>= this value, default: 500.0)"
    )
    parser.add_argument(
        "-w", "--smooth_window",
        type=int,
        default=8,
        help="Moving average window in subchunks to overlay smoothed curve (8 subchunks ≈ 0.43s; 1 to disable)"
    )
    parser.add_argument(
        "--ascending",
        action="store_true",
        help="Sort regions from shortest to longest (default is longest to shortest)"
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

    out_base_dir = args.output_dir or os.path.join(json_dir, "concatenated_anomaly_audio")
    os.makedirs(out_base_dir, exist_ok=True)

    found_any = False
    for cfg_name in CONFIG_NAMES:
        json_path = os.path.join(json_dir, f"anomalies_{cfg_name}.json")
        if os.path.isfile(json_path):
            found_any = True
            process_configuration(
                anomaly_json_path=json_path,
                out_base_dir=out_base_dir,
                min_duration_ms=args.min_duration_ms,
                descending=not args.ascending,
                smooth_window=args.smooth_window,
            )
        else:
            print(f"[WARN] Anomaly JSON not found: {json_path}")

    if not found_any:
        print(f"[ERROR] No 'anomalies_*.json' files found in '{json_dir}'. Run parse_anomaly_regions.py first.")
        sys.exit(1)


if __name__ == "__main__":
    main()
