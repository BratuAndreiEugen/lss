import os
import sys
import csv
import json
import yaml
import argparse
import numpy as np


def normalize_theta_key(key):
    """
    Converts YAML keys under 'detection' into a clean theta string (e.g. '0.18', '0.36').
    Note: In YAML 1.1 (PyYAML), unquoted '0_36' has only digits 0-7 and is parsed as
    octal int 0o36 == 30! Using oct(key)[2:] recovers the exact original digits '36'.
    """
    if isinstance(key, int):
        return f"0.{oct(key)[2:]:0>2}"
    s = str(key).strip()
    if "_" in s:
        return s.replace("_", ".")
    return s


def resolve_json_path(path_str):
    """Accepts either a direct path to a .json file or a directory containing subchunk_scores.json."""
    if os.path.isdir(path_str):
        return os.path.join(path_str, "subchunk_scores.json")
    return path_str


def load_theta_jsons_from_config(config_path):
    """
    Reads 'detection.<theta_key>.subchunk_scores_path' from the YAML config file.
    Returns a list of (theta_label, json_path).
    """
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    detection_cfg = config.get("detection")
    if not detection_cfg or not isinstance(detection_cfg, dict):
        raise KeyError(
            f"Missing 'detection' section in '{config_path}'. Expected:\n"
            "detection:\n"
            "  0_18:\n"
            "    subchunk_scores_path: \"...\"\n"
            "  0_36:\n"
            "    subchunk_scores_path: \"...\""
        )

    found = []
    for raw_key, sub_cfg in detection_cfg.items():
        if not isinstance(sub_cfg, dict) or "subchunk_scores_path" not in sub_cfg:
            continue
        theta_label = normalize_theta_key(raw_key)
        json_path = resolve_json_path(sub_cfg["subchunk_scores_path"])
        if not os.path.isfile(json_path):
            raise FileNotFoundError(f"subchunk_scores.json not found for θ={theta_label} at: {json_path}")
        found.append((theta_label, json_path))

    if not found:
        raise ValueError(f"No valid 'subchunk_scores_path' entries found under 'detection' in '{config_path}'.")

    return found, config


def load_files_by_id(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    by_id = {}
    for item in data.get("files", []):
        fid = item["file_id"]
        pos_scores = np.array([sc["score"] for sc in (item.get("subchunks_pos") or [])], dtype=float)
        neg_scores = np.array([sc["score"] for sc in (item.get("subchunks_neg") or [])], dtype=float)
        by_id[fid] = {
            "not_tampered": neg_scores,   # -c: watermarked & not-tampered
            "tampered": pos_scores,       # -i: watermarked & tampered (spliced)
        }
    return by_id


def compute_stats(scores):
    if scores is None or len(scores) == 0:
        return [None, None, None, None, None, None, None]
    p5, p10, p15, p20, p25, p50, p75, p80, p85, p90, p95 = np.percentile(scores, [5, 10, 15, 20, 25, 50, 75, 80, 85, 90, 95])
    mean_val = np.mean(scores)
    std_val = np.std(scores)
    return [
        round(float(p5), 6),
        round(float(p10), 6),
        round(float(p15), 6),
        round(float(p20), 6),
        round(float(p25), 6),
        round(float(p50), 6),
        round(float(p75), 6),
        round(float(p80), 6),
        round(float(p85), 6),
        round(float(p90), 6),
        round(float(p95), 6),
        round(float(mean_val), 6),
        round(float(std_val), 6),
    ]


def export_stats_csv_from_config(config_path, output_csv=None):
    theta_jsons, config = load_theta_jsons_from_config(config_path)

    datasets = {}
    all_file_ids = []
    for theta_label, jpath in theta_jsons:
        datasets[theta_label] = load_files_by_id(jpath)
        for fid in datasets[theta_label]:
            if fid not in all_file_ids:
                all_file_ids.append(fid)

    if output_csv is None:
        output_csv = config.get("detection", {}).get("output_csv")
    if output_csv is None:
        first_json_dir = os.path.dirname(theta_jsons[0][1])
        parent_dir = os.path.dirname(first_json_dir) or "."
        output_csv = os.path.join(parent_dir, "distribution_stats.csv")

    os.makedirs(os.path.dirname(os.path.abspath(output_csv)), exist_ok=True)

    # Build header: file_id + 5 metrics for each (theta, variant)
    cases = []
    for theta_label, _ in theta_jsons:
        col_theta = theta_label.replace(".", "_")
        cases.append((theta_label, "not_tampered", f"theta_{col_theta}_not_tampered"))
        cases.append((theta_label, "tampered", f"theta_{col_theta}_tampered"))

    header = ["file_id"]
    for _, _, prefix in cases:
        header.extend([
            f"{prefix}_p5",
            f"{prefix}_p10",
            f"{prefix}_p15",
            f"{prefix}_p20",
            f"{prefix}_p25",
            f"{prefix}_p50",
            f"{prefix}_p75",
            f"{prefix}_p80",
            f"{prefix}_p85",
            f"{prefix}_p90",
            f"{prefix}_p95",
            f"{prefix}_mean",
            f"{prefix}_std",
        ])

    rows = []
    for fid in all_file_ids:
        row = [fid]
        for theta_label, variant_key, _ in cases:
            file_data = datasets.get(theta_label, {}).get(fid)
            scores = file_data.get(variant_key) if file_data else None
            row.extend(compute_stats(scores))
        rows.append(row)

    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)

    return output_csv, header, rows


def main():
    parser = argparse.ArgumentParser(
        description="Export a CSV of 5%, 50%, 95% percentiles, mean, and std dev using a YAML config file."
    )
    parser.add_argument(
        "--config", "-c",
        default="config/config_LSS.yaml",
        help="Path to YAML configuration file (default: config/config_LSS.yaml)"
    )
    parser.add_argument(
        "-o", "--output_csv",
        default=None,
        help="Output CSV file path (overrides config or default location)"
    )
    args = parser.parse_args()

    if not os.path.isfile(args.config):
        print(f"[ERROR] Configuration file not found: {args.config}")
        sys.exit(1)

    out_csv, header, rows = export_stats_csv_from_config(args.config, args.output_csv)
    print(f"[DONE] Wrote {len(rows)} file rows ({len(header)} columns) to: {out_csv}")


if __name__ == "__main__":
    main()
