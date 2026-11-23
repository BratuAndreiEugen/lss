import os
import sys
import csv
import json
import yaml
import argparse


def normalize_theta_key(key):
    """
    Normalizes YAML keys like '0_18' or unquoted 0_36 into ('0.18', '0_18') / ('0.36', '0_36').
    Note: In YAML 1.1 (PyYAML), unquoted '0_36' has only digits 0-7 and is parsed as
    octal int 0o36 == 30! Using oct(key)[2:] recovers the exact original digits '36'.
    """
    if isinstance(key, int):
        dot_str = f"0.{oct(key)[2:]:0>2}"
    else:
        s = str(key).strip()
        dot_str = s.replace("_", ".") if "_" in s else s
    underscore_str = dot_str.replace(".", "_")
    return dot_str, underscore_str


def get_theta_subconfig(section_dict, target_underscore):
    """Looks up a theta sub-dictionary (e.g. '0_18' or int 18) inside a YAML section."""
    if not isinstance(section_dict, dict):
        return {}
    for k, v in section_dict.items():
        if isinstance(v, dict):
            _, u_str = normalize_theta_key(k)
            if u_str == target_underscore:
                return v
    return {}


def resolve_json_path(path_str):
    if os.path.isdir(path_str):
        return os.path.join(path_str, "subchunk_scores.json")
    return path_str


def load_csv_percentiles(csv_path):
    """
    Loads distribution_stats.csv into a dict keyed by file_id -> {column_name: float_value}.
    """
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"Statistics CSV not found at: {csv_path}")

    stats_by_file = {}
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            fid = row["file_id"]
            parsed_row = {}
            for k, v in row.items():
                if k == "file_id" or v is None or v == "":
                    continue
                try:
                    parsed_row[k] = float(v)
                except ValueError:
                    pass
            stats_by_file[fid] = parsed_row
    return stats_by_file


def extract_merged_regions(subchunks, lower_val, upper_val, lower_type, upper_type, orig_file, sample_rate=24000, frames_per_sec=75.0):
    """
    Finds consecutive subchunks below lower_val (<pX) or above upper_val (>pY),
    merging adjacent out-of-limit subchunks only if they share the exact same type.
    """
    samples_per_frame = int(round(sample_rate / frames_per_sec))  # 320 samples/frame at 24kHz
    merged_regions = []
    current_region = None

    for idx, sc in enumerate(subchunks):
        score = float(sc["score"])
        if score < lower_val:
            anom_type = lower_type
        elif score > upper_val:
            anom_type = upper_type
        else:
            anom_type = None

        if anom_type is None:
            if current_region is not None:
                merged_regions.append(current_region)
                current_region = None
            continue

        s_frame = int(sc["start_frame"])
        e_frame = int(sc["end_frame"])
        s_sec = round(s_frame / frames_per_sec, 6)
        e_sec = round(e_frame / frames_per_sec, 6)
        s_ms = round(s_sec * 1000.0, 3)
        e_ms = round(e_sec * 1000.0, 3)
        s_sample = s_frame * samples_per_frame
        e_sample = e_frame * samples_per_frame

        # Merge with current_region if consecutive and of the exact same type
        if (
            current_region is not None
            and current_region["type"] == anom_type
            and current_region["end_frame"] == s_frame
        ):
            current_region["end_frame"] = e_frame
            current_region["end_sec"] = e_sec
            current_region["end_ms"] = e_ms
            current_region["end_sample"] = e_sample
            current_region["num_subchunks"] += 1
            current_region["subchunk_scores"].append(round(score, 8))
        else:
            if current_region is not None:
                merged_regions.append(current_region)
            current_region = {
                "original_file": orig_file,
                "type": anom_type,
                "start_sample": s_sample,
                "end_sample": e_sample,
                "start_sec": s_sec,
                "end_sec": e_sec,
                "start_ms": s_ms,
                "end_ms": e_ms,
                "start_frame": s_frame,
                "end_frame": e_frame,
                "num_subchunks": 1,
                "subchunk_scores": [round(score, 8)],
            }

    if current_region is not None:
        merged_regions.append(current_region)

    return merged_regions


def main():
    arg_parser = argparse.ArgumentParser(
        description="Parse subchunk scores against CSV percentile limits and save merged anomaly regions to 4 JSON files."
    )
    arg_parser.add_argument(
        "--config", "-c",
        default="config/anomaly_pipeline_config.yaml",
        help="Path to YAML configuration file (default: config/anomaly_pipeline_config.yaml)"
    )
    # CLI overrides for all variables under 'parser:' in anomaly_pipeline_config.yaml
    arg_parser.add_argument("--lower_limit", type=int, default=None, help="Lower percentile limit (e.g. 5 or 10)")
    arg_parser.add_argument("--upper_limit", type=int, default=None, help="Upper percentile limit (e.g. 95 or 90)")
    arg_parser.add_argument("--csv_path", type=str, default=None, help="Override parser.csv_path")
    arg_parser.add_argument(
        "--watermarked_files_path_0_18", type=str, default=None,
        help="Override parser.0_18.watermarked_files_path"
    )
    arg_parser.add_argument(
        "--tampered_files_path_0_18", type=str, default=None,
        help="Override parser.0_18.tampered_files_path"
    )
    arg_parser.add_argument(
        "--watermarked_files_path_0_36", type=str, default=None,
        help="Override parser.0_36.watermarked_files_path"
    )
    arg_parser.add_argument(
        "--tampered_files_path_0_36", type=str, default=None,
        help="Override parser.0_36.tampered_files_path"
    )
    arg_parser.add_argument(
        "-o", "--output_dir", type=str, default=None,
        help="Directory to save the 4 output JSON files (defaults to directory of csv_path)"
    )
    arg_parser.add_argument(
        "--sample_rate", type=int, default=24000,
        help="Audio sample rate used to compute sample coordinates (default: 24000)"
    )

    args = arg_parser.parse_args()

    if not os.path.isfile(args.config):
        print(f"[ERROR] Configuration file not found: {args.config}")
        sys.exit(1)

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    detection_cfg = config.get("detection", {})
    parser_cfg = config.get("parser", {})

    # Resolve general parser parameters (CLI overrides YAML)
    lower_limit = args.lower_limit if args.lower_limit is not None else int(parser_cfg.get("lower_limit", 10))
    upper_limit = args.upper_limit if args.upper_limit is not None else int(parser_cfg.get("upper_limit", 90))
    csv_path = args.csv_path or parser_cfg.get("csv_path")

    if not csv_path:
        print("[ERROR] 'csv_path' must be specified in config under 'parser' or via --csv_path.")
        sys.exit(1)

    lower_type = f"<p{lower_limit}"
    upper_type = f">p{upper_limit}"

    # Load CSV percentiles
    stats_by_file = load_csv_percentiles(csv_path)

    out_dir = args.output_dir or os.path.dirname(os.path.abspath(csv_path))
    os.makedirs(out_dir, exist_ok=True)

    # CLI overrides map for per-theta audio folders
    cli_folder_overrides = {
        ("0_18", "not_tampered"): args.watermarked_files_path_0_18,
        ("0_18", "tampered"):     args.tampered_files_path_0_18,
        ("0_36", "not_tampered"): args.watermarked_files_path_0_36,
        ("0_36", "tampered"):     args.tampered_files_path_0_36,
    }

    # Discover all configured theta keys from detection section
    theta_keys = []
    for k, v in detection_cfg.items():
        if isinstance(v, dict) and "subchunk_scores_path" in v:
            dot_str, u_str = normalize_theta_key(k)
            theta_keys.append((dot_str, u_str, resolve_json_path(v["subchunk_scores_path"])))

    for dot_theta, u_theta, subchunk_json_path in theta_keys:
        if not os.path.isfile(subchunk_json_path):
            print(f"[ERROR] subchunk_scores.json not found for θ={dot_theta}: {subchunk_json_path}")
            continue

        with open(subchunk_json_path, "r", encoding="utf-8") as f:
            subchunk_data = json.load(f)

        theta_parser_cfg = get_theta_subconfig(parser_cfg, u_theta)

        variants = [
            (
                "not_tampered",
                "subchunks_neg",
                cli_folder_overrides.get((u_theta, "not_tampered")) or theta_parser_cfg.get("watermarked_files_path"),
            ),
            (
                "tampered",
                "subchunks_pos",
                cli_folder_overrides.get((u_theta, "tampered")) or theta_parser_cfg.get("tampered_files_path"),
            ),
        ]

        for variant_name, subchunk_field, audio_folder in variants:
            col_prefix = f"theta_{u_theta}_{variant_name}"
            lower_col = f"{col_prefix}_p{lower_limit}"
            upper_col = f"{col_prefix}_p{upper_limit}"

            # Optional filter: if audio_folder exists on disk, restrict/verify against files in that folder
            valid_folder_files = None
            if audio_folder and os.path.isdir(audio_folder):
                valid_folder_files = {
                    f for f in os.listdir(audio_folder)
                    if not f.startswith(".") and f.lower().endswith(".wav")
                }

            all_regions = []
            files_processed = 0

            for file_entry in subchunk_data.get("files", []):
                fid = file_entry["file_id"]
                if valid_folder_files is not None and fid not in valid_folder_files:
                    continue

                file_stats = stats_by_file.get(fid)
                if not file_stats:
                    print(f"[WARN] File '{fid}' not found in CSV '{csv_path}'. Skipping.")
                    continue

                if lower_col not in file_stats or upper_col not in file_stats:
                    raise KeyError(
                        f"Columns '{lower_col}' and/or '{upper_col}' not found in '{csv_path}'. "
                        f"Available columns: {list(file_stats.keys())}"
                    )

                lower_val = file_stats[lower_col]
                upper_val = file_stats[upper_col]

                subchunks = file_entry.get(subchunk_field) or []
                regions = extract_merged_regions(
                    subchunks=subchunks,
                    lower_val=lower_val,
                    upper_val=upper_val,
                    lower_type=lower_type,
                    upper_type=upper_type,
                    orig_file=fid,
                    sample_rate=args.sample_rate,
                )
                all_regions.extend(regions)
                files_processed += 1

            output_payload = {
                "configuration": col_prefix,
                "theta": float(dot_theta),
                "variant": variant_name,
                "audio_folder": audio_folder,
                "subchunk_scores_path": subchunk_json_path,
                "csv_path": csv_path,
                "lower_limit_percentile": lower_limit,
                "upper_limit_percentile": upper_limit,
                "total_files_processed": files_processed,
                "total_anomaly_regions": len(all_regions),
                "regions": all_regions,
            }

            out_json_name = f"anomalies_{col_prefix}.json"
            out_json_path = os.path.join(out_dir, out_json_name)
            with open(out_json_path, "w", encoding="utf-8") as f:
                json.dump(output_payload, f, indent=4)

            print(f"[DONE] Saved {len(all_regions)} regions across {files_processed} files -> {out_json_path}")


if __name__ == "__main__":
    main()
