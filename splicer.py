import os
import re
import sys
import json
import wave
import random
import argparse
from collections import defaultdict
import torchaudio

def get_valid_wav_files(folder):
    """Returns a sorted list of valid audio files, ignoring hidden macOS metadata files."""
    files = []
    for f in os.listdir(folder):
        if f.startswith("._") or f.startswith("."):
            continue
        if os.path.isfile(os.path.join(folder, f)):
            files.append(f)
    return sorted(files)

def main():
    parser = argparse.ArgumentParser(description="Tamper 1s regions of 200 recordings using same-speaker donor files.")
    parser.add_argument("selected_folder", help="Path to the folder with the 200 extracted recordings")
    parser.add_argument("full_dataset_folder", help="Path to the full source dataset (960 recordings)")
    parser.add_argument("output_folder", help="Path to save the tampered wav files")
    parser.add_argument("json_output", help="Path to save the output JSON metadata file (e.g., metadata.json)")
    parser.add_argument("--seed", type=int, default=None, help="Optional random seed for reproducibility")
    
    args = parser.parse_args()

    seed_used = args.seed if args.seed is not None else random.randrange(sys.maxsize)
    random.seed(seed_used)
    print(f"Using random seed: {seed_used}")

    os.makedirs(args.output_folder, exist_ok=True)
    speaker_pattern = re.compile(r"_(spk\d+)_")

    selected_files = get_valid_wav_files(args.selected_folder)
    selected_set = set(selected_files)

    available_donors = defaultdict(list)
    for f in get_valid_wav_files(args.full_dataset_folder):
        if f not in selected_set:
            match = speaker_pattern.search(f)
            if match:
                speaker_id = match.group(1)
                available_donors[speaker_id].append(f)

    for spk in available_donors:
        random.shuffle(available_donors[spk])

    records = []

    for idx, orig_filename in enumerate(selected_files):
        match = speaker_pattern.search(orig_filename)
        if not match:
            print(f"Skipping {orig_filename}: Could not detect speaker ID.")
            continue
        
        speaker_id = match.group(1)

        if not available_donors[speaker_id]:
            raise RuntimeError(f"Ran out of unused donor recordings for speaker {speaker_id}!")

        donor_filename = available_donors[speaker_id].pop()

        orig_path = os.path.join(args.selected_folder, orig_filename)
        donor_path = os.path.join(args.full_dataset_folder, donor_filename)

        orig_wav, sample_rate = torchaudio.load(orig_path)
        donor_wav, donor_sr = torchaudio.load(donor_path)

        # Resample clean donor (16kHz) to match watermarked file (24kHz)
        if donor_sr != sample_rate:
            donor_wav = torchaudio.functional.resample(donor_wav, donor_sr, sample_rate)

        orig_total_frames = orig_wav.shape[-1]
        donor_total_frames = donor_wav.shape[-1]
        frames_1s = sample_rate

        if orig_total_frames < frames_1s or donor_total_frames < frames_1s:
            raise ValueError(f"File shorter than 1 second: {orig_filename} or {donor_filename}")

        orig_start_frame = random.randint(0, orig_total_frames - frames_1s)
        orig_end_frame = orig_start_frame + frames_1s

        donor_start_frame = random.randint(0, donor_total_frames - frames_1s)
        donor_end_frame = donor_start_frame + frames_1s

        tampered_wav = orig_wav.clone()
        tampered_wav[:, orig_start_frame:orig_end_frame] = donor_wav[:, donor_start_frame:donor_end_frame]

        # Keep original filename so LSS_detection.py derives the matching nonce (or strip "tampered_" in LSS_detection.py)
        tampered_id_name = orig_filename
        tampered_path = os.path.join(args.output_folder, tampered_id_name)

        torchaudio.save(tampered_path, tampered_wav, sample_rate)

        records.append({
            "tampered_file_id": tampered_id_name,
            "original_file": orig_filename,
            "tampered_region": {
                "start_sec": round(orig_start_frame / sample_rate, 6),
                "end_sec": round(orig_end_frame / sample_rate, 6),
                "start_sample": orig_start_frame,
                "end_sample": orig_end_frame
            },
            "donor_file": donor_filename,
            "donor_region": {
                "start_sec": round(donor_start_frame / sample_rate, 6),
                "end_sec": round(donor_end_frame / sample_rate, 6),
                "start_sample": donor_start_frame,
                "end_sample": donor_end_frame
            }
        })

    output_data = {
        "seed": seed_used,
        "total_tampered_files": len(records),
        "tampered_recordings": records
    }

    with open(args.json_output, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=4)

    print(f"Done! Created {len(records)} tampered files in '{args.output_folder}'.")
    print(f"Metadata written to '{args.json_output}'.")

if __name__ == "__main__":
    main()