import os
import sys
import shutil
import argparse
import torch
import torchaudio
from tqdm import tqdm


def get_valid_wav_files(folder):
    """Returns a sorted list of valid .wav files, ignoring hidden metadata files."""
    files = []
    for f in os.listdir(folder):
        if f.startswith("._") or f.startswith("."):
            continue
        full_path = os.path.join(folder, f)
        if os.path.isfile(full_path) and f.lower().endswith(".wav"):
            files.append(f)
    return sorted(files)


def main():
    parser = argparse.ArgumentParser(
        description="Multiply the signal amplitude of all WAV files in a dataset and save them with identical names to a new folder."
    )
    parser.add_argument(
        "-i", "--input_dir",
        required=True,
        help="Path to the input dataset directory (or a single WAV file)"
    )
    parser.add_argument(
        "-o", "--output_dir",
        required=True,
        help="Path to the output directory where scaled WAV files will be saved"
    )
    parser.add_argument(
        "-m", "--multiplier",
        type=float,
        required=True,
        help="Amplitude multiplier factor (e.g. 0.5, 2.0)"
    )
    parser.add_argument(
        "--clamp",
        action="store_true",
        help="Optional: clamp the scaled waveform values to [-1.0, 1.0] before saving"
    )

    args = parser.parse_args()

    if not os.path.exists(args.input_dir):
        print(f"[ERROR] Input path not found: {args.input_dir}")
        sys.exit(1)

    os.makedirs(args.output_dir, exist_ok=True)

    if os.path.isfile(args.input_dir):
        input_base = os.path.dirname(args.input_dir)
        files = [os.path.basename(args.input_dir)]
    else:
        input_base = args.input_dir
        files = get_valid_wav_files(args.input_dir)

    if not files:
        print(f"[ERROR] No valid .wav files found in: {args.input_dir}")
        sys.exit(1)

    print(f"[INFO] Scaling amplitude of {len(files)} file(s) by multiplier = {args.multiplier}...")

    for fname in tqdm(files, desc="Scaling Amplitude"):
        in_path = os.path.join(input_base, fname)
        out_path = os.path.join(args.output_dir, fname)

        if args.multiplier == 1.0:
            shutil.copy2(in_path, out_path)
            continue

        info = torchaudio.info(in_path)
        wav, sr = torchaudio.load(in_path)
        wav_scaled = wav * args.multiplier

        if args.clamp:
            wav_scaled = torch.clamp(wav_scaled, -1.0, 1.0)

        torchaudio.save(
            out_path,
            wav_scaled,
            sr,
            encoding=info.encoding,
            bits_per_sample=info.bits_per_sample,
        )

    print(f"[DONE] Saved {len(files)} scaled WAV file(s) to '{args.output_dir}'.")


if __name__ == "__main__":
    main()
