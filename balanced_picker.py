import os
import re
import shutil
import random
import argparse
from collections import defaultdict

def main():
    parser = argparse.ArgumentParser(description="Extract a specific number of audio recordings per speaker.")
    
    parser.add_argument("source_folder", help="Path to the folder containing the original recordings")
    parser.add_argument("destination_folder", help="Path to the folder where extracted files will be saved")
    
    parser.add_argument(
        "-n", "--files_per_speaker", 
        type=int, 
        default=5, 
        help="Number of recordings to extract per speaker (default: 5)"
    )
    parser.add_argument(
        "-r", "--random", 
        action="store_true", 
        help="Select files randomly instead of taking the first N files"
    )

    args = parser.parse_args()

    os.makedirs(args.destination_folder, exist_ok=True)

    speaker_files = defaultdict(list)
    speaker_pattern = re.compile(r"_(spk\d+)_")

    for filename in os.listdir(args.source_folder):
        # Skip macOS hidden metadata files (like ._0331_spk2094...)
        if filename.startswith("._") or filename.startswith("."):
            continue
        
        filepath = os.path.join(args.source_folder, filename)
        if not os.path.isfile(filepath):
            continue

        match = speaker_pattern.search(filename)
        if match:
            speaker_id = match.group(1)
            speaker_files[speaker_id].append(filename)

    print(f"Found {len(speaker_files)} unique speakers.")

    total_copied = 0

    for speaker_id, files in sorted(speaker_files.items()):
        files.sort()
        
        if len(files) < args.files_per_speaker:
            print(f"Warning: {speaker_id} only has {len(files)} files (expected at least {args.files_per_speaker}).")
            selected_files = files
        else:
            if args.random:
                selected_files = random.sample(files, args.files_per_speaker)
            else:
                selected_files = files[:args.files_per_speaker]
                
        for filename in selected_files:
            src_path = os.path.join(args.source_folder, filename)
            dst_path = os.path.join(args.destination_folder, filename)
            shutil.copy2(src_path, dst_path)
            total_copied += 1

    print(f"Successfully copied {total_copied} recordings to '{args.destination_folder}'.")

if __name__ == "__main__":
    main()