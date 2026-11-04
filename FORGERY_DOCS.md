### First Iteration

#### Dataset

From the original dataset with 960 files with 40 speakers, we picked 5 files for each speaker. This constitutes our dataset. The script for this is in `balanced_picker.py`.

#### Splicing Strategy
For each speaker a set of candidate files from the original dataset was made. A candidate file is a file of that speaker which was not picked into our small dataset.
The following was done for each file in the picked dataset:
1. Select a random 1s region.
2. Select a random candidate file of the same speaker from the original dataset.
3. Select a random 1s region from this candidate file.
4. Replace the region in the file from our dataset with the region from the candidate file.
5. Append to `metadata.json` as follows:
```json
{
    "seed": 6582090840510761152,
    "total_tampered_files": 200,
    "tampered_recordings": [
        {
            "tampered_file_id": "tampered_0000_spk61.wav",
            "original_file": "0000_spk61_parent0000_0000000000_0000320000_approx10s_16k.wav",
            "tampered_region": {
                "start_sec": 6.33025,
                "end_sec": 7.33025,
                "start_sample": 101284,
                "end_sample": 117284
            },
            "donor_file": "0014_spk61_parent0014_0004480000_0004800000_approx10s_16k.wav",
            "donor_region": {
                "start_sec": 0.517,
                "end_sec": 1.517,
                "start_sample": 8272,
                "end_sample": 24272
            }
        },
```
*This is all done in the `splicer.py` script. At the top of `metadata.json` the seed for the random operations is saved for replicability*