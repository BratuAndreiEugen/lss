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

#### 1 Second Splicing Report

* Set `script.bits: "01011010"`, `modes.fit_pca: true`, `modes.save_pca: true`, `run_lss.key: "0123456789abcdef0123456789abcdef"` in `config/config_LSS.yaml`
* Estimate PCA:
```bash
./estimate_pca.py -i "../LSS_forgery_project/audio_data/train-clean-100_wav" -o "../checkpoints" --max_files 25000
```
* Apply watermarks to clean recordings:
```bash
./LSS_embedding.py -i "../test_dataset" -o "../0.18_watermarked_test_dataset" --pca_load_dir "../checkpoints/PCA_20261105_1527_1337_25000.pt"
```
* Run detection on watermarked recordings:
```bash
./LSS_detection.py -i "../0.18_watermarked_test_dataset" -c "../test_dataset" --pca_load_dir "../checkpoints/PCA_20261105_1527_1337_25000.pt" -r "../1s_random_splicing/0.18_pre_splice_detection" --compute_pesq false
```
***97%** of the recordings are classified as `WATERMARKED`*

* Splice the watermarked recordings with unwatermarked sections from the full dataset:
```bash
./splicer.py "../0.18_watermarked_test_dataset" "../LSS_forgery_project/audio_data/test-clean_eval960_wordsafe/10s" "../1s_random_splicing/0.18_spliced_watermarked_test_dataset" "../1s_random_splicing/0.18_spliced_watermarked_test_dataset/metadata.json"
```
* Run detection again on the tampered recordings:
```bash
./LSS_detection.py -i "../1s_random_splicing/0.18_spliced_watermarked_test_dataset" -c "../test_dataset" --pca_load_dir "../checkpoints/PCA_20261105_1527_1337_25000.pt" -r "../1s_random_splicing/0.18_post_splice_detection" --compute_pesq false
```
***96%** of the recordings are classified as `WATERMARKED`*

### Second Iteration

#### Repeating the Last Iteration with Theta = 0.36

* Apply watermarks to clean recordings again with theta changed in `config/config_LSS.yaml`:
```bash
./LSS_embedding.py -i "../test_dataset" -o "../0.36_watermarked_test_dataset" --pca_load_dir "../checkpoints/PCA_20261105_1527_1337_25000.pt"
```
* Run detection on watermarked recordings:
```bash
./LSS_detection.py -i "../0.36_watermarked_test_dataset" -c "../test_dataset" --pca_load_dir "../checkpoints/PCA_20261105_1527_1337_25000.pt" -r "../1s_random_splicing/0.36_pre_splice_detection" --compute_pesq false
```
***99.50%** of the recordings are classified as `WATERMARKED`*

* Splice the watermarked recordings with unwatermarked sections from the full dataset (use the same seed as iteration1: `6977636903463324403`):
```bash
./splicer.py "../0.36_watermarked_test_dataset" "../LSS_forgery_project/audio_data/test-clean_eval960_wordsafe/10s" "../1s_random_splicing/0.36_spliced_watermarked_test_dataset" "../1s_random_splicing/0.36_spliced_watermarked_test_dataset/metadata.json" --seed <SEED_FROM_ITERATION_1(metadata.json)>
```
* Run detection again on the tampered recordings:
```bash
./LSS_detection.py -i "../1s_random_splicing/0.36_spliced_watermarked_test_dataset" -c "../test_dataset" --pca_load_dir "../checkpoints/PCA_20261105_1527_1337_25000.pt" -r "../1s_random_splicing/0.36_post_splice_detection" --compute_pesq false
```
***99.50%** of the recordings are classified as `WATERMARKED`*


#### Granular Scores

* We modified `lss_pca.py` & `LSS_detection.py` to return scores on a subchunk level. The output is saved in the `-r` folder in `subchunk_scores.json`:

```json
{
    "threshold": 0.0739840567111969,
    "total_files": 5,
    "files": [
        {
            "file_id": "0000_spk61_parent0000_0000000000_0000320000_approx10s_16k.wav",
            "pos_global_score": 0.09705414,
            "pos_global_classification": "WATERMARKED",
            "neg_global_score": 0.11607999,
            "neg_global_classification": "WATERMARKED",
            "subchunks_pos": [
                {
                    "chunk_idx": 0,
                    "subchunk_idx": 0,
                    "start_frame": 0,
                    "end_frame": 4,
                    "start_sec": 0.0,
                    "end_sec": 0.0533,
                    "score": -0.01319575,
                    "classification": "CLEAN"
                },
                {
                    "chunk_idx": 0,
                    "subchunk_idx": 1,
                    "start_frame": 4,
                    "end_frame": 8,
                    "start_sec": 0.0533,
                    "end_sec": 0.1067,
                    "score": 0.00012474,
                    "classification": "CLEAN"
                },
                {
                    "chunk_idx": 0,
                    "subchunk_idx": 2,
                    "start_frame": 8,
                    "end_frame": 12,
                    "start_sec": 0.1067,
                    "end_sec": 0.16,
                    "score": 0.01064525,
                    "classification": "CLEAN"
                },
            .......
            .......
```

* We picked 5 files from different speakers and extracted their watermarked and watermarked forged versions with both `theta = 0.18` and `theta = 0.36`.

* For `theta = 0.18` we ran:
    * Detection:
    ```bash
    ./LSS_detection.py -i "../1s_random_splicing/iteration2_files/0.18_watermarked_spliced" -c "../1s_random_splicing/iteration2_files/0.18_watermarked" --pca_load_dir "../checkpoints/PCA_20261105_1527_1337_25000.pt" -r "../1s_random_splicing/iteration2_files/0.18_granular_detection_output" --compute_pesq false
    ```
    * Plotting:
    ```bash
    ./plot_subchunks "../1s_random_splicing/iteration2_files/0.18_granular_detection_output/subchunk_scores.json" -o "../1s_random_splicing/iteration2_files/0.18_granular_detection_output/plots" -m "../1s_random_splicing/0.18_spliced_watermarked_test_dataset/metadata.json"   
    ```
* For `theta = 0.36` we ran:
    * Detection:
    ```bash
    ./LSS_detection.py -i "../1s_random_splicing/iteration2_files/0.36_watermarked_spliced" -c "../1s_random_splicing/iteration2_files/0.36_watermarked" --pca_load_dir "../checkpoints/PCA_20261105_1527_1337_25000.pt" -r "../1s_random_splicing/iteration2_files/0.36_granular_detection_output" --compute_pesq false
    ```
    * Plotting:
    ```bash
    ./plot_subchunks "../1s_random_splicing/iteration2_files/0.36_granular_detection_output/subchunk_scores.json" -o "../1s_random_splicing/iteration2_files/0.36_granular_detection_output/plots" -m "../1s_random_splicing/0.36_spliced_watermarked_test_dataset/metadata.json"   
    ```
* The plots for both these runs are found in `plots/0.18_plots` & `plots/0.36_plots` respectively.