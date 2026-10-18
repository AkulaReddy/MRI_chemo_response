# datasets/BreastDCEDL: the full BreastDCEDL dataset goes here

Only this README, `dataset-metadata.json` and `.gitignore` are in git; the data files are ignored.

## What "full dataset" means
BreastDCEDL, Zenodo record 18114231 (Fridman et al., *Scientific Data* 2026), licence CC BY-NC 4.0, no login.
All 2,070 patients: I-SPY1 172, I-SPY2 982, Duke 916. 1,452 of them have a pCR label.

| file | size | contents |
|---|---|---|
| `BreastDCEDL_ISPY1_min_crop.tar.gz` | 1.25 GB | I-SPY1: 172 patients, 3 DCE phases + tumour mask each |
| `BreastDCEDL_ISPY2_min_crop.tar.gz` | 13.63 GB | I-SPY2 + ACRIN-6698: 982 patients, 3 phases + mask |
| `BreastDCEDL_DUKE_min_crop.tar.gz` | 8.64 GB | Duke: 916 patients, 3 phases (boxes are in the CSV) |
| `BreastDCEDL_models.tar.gz` | 0.32 GB | the authors' ViT weights (not needed for training) |
| `BreastDCEDL_metadata_min_crop.csv` | 0.3 MB | labels, clinical data, tumour boxes, train/val/test split (also in `data/metadata/`) |
| **total** | **23.85 GB** | |

`BreastDCEDL_demo_data.tar.gz` (55 MB, 7 patients) is only a sample; it is not part of the full set.

## Option 1, recommended: download inside Kaggle (no laptop download)
Run `kaggle/vaib/step1_download_full_dataset.ipynb` three times, with `COHORT = "ispy1"`, `"ispy2"` and `"duke"`.
Save each output as a Dataset named `mnj-raw-ispy1`, `mnj-raw-ispy2` and `mnj-raw-duke`. Three Datasets are
needed only because a Kaggle notebook can save at most 20 GB of output; attached together they are the full set.

## Option 2: download here, then upload to Kaggle as ONE Dataset
```
python scripts/download_breastdcedl.py                # -> datasets/BreastDCEDL/, resumable, md5-checked
pip install kaggle                                    # put kaggle.json (Kaggle -> Settings -> API) in ~/.kaggle/
# edit dataset-metadata.json: replace YOUR_KAGGLE_USERNAME
kaggle datasets create -p datasets/BreastDCEDL        # private by default; uploads ~24 GB
```
This needs about 24 GB of free disk. Measured from this laptop on 2026-10-18: 0.23 MB/s from Zenodo (8 parallel
connections), so allow over a day for the download, then the upload time on top. Rerunning the script resumes.
The step-2 notebook finds the archives by file name in any attached Dataset, so both options work.
