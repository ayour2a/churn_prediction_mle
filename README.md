# Customer Churn Prediction

A machine-learning project for exploring and preparing the IBM Telco Customer
Churn dataset. The current workflow downloads the source data, creates
reproducible train/evaluation/holdout splits, and starts exploratory data
analysis. Model training and prediction are not implemented yet.

## Current workflow

1. Download the public
   [Telco Customer Churn dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
   using KaggleHub.
2. Split the records into training (70%), evaluation (15%), and holdout (15%)
   sets. The split is stratified by `Churn` and uses random seed `42`.
3. Check that the splits are complete and that no `customerID` appears in more
   than one split.
4. Explore the training and evaluation data in the EDA notebook.

## Requirements

- Python 3.10 or newer
- [uv](https://docs.astral.sh/uv/)

## Setup

From the project directory:

```powershell
uv sync --group dev
```

This installs the project dependencies and the development dependencies used
for notebooks and tests.

## Download the data

Download the raw CSV to `data/raw/telco_churn.csv`:

```powershell
uv run python -m src.feature_pipeline.load
```

The downloader skips the download when that file already exists. To force a
fresh download, call it with `force=True` from Python:

```python
from src.feature_pipeline.load import download_raw_data

download_raw_data(force=True)
```

If the automatic download is unavailable, download the CSV from the Kaggle
dataset page linked above and save it as `data/raw/telco_churn.csv`.

## Run the notebooks

Open the notebooks in `notebooks/` in VS Code with the project Python
environment selected, then run their cells in order:

- `00_data_spilt.ipynb` downloads and inspects the source data, makes the
  stratified splits, and writes `train.csv`, `eval.csv`, and `holdout.csv` to
  `data/raw/`.
- `01_EDA_cleaning.ipynb` loads the training and evaluation splits and begins
  exploratory analysis.

The split notebook must be run before the EDA notebook. Despite its filename,
the EDA notebook currently contains the initial inspection steps; a completed
cleaning and processed-data pipeline is not yet in place.

## Run tests

```powershell
uv run pytest
```

## Project layout

```text
configs/                 Configuration files (currently a placeholder)
data/
  kagglehub/             KaggleHub download cache
  raw/                   Downloaded source data and notebook-generated splits
  processed/             Reserved for processed data
models/                  Reserved for model artifacts
notebooks/
  00_data_spilt.ipynb    Download and stratified split workflow
  01_EDA_cleaning.ipynb  Initial exploratory data analysis
src/
  feature_pipeline/
    load.py              Kaggle dataset download helper
  utils/
    paths.py             Project and data directory paths
tests/                   Pytest test directory
```

Data under `data/` is ignored by Git. Run the download and notebook steps
locally to generate it.

## Planned work

The project metadata describes a broader churn-prediction application, but the
current repository does not yet include feature engineering, model training or
evaluation code, a prediction API, or a user interface.
