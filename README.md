# Structural Forest Integrity Index (SFII) - Nilgiris Research Project

This repository contains the complete codebase for the SFII analytical framework and machine learning pipeline, designed to detect and forecast persistent forest degradation beyond mere spectral recovery.

## Project Architecture

The repository is structured around a multi-phase data processing and machine learning pipeline, ensuring scientific reproducibility.

### Core Directories
- `data/`: Contains raw and processed satellite data.
- `03_sfii_components/`: Derived SFII rasters (SRT, SBP, DMF, ERS).
- `04_sfii_outputs/`: Final index maps, ML predictions, and forecasts.
- `05_ml/`: Contains compiled tabular feature matrices and trained models.
- `08_scripts/`: The master execution codebase, divided by domain:
  - `sfii_preproc/`: Image preprocessing and compositing.
  - `feature_engineering/`: Computation of covariates.
  - `disturbance_intelligence/`: LandTrendr and fire mask integrations.
  - `math_modeling/`: Core SFII mathematical computations.
  - `ml_pipeline/`: Model training (LSTM, Transformer, XGBoost, Random Forest).
- `paper/`: The final drafted manuscript and scientific visualization assets.

## Reproducibility
To reproduce the laboratory's findings:
1. Ensure the real datasets are completely downloaded/generated (e.g. Sentinel-1, Sentinel-2, GEDI, LandTrendr).
2. Execute preprocessing in `08_scripts/sfii_preproc/` and `08_scripts/feature_engineering/`.
3. Generate the ML dataset via `08_scripts/feature_engineering/01_build_ml_dataset.py`.
4. Execute the ML pipeline: `python 08_scripts/ml_pipeline/main.py --final`.

## Version Control & Security
A `.gitignore` file has been provided to ensure that sensitive files and large datasets are not accidentally committed to version control. This includes:
- Large data directories (`data/`, `04_sfii_outputs/`, `05_ml/`)
- Environment configurations and credentials (`.env`, `credentials.json`, `*secret*`, etc.)
- Python cache and virtual environments.
