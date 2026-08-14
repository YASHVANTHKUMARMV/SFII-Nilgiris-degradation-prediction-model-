# SFII Geospatial Preprocessing Pipeline

This Python package converts raw downloaded GeoTIFFs (Sentinel-2, Sentinel-1, GEDI) into a stacked, normalized, and machine-learning-ready dataset (e.g., Zarr arrays) for the Structural Forest Integrity Index (SFII) project.

## Architecture

* `core/`: Geospatial algorithms (alignment, gap-filling, normalization).
* `io/`: Reading and writing chunked out-of-core rasters.
* `utils/`: Logging and checkpoint state management.
* `config/`: Configuration parameters (`settings.yaml`).

## Requirements

Ensure you have a Conda environment or virtualenv with the dependencies installed. The pipeline heavily relies on `xarray`, `rioxarray`, and `dask` to manage memory when processing the large (~470MB) Sentinel-2 monthly composites.

```bash
pip install -r requirements.txt
```

## Usage

1. Review and adjust parameters in `config/settings.yaml`.
2. Ensure raw data is in the directory specified in the config.
3. Run the pipeline:

```bash
python pipeline.py
```

## Features
- **Out-of-core processing:** Uses Dask to prevent RAM crashes.
- **Checkpointing:** If the pipeline crashes, restarting `pipeline.py` will resume from the last completed stage.
- **Auto Gap-filling:** Linearly interpolates missing pixels (e.g., from cloud masks) across the temporal dimension.
- **ML Ready Output:** Outputs consolidated `.zarr` stores optimized for PyTorch/TensorFlow DataLoaders.
