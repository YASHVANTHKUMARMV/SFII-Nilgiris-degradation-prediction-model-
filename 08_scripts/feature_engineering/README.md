# SFII Feature Engineering Pipeline

This Python package executes Phase 5 of the SFII workflow: converting raw monthly Sentinel-2 GeoTIFFs into machine-learning-ready tensors (NumPy) and tabular formats (Parquet/CSV).

## Architecture

- `core/spatial.py`: Handles CRS, alignment, grid consistency, and resolution harmonization.
- `core/temporal.py`: Manages temporal ordering, missing month/pixel interpolation, and temporal aggregations (annual, monthly, sliding windows).
- `core/features.py`: Normalizes, scales, identifies artifacts, and stacks the final multi-dimensional array `[Batch, Time, Height, Width, Features]`.
- `io/`: Loaders and exporters.
- `utils/`: QA reporting and robust logging.

## Usage

1. Configure parameters in `config/settings.yaml`
2. Ensure dependencies are met (`pip install -r requirements.txt`)
3. Run the pipeline: `python pipeline.py`

*Note: Parquet and CSV exports flatten the 3D spatial arrays into tabular data. This is extremely memory-intensive and should be run on subsets or distributed Dask clusters.*
