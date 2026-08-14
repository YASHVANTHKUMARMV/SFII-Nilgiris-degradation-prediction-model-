# Environmental Variable Integration

This package executes Phase 7 of the SFII workflow: integrating multi-source environmental drivers into the 10m Sentinel-2 structural grid.

## Architecture

- `core/topography.py`: Computes slopes and continuous aspect proxies (sine/cosine) from the Copernicus DEM.
- `core/anthropogenic.py`: Log-transforms Euclidean distances to roads/settlements to capture non-linear human footprint decay.
- `core/climate_baseline.py`: (Stage 1) Rapid bilinear interpolation of coarse climate variables to the target grid.
- `core/climate_downscaled.py`: (Stage 2) Experimental module to statistically downscale macro-climate variables to micro-scales using Random Forests.

## Usage
Run the pipeline to fuse the topographic, anthropogenic, and climatic drivers into the final ML tensors.
```bash
python pipeline.py
```
