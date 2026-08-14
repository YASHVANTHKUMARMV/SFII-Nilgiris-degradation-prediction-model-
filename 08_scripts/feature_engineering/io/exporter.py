import os
import logging
import numpy as np
import xarray as xr
import pandas as pd

logger = logging.getLogger("FeatureEng.Exporter")

# 16. Export to NumPy
def export_numpy(da: xr.DataArray, output_path: str):
    """Saves arrays in raw binary format for ultra-fast DataLoader ingestion in PyTorch."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    logger.info(f"Exporting tensor to NumPy format at {output_path}...")
    
    # Warning: For massive out-of-core datasets, calling .values will load entirely into RAM.
    # Safe usage requires chunked saving or guaranteeing it fits in memory.
    np.save(output_path, da.values)
    logger.info("NumPy export complete.")

# 17. Export to Parquet
def export_parquet(da: xr.DataArray, output_path: str):
    """Flattens the array into tabular columnar format, highly optimized for XGBoost/Random Forest."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    logger.info(f"Exporting to Parquet format at {output_path}...")
    
    # Convert DataArray to a pandas DataFrame. 
    # .to_dataframe() unrolls dimensions (time, y, x, feature) into a MultiIndex.
    df = da.to_dataframe(name="value").reset_index()
    
    # Drop NaNs to save space, common in masked satellite data
    df = df.dropna()
    
    df.to_parquet(output_path, index=False, engine='pyarrow')
    logger.info("Parquet export complete.")

# 18. Export to CSV
def export_csv(da: xr.DataArray, output_path: str):
    """
    Legacy tabular export. Highly inefficient for dense 3D arrays. 
    Implemented specifically for 2D flattened subsets or point-based ablation studies.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    logger.info(f"Exporting subset to CSV format at {output_path}...")
    
    df = da.to_dataframe(name="value").reset_index()
    df = df.dropna()
    
    df.to_csv(output_path, index=False)
    logger.info("CSV export complete.")
