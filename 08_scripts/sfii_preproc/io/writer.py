import os
import logging
import xarray as xr
import pandas as pd
import numpy as np
try:
    import torch
except ImportError:
    torch = None
from typing import Optional

logger = logging.getLogger("SFII_Preproc.Writer")

def write_cog(da: xr.DataArray, output_path: str, nodata: float = -9999.0) -> None:
    """
    Writes a DataArray to a Cloud-Optimized GeoTIFF (COG).
    
    Args:
        da (xr.DataArray): The array to write.
        output_path (str): The destination file path.
        nodata (float): NoData value to assign.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Ensure NoData is set properly for rioxarray export
    da = da.rio.write_nodata(nodata, encoded=True)
    
    logger.info(f"Writing COG to {output_path}")
    
    # We use rasterio via rioxarray, enabling tiling and compression for COGs
    da.rio.to_raster(
        output_path,
        driver="GTiff",
        tiled=True,
        blockxsize=256,
        blockysize=256,
        compress="DEFLATE",
        windowed=True # Important for dask-backed arrays
    )
    logger.info(f"Successfully wrote {output_path}")

def write_zarr(ds: xr.Dataset, output_path: str, chunk_dict: Optional[dict] = None) -> None:
    """
    Writes a Dataset to a Zarr store for ML-ready ingestion.
    
    Args:
        ds (xr.Dataset): The dataset (often stacked time series) to write.
        output_path (str): The destination directory (.zarr).
        chunk_dict (dict): Dask chunk dictionary.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    if chunk_dict:
        ds = ds.chunk(chunk_dict)
        
    logger.info(f"Writing Zarr store to {output_path}")
    
    # mode='w' overwrites if it exists
    ds.to_zarr(output_path, mode='w', consolidated=True)
    logger.info(f"Successfully wrote Zarr store to {output_path}")

def write_numpy(da: xr.DataArray, output_path: str) -> None:
    """Writes a DataArray to a NumPy .npy file."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    logger.info(f"Writing NumPy array to {output_path}")
    np.save(output_path, da.values)
    logger.info(f"Successfully wrote {output_path}")

def write_pytorch(da: xr.DataArray, output_path: str) -> None:
    """Writes a DataArray to a PyTorch .pt file."""
    if torch is None:
        logger.error("PyTorch is not installed. Cannot export to .pt")
        return
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    logger.info(f"Writing PyTorch tensor to {output_path}")
    tensor = torch.from_numpy(da.values)
    torch.save(tensor, output_path)
    logger.info(f"Successfully wrote {output_path}")

def write_csv(ds: xr.Dataset, output_path: str) -> None:
    """Writes a Dataset to a CSV file (flattens spatial dimensions)."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    logger.info(f"Writing CSV to {output_path}")
    df = ds.to_dataframe().dropna().reset_index()
    df.to_csv(output_path, index=False)
    logger.info(f"Successfully wrote {output_path}")

def write_parquet(ds: xr.Dataset, output_path: str) -> None:
    """Writes a Dataset to a Parquet file (flattens spatial dimensions)."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    logger.info(f"Writing Parquet to {output_path}")
    df = ds.to_dataframe().dropna().reset_index()
    df.to_parquet(output_path, index=False)
    logger.info(f"Successfully wrote {output_path}")
