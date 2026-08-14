import logging
import xarray as xr
import pandas as pd
import numpy as np
import os

logger = logging.getLogger("Disturbance.Metrics")

def standardize_metrics(raw_metrics: xr.Dataset, current_year: int = 2024) -> xr.Dataset:
    """
    Converts raw algorithmic outputs into the core SFII metrics.
    Assumes bands: DISTYR, DISTMAG, RECDUR, RECRATE, DISTAGE, CONF.
    """
    logger.info("Standardizing raw breaks into SFII core disturbance metrics...")
    
    # If the inputs are already named correctly from GEE export, just verify them.
    expected_vars = ['DISTYR', 'DISTMAG', 'RECDUR', 'RECRATE', 'DISTAGE', 'CONF']
    
    for var in expected_vars:
        if var not in raw_metrics.data_vars:
            logger.warning(f"Variable {var} missing from raw metrics.")
            
    return raw_metrics

def generate_parquet_database(ds: xr.Dataset, output_path: str):
    """
    Flattens the standardized xarray Dataset into a Parquet database.
    """
    logger.info("Generating disturbance history Parquet database...")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    df = ds.to_dataframe().reset_index().dropna()
    df.to_parquet(output_path, index=False)
    logger.info(f"Saved Parquet database to {output_path}")
