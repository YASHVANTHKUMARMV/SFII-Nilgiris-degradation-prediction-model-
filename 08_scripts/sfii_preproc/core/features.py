import logging
import xarray as xr
import numpy as np

logger = logging.getLogger("SFII_Preproc.Features")

def compute_indices(ds: xr.Dataset) -> xr.Dataset:
    """
    Computes standard spectral indices from a Sentinel-2 Dataset.
    Assumes bands are available as data variables (e.g., 'B2', 'B3', 'B4', 'B8', 'B11', 'B12').
    If the input is a DataArray with a 'band' dimension, it should be converted to a Dataset first.
    """
    logger.info("Computing derived spectral indices...")
    
    # We use .where() to avoid division by zero
    
    if 'B8' in ds and 'B4' in ds:
        logger.debug("Computing NDVI")
        denom = (ds['B8'] + ds['B4']).where((ds['B8'] + ds['B4']) != 0, np.nan)
        ds['NDVI'] = (ds['B8'] - ds['B4']) / denom
        
    if 'B8' in ds and 'B12' in ds:
        logger.debug("Computing NBR")
        denom = (ds['B8'] + ds['B12']).where((ds['B8'] + ds['B12']) != 0, np.nan)
        ds['NBR'] = (ds['B8'] - ds['B12']) / denom
        
    if 'B8' in ds and 'B4' in ds and 'B2' in ds:
        logger.debug("Computing EVI2")
        # EVI2 = 2.5 * (NIR - Red) / (NIR + 2.4 * Red + 1)
        denom = (ds['B8'] + 2.4 * ds['B4'] + 1).where((ds['B8'] + 2.4 * ds['B4'] + 1) != 0, np.nan)
        ds['EVI2'] = 2.5 * (ds['B8'] - ds['B4']) / denom

    if 'B8' in ds and 'B3' in ds:
        logger.debug("Computing NDWI")
        denom = (ds['B3'] + ds['B8']).where((ds['B3'] + ds['B8']) != 0, np.nan)
        ds['NDWI'] = (ds['B3'] - ds['B8']) / denom

    # Tasseled Cap Wetness (TCW) for Sentinel-2 (Crist 1985 / generic S2 coefficients)
    # Approx coefficients for B2, B3, B4, B8, B11, B12
    if all(b in ds for b in ['B2', 'B3', 'B4', 'B8', 'B11', 'B12']):
        logger.debug("Computing TCW")
        ds['TCW'] = (0.1509 * ds['B2'] + 0.1973 * ds['B3'] + 0.3279 * ds['B4'] + 
                     0.3406 * ds['B8'] - 0.7112 * ds['B11'] - 0.4572 * ds['B12'])

    logger.info("Index computation complete.")
    return ds

def generate_annual_summaries(ds: xr.Dataset, time_dim: str = 'time') -> xr.Dataset:
    """
    Generates annual feature summaries (mean, min, max, std, amplitude) from a monthly time series.
    """
    logger.info("Generating annual feature summaries...")
    
    # Group by year
    annual_groups = ds.groupby(f"{time_dim}.year")
    
    # Calculate statistics
    ds_mean = annual_groups.mean(skipna=True).rename({v: f"{v}_mean" for v in ds.data_vars})
    ds_min = annual_groups.min(skipna=True).rename({v: f"{v}_min" for v in ds.data_vars})
    ds_max = annual_groups.max(skipna=True).rename({v: f"{v}_max" for v in ds.data_vars})
    ds_std = annual_groups.std(skipna=True).rename({v: f"{v}_std" for v in ds.data_vars})
    
    # Amplitude = max - min
    ds_amp = (annual_groups.max(skipna=True) - annual_groups.min(skipna=True)).rename({v: f"{v}_amp" for v in ds.data_vars})
    
    # Merge all summaries into a single dataset
    annual_summary = xr.merge([ds_mean, ds_min, ds_max, ds_std, ds_amp])
    
    logger.info(f"Annual summaries generated with {len(annual_summary.data_vars)} features.")
    return annual_summary
