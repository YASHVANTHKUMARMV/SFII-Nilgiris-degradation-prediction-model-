import logging
import xarray as xr

logger = logging.getLogger("SFII_Preproc.Normalization")

def normalize_min_max(da: xr.DataArray, feature_range: tuple = (0, 1)) -> xr.DataArray:
    """
    Applies Min-Max normalization to the DataArray over the spatial and temporal dimensions.
    """
    logger.info(f"Applying Min-Max normalization to range {feature_range}...")
    
    # Calculate min and max (ignoring NaNs)
    da_min = da.min(skipna=True)
    da_max = da.max(skipna=True)
    
    # Avoid division by zero
    diff = da_max - da_min
    diff = diff.where(diff != 0, 1)
    
    da_norm = (da - da_min) / diff
    
    if feature_range != (0, 1):
        da_norm = da_norm * (feature_range[1] - feature_range[0]) + feature_range[0]
        
    logger.debug("Min-Max normalization complete.")
    return da_norm

def normalize_z_score(da: xr.DataArray) -> xr.DataArray:
    """
    Applies Z-score (standard) normalization.
    """
    logger.info("Applying Z-score normalization...")
    
    da_mean = da.mean(skipna=True)
    da_std = da.std(skipna=True)
    
    # Avoid division by zero
    da_std = da_std.where(da_std != 0, 1)
    
    da_norm = (da - da_mean) / da_std
    
    logger.debug("Z-score normalization complete.")
    return da_norm
