import logging
import xarray as xr
import numpy as np

logger = logging.getLogger("EnvInt.Anthropogenic")

def compute_log_distance(distance_da: xr.DataArray) -> xr.DataArray:
    """
    Applies log transformation to Euclidean distance arrays (e.g., distance to roads)
    to model the non-linear decay of human influence.
    
    Args:
        distance_da: Array of distances in meters.
        
    Returns:
        xr.DataArray: Log-transformed distance, normalized to [0, 1].
    """
    logger.info(f"Computing log-distance transformation for anthropogenic feature...")
    
    # Add 1 to avoid log(0)
    log_dist = np.log1p(distance_da)
    
    # Min-Max normalize
    d_min = log_dist.min(skipna=True)
    d_max = log_dist.max(skipna=True)
    
    norm_dist = (log_dist - d_min) / (d_max - d_min)
    
    return norm_dist
