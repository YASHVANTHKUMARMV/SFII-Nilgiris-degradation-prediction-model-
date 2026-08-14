import logging
import numpy as np
import xarray as xr
from statsmodels.tsa.seasonal import STL
from scipy import stats

logger = logging.getLogger("Disturbance.BFAST")

def detect_breaks_bfast_approx(da: xr.DataArray, period: int = 12) -> xr.Dataset:
    """
    Python approximation of BFAST structural break detection.
    Applies STL (Seasonal and Trend decomposition using Loess) to remove seasonality,
    then uses a rolling z-score/MOSUM approach on the residuals to find breaks.
    
    Args:
        da: xarray DataArray representing a stacked time series (e.g., NDVI) with a 'time' dimension.
        period: Seasonality period (e.g., 12 for monthly).
        
    Returns:
        xr.Dataset containing break timing and magnitudes.
    """
    logger.info("Starting BFAST-approx structural break detection...")
    
    # We define a function to apply along the time axis for each pixel
    def pixel_bfast(ts):
        if np.isnan(ts).all():
            return np.array([np.nan, np.nan]) # break_idx, magnitude
            
        # 1. STL Decomposition
        try:
            # handle NaNs by linear interpolation for STL
            ts_clean = pd.Series(ts).interpolate().bfill().ffill().values
            res = STL(ts_clean, period=period, robust=True).fit()
            trend = res.trend
            
            # 2. Structural break detection (Simplified MOSUM / CUSUM on trend derivative)
            diff = np.diff(trend)
            z_scores = np.abs(stats.zscore(diff))
            
            # Find the largest break that exceeds a threshold (e.g., Z > 3)
            break_candidates = np.where(z_scores > 3.0)[0]
            
            if len(break_candidates) > 0:
                # Get the largest one
                largest_break_idx = break_candidates[np.argmax(np.abs(diff[break_candidates]))]
                magnitude = diff[largest_break_idx]
                return np.array([largest_break_idx, magnitude])
            else:
                return np.array([np.nan, 0.0])
                
        except Exception:
            return np.array([np.nan, np.nan])

    # Apply across the DataArray (assuming spatial dims x, y)
    # Using xr.apply_ufunc for dask-parallelized execution
    breaks = xr.apply_ufunc(
        pixel_bfast,
        da,
        input_core_dims=[['time']],
        output_core_dims=[['metrics']],
        vectorize=True,
        dask='parallelized',
        output_dtypes=[float],
        dask_gufunc_kwargs={'output_sizes': {'metrics': 2}}
    )
    
    # Assign coordinates
    ds = xr.Dataset({
        'break_index': breaks.isel(metrics=0),
        'magnitude': breaks.isel(metrics=1)
    })
    
    return ds
