import logging
import xarray as xr

logger = logging.getLogger("SFII_Preproc.Interpolation")

def gap_fill(da: xr.DataArray, method: str = "linear", max_gap: int = 3) -> xr.DataArray:
    """
    Interpolates missing (NaN) pixels along the time dimension.
    This effectively gap-fills pixels obscured by clouds/shadows or missing months.
    
    Args:
        da (xr.DataArray): The stacked DataArray.
        method (str): Interpolation method ('linear', 'nearest', 'spline').
        max_gap (int): Maximum consecutive NaNs to fill.
        
    Returns:
        xr.DataArray: The gap-filled DataArray.
    """
    logger.info(f"Applying temporal interpolation (method={method}, max_gap={max_gap})...")
    
    # Perform interpolation over the 'time' dimension.
    # We use dask-compatible interpolation if chunked properly (chunking should not be over time dimension for this).
    
    try:
        # Note: xarray's interpolate_na requires the dimension to not be chunked.
        # If 'time' is chunked, we might need to rechunk first: da = da.chunk({'time': -1})
        da_filled = da.interpolate_na(dim="time", method=method, limit=max_gap)
        logger.info("Gap filling completed successfully.")
        return da_filled
    except Exception as e:
        logger.error(f"Interpolation failed: {e}")
        logger.warning("Ensure the 'time' dimension is not chunked during interpolation.")
        raise
