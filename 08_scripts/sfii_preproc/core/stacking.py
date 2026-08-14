import logging
import xarray as xr
import pandas as pd
import numpy as np

logger = logging.getLogger("SFII_Preproc.Stacking")

def stack_monthly_rasters(da_list: list, dates: list) -> xr.DataArray:
    """
    Takes a list of DataArrays and corresponding date strings (YYYYMM)
    and stacks them along a new 'time' dimension.
    """
    logger.info(f"Stacking {len(da_list)} rasters along time dimension...")
    
    # Convert string dates (YYYYMM) to datetime objects
    time_index = pd.to_datetime(dates, format="%Y%m")
    
    # Stack along time dimension
    stacked = xr.concat(da_list, dim=pd.Index(time_index, name="time"))
    
    # Sort by time just in case
    stacked = stacked.sortby("time")
    
    logger.info(f"Successfully stacked rasters. Time dimension size: {stacked.sizes['time']}")
    return stacked

def handle_missing_months(da_stacked: xr.DataArray, start_date: str, end_date: str) -> xr.DataArray:
    """
    Identifies missing months in a monthly stacked DataArray within a date range.
    Resamples the array to a continuous monthly frequency ('MS') inserting NaNs where missing.
    """
    logger.info(f"Checking for missing months between {start_date} and {end_date}...")
    
    # Resample to start of month frequency. This inserts NaNs for missing months.
    # Interpolation of these NaNs is handled in the interpolation module.
    continuous = da_stacked.resample(time="MS").asfreq()
    
    original_size = da_stacked.sizes['time']
    new_size = continuous.sizes['time']
    
    if new_size > original_size:
        logger.warning(f"Detected {new_size - original_size} missing months. Inserted NaN slices.")
    else:
        logger.info("No missing months detected in the time series.")
        
    return continuous
