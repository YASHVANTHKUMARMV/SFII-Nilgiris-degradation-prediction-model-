import logging
import pandas as pd
import xarray as xr

logger = logging.getLogger("FeatureEng.Temporal")

# 6. Temporal Ordering
def order_by_time(da_list: list, dates: list) -> xr.DataArray:
    """Sorts arrays strictly by time dimension to ensure LSTM sequence integrity."""
    time_index = pd.to_datetime(dates, format="%Y%m")
    stacked = xr.concat(da_list, dim=pd.Index(time_index, name="time"))
    ordered = stacked.sortby("time")
    logger.info(f"Temporally ordered sequence from {ordered.time.values[0]} to {ordered.time.values[-1]}")
    return ordered

# 7. Missing Month Interpolation
def insert_missing_months(da: xr.DataArray) -> xr.DataArray:
    """Inserts NaN slices for months where the entire composite is missing."""
    expected_size = len(pd.date_range(start=da.time.values[0], end=da.time.values[-1], freq='MS'))
    continuous = da.resample(time="MS").asfreq()
    
    if continuous.sizes['time'] > da.sizes['time']:
        logger.warning(f"Inserted {continuous.sizes['time'] - da.sizes['time']} missing month NaN slices.")
    return continuous

# 8. Missing Pixel Interpolation
def interpolate_missing_pixels(da: xr.DataArray, limit: int = 3) -> xr.DataArray:
    """Uses temporal linear interpolation to gap-fill localized NaNs caused by cloud masks."""
    logger.info("Interpolating localized NaNs along time dimension...")
    # Requires rechunking if chunked across time
    da = da.chunk({'time': -1})
    filled = da.interpolate_na(dim="time", method="linear", limit=limit)
    return filled

# 13. Annual Aggregation
def aggregate_annual(da: xr.DataArray) -> xr.Dataset:
    """Computes yearly medians and amplitudes to filter out intra-annual seasonal noise."""
    annual_median = da.resample(time="YS").median(skipna=True)
    annual_max = da.resample(time="YS").max(skipna=True)
    annual_min = da.resample(time="YS").min(skipna=True)
    annual_amplitude = annual_max - annual_min
    
    ds = xr.Dataset({
        'median': annual_median,
        'amplitude': annual_amplitude
    })
    return ds

# 14. Monthly Aggregation
def extract_monthly_sequence(da: xr.DataArray) -> xr.DataArray:
    """Retains the dense 12-month sequence for the LSTM."""
    # Already monthly if we passed step 7, this is essentially a passthrough
    # or could compute monthly climatologies if requested.
    return da

# 15. Sliding Temporal Windows
def compute_rolling_slope(da: xr.DataArray, window_years: int = 3) -> xr.DataArray:
    """Computes rolling linear trend (e.g., 3-year NBR slope) to provide recovery velocity."""
    window_months = window_years * 12
    logger.info(f"Computing rolling slope over {window_months} months...")
    
    # Since rolling regression in xarray can be complex, we often use a simplified diff
    # or dask-mapped function. Here we approximate with a delta over the window.
    # True linear regression would require xr.apply_ufunc with scipy.stats.linregress
    rolling_delta = da - da.shift(time=window_months)
    slope = rolling_delta / window_years
    
    return slope
