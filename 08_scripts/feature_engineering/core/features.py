import logging
import xarray as xr
import numpy as np

logger = logging.getLogger("FeatureEng.Features")

# 9. Cloud Artifact Verification
def detect_cloud_artifacts(da: xr.DataArray, threshold: float = -0.4) -> xr.DataArray:
    """
    Scans for anomalous extreme drops (e.g., NDVI plummeting) which indicate missed cloud shadows.
    Replaces them with NaN for subsequent interpolation.
    """
    logger.info("Scanning for unmasked cloud artifacts...")
    # E.g., if a pixel drops significantly below historical minimum or a hard threshold
    mask = da < threshold
    da_clean = da.where(~mask, np.nan)
    artifacts_found = mask.sum().compute().item()
    if artifacts_found > 0:
        logger.warning(f"Found and masked {artifacts_found} cloud artifact pixels.")
    return da_clean

# 10. Feature Normalization
def normalize_zscore(da: xr.DataArray) -> xr.DataArray:
    """Z-score standardization (mu=0, std=1) for gradient-based optimizers."""
    logger.info(f"Applying Z-score normalization to {da.name}...")
    mean = da.mean(dim='time', skipna=True)
    std = da.std(dim='time', skipna=True)
    
    # Avoid division by zero
    std = std.where(std != 0, 1)
    return (da - mean) / std

# 11. Feature Scaling
def scale_minmax(da: xr.DataArray) -> xr.DataArray:
    """Min-Max scaling [0, 1] for specific features."""
    logger.info(f"Applying Min-Max scaling to {da.name}...")
    d_min = da.min(dim='time', skipna=True)
    d_max = da.max(dim='time', skipna=True)
    
    diff = d_max - d_min
    diff = diff.where(diff != 0, 1)
    return (da - d_min) / diff

# 12. Feature Stacking
def stack_features(features_dict: dict) -> xr.Dataset:
    """
    Concatenates individual feature DataArrays into a Dataset.
    Ultimately this forms the [B, T, H, W, F] tensor in PyTorch.
    """
    logger.info(f"Stacking {len(features_dict)} features into ML tensor...")
    ds = xr.Dataset(features_dict)
    
    # In xarray, converting to an array with a new 'feature' dimension
    stacked_da = ds.to_array(dim="feature")
    # Resulting shape: [feature, time, y, x]
    # We transpose to standard ML format: [time, y, x, feature] (batch is implied later)
    ml_tensor = stacked_da.transpose('time', 'y', 'x', 'feature')
    
    logger.info(f"Stacked tensor shape: {ml_tensor.shape}")
    return ml_tensor
