import logging
import xarray as xr

logger = logging.getLogger("FeatureEng.Spatial")

# 2. CRS Verification
def verify_crs(da_list: list, target_crs: str = "EPSG:32643") -> list:
    """Prevents catastrophic coordinate drift by verifying EPSG codes."""
    valid_da = []
    for da in da_list:
        crs = da.rio.crs
        if crs is not None and crs.to_string() == target_crs:
            valid_da.append(da)
        else:
            logger.error(f"CRS mismatch in {da.attrs.get('source_file', 'unknown')}. Expected {target_crs}.")
    return valid_da

# 3. Spatial Alignment & 5. Resolution Harmonization
def harmonize_and_align(da: xr.DataArray, target_extent: tuple, target_resolution: tuple) -> xr.DataArray:
    """
    Ensures every pixel refers to the exact same 10x10m ground footprint.
    Resamples natively 20m bands (e.g. RedEdge) to 10m to match NDVI.
    """
    minx, miny, maxx, maxy = target_extent
    x_res, y_res = target_resolution
    
    # Clip first to reduce data size before potential resampling
    aligned = da.rio.clip_box(minx=minx, miny=miny, maxx=maxx, maxy=maxy)
    
    # Check resolution
    current_res = aligned.rio.resolution()
    if abs(current_res[0]) != abs(x_res) or abs(current_res[1]) != abs(y_res):
        logger.info(f"Resampling from {current_res} to {target_resolution}")
        # Resample to match target
        aligned = aligned.rio.reproject(
            aligned.rio.crs,
            resolution=target_resolution,
            resampling=5 # Average resampling
        )
        
    return aligned

# 4. Pixel Grid Consistency Checking
def verify_grid_consistency(da_list: list) -> bool:
    """Validates that all index arrays have identical [X, Y] dimensions."""
    if not da_list:
        return False
        
    ref_shape = (da_list[0].sizes['x'], da_list[0].sizes['y'])
    
    for da in da_list[1:]:
        shape = (da.sizes['x'], da.sizes['y'])
        if shape != ref_shape:
            logger.error(f"Grid mismatch: {shape} != {ref_shape} in {da.attrs.get('source_file')}")
            return False
            
    logger.info(f"Grid consistency verified across {len(da_list)} arrays. Shape: {ref_shape}")
    return True
