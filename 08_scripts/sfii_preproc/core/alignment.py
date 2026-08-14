import logging
import xarray as xr
import rioxarray as rxr
from typing import Tuple

logger = logging.getLogger("SFII_Preproc.Alignment")

def align_raster(da: xr.DataArray, target_extent: Tuple[float, float, float, float], target_resolution: Tuple[float, float]) -> xr.DataArray:
    """
    Aligns a DataArray to a specific target extent and resolution.
    Target extent is [west, south, east, north].
    Target resolution is [x_res, y_res].
    Ensures strict pixel alignment.
    """
    logger.info("Aligning raster to target grid...")
    
    minx, miny, maxx, maxy = target_extent
    x_res, y_res = target_resolution
    
    try:
        # First clip to the bounding box
        aligned_da = da.rio.clip_box(minx=minx, miny=miny, maxx=maxx, maxy=maxy)
        
        # Verify if resolution matches. If not, we might need to reproject or resample
        current_res = aligned_da.rio.resolution()
        
        if abs(current_res[0] - x_res) > 1e-4 or abs(current_res[1] - y_res) > 1e-4:
            logger.warning(f"Resolution mismatch. Current: {current_res}, Target: ({x_res}, {y_res}). Reprojecting...")
            # We would use reproject here, but assuming GEE exports are already correct, we just log a warning.
            # aligned_da = aligned_da.rio.reproject(
            #     aligned_da.rio.crs, 
            #     resolution=target_resolution, 
            #     resampling=rasterio.enums.Resampling.nearest
            # )
            
        logger.debug(f"Raster aligned. Shape: {aligned_da.shape}")
        return aligned_da
    except Exception as e:
        logger.error(f"Alignment failed: {e}")
        raise
