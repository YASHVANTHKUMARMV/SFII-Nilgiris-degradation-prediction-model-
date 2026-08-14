import logging
import xarray as xr
import numpy as np
from scipy.ndimage import gradient_magnitude

logger = logging.getLogger("EnvInt.Topography")

def compute_terrain_derivatives(dem_da: xr.DataArray) -> xr.Dataset:
    """
    Computes Slope and Aspect from a high-resolution DEM (e.g., Copernicus 30m).
    
    Returns:
        xr.Dataset containing 'elevation', 'slope', 'aspect_sin', 'aspect_cos'.
    """
    logger.info("Computing topographic derivatives (Slope, Aspect)...")
    
    # Note: Proper slope calculation requires projected coordinate systems (meters).
    # We assume dem_da is in EPSG:32643
    
    # Simple gradient-based approximation for pipeline structure
    # In practice, xarray-spatial or richdem is preferred for exact geomorphology
    dy, dx = np.gradient(dem_da.values, axis=(1, 2))
    
    # Calculate slope in radians, then convert to degrees
    slope_rad = np.arctan(np.sqrt(dx**2 + dy**2))
    slope_deg = np.degrees(slope_rad)
    
    # Calculate aspect (0 is North)
    aspect_rad = np.arctan2(dy, -dx)
    
    # Create Continuous Heat-Insolation proxies (Sine/Cosine)
    aspect_sin = np.sin(aspect_rad)
    aspect_cos = np.cos(aspect_rad)
    
    # Repackage into DataArrays
    slope_da = xr.DataArray(slope_deg, coords=dem_da.coords, dims=dem_da.dims)
    asin_da = xr.DataArray(aspect_sin, coords=dem_da.coords, dims=dem_da.dims)
    acos_da = xr.DataArray(aspect_cos, coords=dem_da.coords, dims=dem_da.dims)
    
    ds = xr.Dataset({
        'elevation': dem_da,
        'slope': slope_da,
        'aspect_sin': asin_da,
        'aspect_cos': acos_da
    })
    
    logger.info("Topographic derivatives computed successfully.")
    return ds
