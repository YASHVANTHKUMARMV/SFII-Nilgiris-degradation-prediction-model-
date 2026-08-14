import logging
import xarray as xr
import rioxarray as rxr

logger = logging.getLogger("EnvInt.ClimateBaseline")

def align_climate_baseline(coarse_da: xr.DataArray, target_da: xr.DataArray) -> xr.DataArray:
    """
    Stage 1 Implementation:
    Resamples coarse native climate variables (~4km/11km) to the exact 10m Sentinel-2 grid
    using bilinear interpolation.
    
    Args:
        coarse_da: Raw climate DataArray (e.g., TerraClimate VPD).
        target_da: Sentinel-2 10m reference DataArray for alignment.
        
    Returns:
        xr.DataArray: 10m aligned climate data.
    """
    logger.info(f"Aligning coarse climate grid ({coarse_da.rio.resolution()}) to baseline target...")
    
    # Reproject Match handles CRS conversion, extent clipping, and resampling all at once.
    # Resampling=2 is Bilinear interpolation in rasterio.
    aligned = coarse_da.rio.reproject_match(target_da, resampling=2)
    
    logger.info("Baseline alignment complete.")
    return aligned
