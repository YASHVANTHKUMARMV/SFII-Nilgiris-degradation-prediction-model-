import os
import glob
import logging
import xarray as xr
import rioxarray as rxr
from typing import List

logger = logging.getLogger("FeatureEng.Loader")

# 1. Raster loading pipeline
def load_geotiffs(directory: str, pattern: str = "*.tif", chunk_size: dict = None) -> List[xr.DataArray]:
    """
    Efficiently loads out-of-core rasters using xarray and dask to prevent memory overflows.
    """
    if chunk_size is None:
        chunk_size = {'x': 1024, 'y': 1024}
        
    search_path = os.path.join(directory, pattern)
    files = sorted(glob.glob(search_path))
    
    if not files:
        logger.warning(f"No files found matching {pattern} in {directory}")
        return []
        
    da_list = []
    for f in files:
        try:
            # Load lazily with rioxarray
            da = rxr.open_rasterio(f, chunks=chunk_size, masked=True)
            # Tag with filename for debugging/temporal ordering
            da.attrs['source_file'] = os.path.basename(f)
            da_list.append(da)
        except Exception as e:
            logger.error(f"Failed to load {f}: {e}")
            
    logger.info(f"Loaded {len(da_list)} rasters from {directory}")
    return da_list
