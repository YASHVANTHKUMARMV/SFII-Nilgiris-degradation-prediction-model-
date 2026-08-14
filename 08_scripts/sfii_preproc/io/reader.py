import os
import glob
import logging
import xarray as xr
import rioxarray as rxr
from typing import List, Optional

logger = logging.getLogger("SFII_Preproc.Reader")

def find_geotiffs(directory: str, pattern: str = "*.tif") -> List[str]:
    """Finds all GeoTIFF files matching a pattern in a directory."""
    search_path = os.path.join(directory, pattern)
    files = glob.glob(search_path)
    logger.info(f"Found {len(files)} files matching {pattern} in {directory}")
    return files

def read_raster_chunked(file_path: str, chunk_size: dict = None) -> xr.DataArray:
    """
    Reads a raster file lazily using rioxarray and dask.
    
    Args:
        file_path (str): Path to the GeoTIFF.
        chunk_size (dict): Dask chunking dimensions (e.g., {'x': 1024, 'y': 1024}).
        
    Returns:
        xr.DataArray: Dask-backed xarray DataArray.
    """
    if chunk_size is None:
        chunk_size = {'x': 1024, 'y': 1024}
        
    try:
        da = rxr.open_rasterio(file_path, chunks=chunk_size, masked=True)
        return da
    except Exception as e:
        logger.error(f"Failed to read {file_path}: {e}")
        raise

def parse_date_from_filename(filename: str) -> Optional[str]:
    """
    Extracts the YYYYMM date string from standard SFII Sentinel-2 filenames.
    Expects format like 'S2L2A_NILGIRIS_10m_201801_monthly_UTM43N.tif'
    """
    base = os.path.basename(filename)
    parts = base.split('_')
    for part in parts:
        # Looking for a 6-digit string representing YYYYMM
        if len(part) == 6 and part.isdigit():
            year = int(part[:4])
            month = int(part[4:])
            if 1980 <= year <= 2050 and 1 <= month <= 12:
                return part
    return None
