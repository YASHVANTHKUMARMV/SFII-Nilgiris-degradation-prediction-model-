import logging
import rasterio
from typing import List, Tuple

logger = logging.getLogger("SFII_Preproc.Quality")

def verify_crs(file_path: str, target_crs: str = "EPSG:32643") -> bool:
    """
    Verifies that the raster matches the target CRS.
    """
    try:
        with rasterio.open(file_path) as src:
            src_crs = src.crs.to_string() if src.crs else "None"
            if src_crs != target_crs:
                logger.error(f"CRS mismatch in {file_path}: Expected {target_crs}, got {src_crs}")
                return False
            return True
    except Exception as e:
        logger.error(f"Failed to read {file_path} for CRS verification: {e}")
        return False

def verify_dataset_integrity(file_path: str) -> bool:
    """
    Verifies that the raster file is readable and not corrupted.
    """
    try:
        with rasterio.open(file_path) as src:
            # Try to read a small window to ensure it's not fully corrupted
            _ = src.read(1, window=((0, 1), (0, 1)))
            return True
    except Exception as e:
        logger.error(f"Integrity check failed for {file_path}: {e}")
        return False

def verify_dataset_quality(file_paths: List[str], target_crs: str) -> Tuple[List[str], List[str]]:
    """
    Scans a list of files and verifies their quality (integrity and CRS).
    
    Returns:
        Tuple containing a list of valid files and a list of invalid files.
    """
    valid_files = []
    invalid_files = []
    
    logger.info(f"Starting quality assessment for {len(file_paths)} files...")
    
    for fp in file_paths:
        if verify_dataset_integrity(fp) and verify_crs(fp, target_crs):
            valid_files.append(fp)
        else:
            invalid_files.append(fp)
            
    logger.info(f"Quality Assessment complete: {len(valid_files)} valid, {len(invalid_files)} invalid.")
    return valid_files, invalid_files
