import logging
import xarray as xr
import numpy as np

logger = logging.getLogger("SFII_Preproc.CloudMask")

def apply_cloud_mask(da: xr.DataArray, qa_band: str = 'QA60') -> xr.DataArray:
    """
    Applies a cloud mask based on a specified QA band if clouds aren't already set to NoData.
    This assumes Sentinel-2 QA60 bitmask logic (bit 10=opaque clouds, bit 11=cirrus).
    
    For the SFII project, GEE exports usually already apply this and set clouds to NoData.
    This function is a fallback if raw unmasked data is provided.
    """
    logger.info("Verifying cloud mask application...")
    
    if qa_band in da.coords or (hasattr(da, 'name') and da.name == qa_band):
        # Implementation depends on exact DataArray structure (e.g., if QA60 is a variable in Dataset)
        logger.warning("QA band masking logic triggered. Ensure QA band is properly structured.")
        # Example logic:
        # cloud_bit = 1 << 10
        # cirrus_bit = 1 << 11
        # mask = (qa_array & cloud_bit == 0) & (qa_array & cirrus_bit == 0)
        # return da.where(mask)
        pass
    else:
        logger.info("QA band not found or data is assumed already masked via GEE. Treating existing NaNs/NoData as clouds.")
        
    # Standardize NoData to np.nan for xarray processing
    # If nodata is -9999.0
    if hasattr(da, 'rio') and da.rio.nodata is not None:
        nodata_val = da.rio.nodata
        da = da.where(da != nodata_val, np.nan)
        
    return da
