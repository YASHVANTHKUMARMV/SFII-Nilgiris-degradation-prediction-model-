import ee
from config import get_roi, export_image, START_DATE, END_DATE, SCALE

def generate_sentinel1_sar():
    roi = get_roi()
    
    # Sentinel-1 SAR Collection
    s1 = ee.ImageCollection('COPERNICUS/S1_GRD') \
        .filterBounds(roi) \
        .filterDate(START_DATE, END_DATE) \
        .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV')) \
        .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH')) \
        .filter(ee.Filter.eq('instrumentMode', 'IW'))
    
    # We'll create a single mosaic or composite (e.g., median over the period) 
    # Or, normally, it's monthly. For simplicity, we calculate a median composite for the baseline,
    # but the pipeline can be adapted to monthly if required.
    s1_median = s1.median().clip(roi)
    
    vv = s1_median.select('VV')
    vh = s1_median.select('VH')
    
    # Calculate VV/VH ratio
    # In dB, division is subtraction: VV - VH (since dB is log scale)
    vv_vh_ratio = vv.subtract(vh).rename('VV_VH_Ratio')
    
    exports = [
        (s1_median, "Sentinel1_SAR_Median"),
        (vv, "Sentinel1_VV"),
        (vh, "Sentinel1_VH"),
        (vv_vh_ratio, "Sentinel1_VV_VH_Ratio")
    ]
    
    tasks = []
    for img, name in exports:
        task = export_image(img, name, roi, scale=SCALE)
        tasks.append(task)
        
    return tasks

if __name__ == "__main__":
    ee.Initialize()
    generate_sentinel1_sar()
