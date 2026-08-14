import ee
from config import get_roi, export_image, START_DATE, END_DATE, SCALE

def generate_climate_data():
    roi = get_roi()
    
    # 1. Rainfall (Precipitation) from CHIRPS
    chirps = ee.ImageCollection("UCSB-CHG/CHIRPS/DAILY") \
        .filterDate(START_DATE, END_DATE) \
        .filterBounds(roi)
        
    total_rainfall = chirps.select('precipitation').sum().clip(roi).rename('Total_Rainfall')
    
    # 2. SPI (Standardized Precipitation Index)
    # SPI requires long-term mean and standard deviation. 
    # We will compute a simplified anomaly as a placeholder for full SPI integration.
    long_term_chirps = ee.ImageCollection("UCSB-CHG/CHIRPS/DAILY").filterBounds(roi)
    mean_rainfall = long_term_chirps.select('precipitation').mean().clip(roi)
    std_rainfall = long_term_chirps.select('precipitation').reduce(ee.Reducer.stdDev()).clip(roi)
    
    # Simplified SPI calculation (Z-score of precipitation)
    spi = total_rainfall.subtract(mean_rainfall).divide(std_rainfall).rename('SPI')
    
    # 3. VPD (Vapor Pressure Deficit) from TerraClimate
    terraclimate = ee.ImageCollection("IDAHO_EPSCOR/TERRACLIMATE") \
        .filterDate(START_DATE, END_DATE) \
        .filterBounds(roi)
        
    vpd = terraclimate.select('vpd').mean().multiply(0.01).clip(roi).rename('VPD') # Scale factor is 0.01
    
    exports = [
        (total_rainfall, "Rainfall"),
        (spi, "SPI"),
        (vpd, "VPD")
    ]
    
    tasks = []
    for img, name in exports:
        # Climate data can be exported at a coarser scale (e.g., 1000m or 4800m) but sticking to SCALE for consistency
        task = export_image(img, name, roi, scale=1000) 
        tasks.append(task)
        
    return tasks

if __name__ == "__main__":
    ee.Initialize()
    generate_climate_data()
