import ee

from config import get_roi
from importlib import import_module

modules = [
    '01_dem_terrain',
    '02_sentinel1_sar',
    '03_landtrendr',
    '04_distance_metrics',
    '05_climate_data'
]

def authenticate_and_initialize():
    try:
        ee.Initialize()
        print("Google Earth Engine initialized successfully.")
    except Exception as e:
        print("Initialization failed. Authenticating...")
        ee.Authenticate()
        ee.Initialize()
        print("Google Earth Engine authenticated and initialized.")

def run_pipeline():
    authenticate_and_initialize()
    
    all_tasks = []
    
    print("--- Starting GEE Data Generation Pipeline ---")
    
    dem_terrain = import_module('01_dem_terrain')
    print("Running DEM & Terrain module...")
    all_tasks.extend(dem_terrain.generate_dem_terrain())
    
    sentinel1_sar = import_module('02_sentinel1_sar')
    print("Running Sentinel-1 SAR module...")
    all_tasks.extend(sentinel1_sar.generate_sentinel1_sar())
    
    landtrendr = import_module('03_landtrendr')
    print("Running LandTrendr module...")
    all_tasks.extend(landtrendr.generate_landtrendr())
    
    distance_metrics = import_module('04_distance_metrics')
    print("Running Distance Metrics module...")
    all_tasks.extend(distance_metrics.generate_distance_metrics())
    
    climate_data = import_module('05_climate_data')
    print("Running Climate Data module...")
    all_tasks.extend(climate_data.generate_climate_data())
    
    print("\n--- Pipeline Execution Complete ---")
    print(f"Total Export Tasks Started: {len(all_tasks)}")
    print("Please check your Google Earth Engine Tasks tab or Google Drive for the outputs.")

if __name__ == "__main__":
    run_pipeline()
