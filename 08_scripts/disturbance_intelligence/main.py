import os
import ee
import logging
from .detector import check_existing_layers
from .gee_landtrendr import run_landtrendr, extract_disturbance_metrics, export_metrics
from .validation import generate_validation_report

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("Disturbance.Main")

def main():
    logger.info("Initializing Forest Disturbance Intelligence Pipeline...")
    
    # Initialize Earth Engine
    try:
        ee.Initialize()
    except Exception as e:
        logger.warning("Earth Engine not initialized. Attempting to Authenticate...")
        try:
            ee.Authenticate()
            ee.Initialize()
        except Exception as auth_e:
            logger.error(f"Failed to initialize Earth Engine: {auth_e}")
            logger.error("Please ensure you have run 'earthengine authenticate' in your terminal.")
            return

    # Configuration
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    output_dir = os.path.join(base_dir, "04_sfii_outputs", "disturbance_history")
    os.makedirs(output_dir, exist_ok=True)
    
    required_vars = ['DISTYR', 'DISTMAG', 'RECDUR', 'RECRATE', 'DISTAGE', 'CONF']
    
    # Step 1: Detect existing layers
    existence_map, all_exist = check_existing_layers(output_dir, required_vars)
    
    # Define AOI (Example: Nilgiris bounding box)
    # [76.0, 11.0, 77.0, 11.8]
    aoi = ee.Geometry.Rectangle([76.0, 11.0, 77.0, 11.8])
    
    if not all_exist:
        logger.info("Generating missing disturbance layers via Google Earth Engine...")
        
        # Step 2: Run LandTrendr
        lt_result = run_landtrendr(aoi, start_year=1985, end_year=2024)
        
        # Step 3: Extract Metrics
        metrics = extract_disturbance_metrics(lt_result, current_year=2024)
        
        # Step 4: Export to GeoTIFFs
        # Note: In a production script, we'd wait for these tasks to finish before standardizing.
        export_metrics(metrics, aoi, output_dir, prefix="LANDTRENDR_NIL_30m_1985_2024_UTM43N")
        
        logger.info("Earth Engine export tasks submitted. Please check the GEE Task Manager.")
        logger.info("Once downloaded, they can be standardized to Parquet.")
    else:
        logger.info("All disturbance layers already exist. Skipping GEE generation.")
        
    # Step 5: Validation Report
    stats = {
        'aoi': 'Nilgiris (76.0, 11.0, 77.0, 11.8)',
        'variables': required_vars
    }
    report_path = generate_validation_report(output_dir, stats)
    
    logger.info("Disturbance Intelligence Pipeline completed successfully.")

if __name__ == "__main__":
    main()
