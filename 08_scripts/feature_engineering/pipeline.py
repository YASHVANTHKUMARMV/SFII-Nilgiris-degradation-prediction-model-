import os
import yaml
from utils.qa_report import setup_logger, generate_quality_report
from io.loader import load_geotiffs
from core.spatial import verify_crs, harmonize_and_align, verify_grid_consistency
from core.temporal import order_by_time, insert_missing_months, interpolate_missing_pixels
from core.features import detect_cloud_artifacts, normalize_zscore, scale_minmax, stack_features
from io.exporter import export_numpy, export_parquet, export_csv

def load_config(config_path: str = "config/settings.yaml") -> dict:
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def run_feature_engineering_pipeline():
    config = load_config()
    
    # Setup logging
    log_dir = config['directories']['logs']
    logger = setup_logger(log_dir)
    logger.info("Starting Phase 5: Feature Engineering Pipeline")
    
    # 1. Raster loading
    raw_dir = config['directories']['input_geotiffs']
    da_list = load_geotiffs(raw_dir)
    if not da_list:
        logger.error("No input files found. Exiting.")
        return
        
    # 2. CRS Verification
    target_crs = config['spatial']['target_crs']
    valid_da = verify_crs(da_list, target_crs)
    
    # 3 & 5. Spatial Alignment and Harmonization
    target_extent = tuple(config['spatial']['target_extent'])
    target_resolution = tuple(config['spatial']['target_resolution'])
    aligned_da = [harmonize_and_align(da, target_extent, target_resolution) for da in valid_da]
    
    # 4. Pixel Grid Consistency
    if not verify_grid_consistency(aligned_da):
        logger.error("Grid consistency failed. Exiting.")
        return
        
    # Assume we extract dates from filenames (e.g. YYYYMM) to order them
    # For this skeleton, we assume `aligned_da` is naturally sorted by filename
    # 6. Temporal Ordering
    dates = [f"2018{str(i).zfill(2)}" for i in range(1, len(aligned_da)+1)] # Mock dates
    time_series = order_by_time(aligned_da, dates)
    
    # 7. Missing Month Interpolation
    time_series = insert_missing_months(time_series)
    
    # 9. Cloud Artifact Verification
    time_series = detect_cloud_artifacts(time_series)
    
    # 8. Missing Pixel Interpolation
    time_series = interpolate_missing_pixels(time_series)
    
    # 10 & 11. Feature Normalization/Scaling (Simulated dictionary of features)
    # Normally we would process NDVI, NBR, TCW separately. 
    # Here we simulate the processing of one index.
    if config['features']['scaling_method'] == "z-score":
        normalized = normalize_zscore(time_series)
    else:
        normalized = scale_minmax(time_series)
        
    # 12. Feature Stacking
    features_dict = {'NDVI': normalized} # Would include NBR, EVI2, etc.
    stacked_tensor = stack_features(features_dict)
    
    # 19 & 20. Quality Reports
    report_dir = config['directories']['reports']
    # Convert back to Dataset briefly for reporting
    ds_report = stacked_tensor.to_dataset(dim='feature')
    generate_quality_report(ds_report, report_dir)
    
    # 16, 17, 18. Exports
    # For demonstration, we export the stacked tensor
    export_numpy(stacked_tensor, os.path.join(config['directories']['output_numpy'], 'features.npy'))
    
    # Export Parquet/CSV requires 2D tabular formats, which is highly memory intensive for the full stack.
    # Usually performed on a spatial subset or downsampled version.
    subset = stacked_tensor.isel(time=slice(0, 10), x=slice(0, 100), y=slice(0, 100))
    export_parquet(subset, os.path.join(config['directories']['output_parquet'], 'features.parquet'))
    export_csv(subset, os.path.join(config['directories']['output_csv'], 'features_subset.csv'))
    
    logger.info("Feature Engineering Pipeline completed successfully.")

if __name__ == "__main__":
    run_feature_engineering_pipeline()
