import os
import yaml
import logging
from utils.logger import setup_logger
from utils.checkpoints import CheckpointManager
from io.reader import find_geotiffs, read_raster_chunked, parse_date_from_filename
from io.writer import write_cog, write_zarr, write_numpy, write_pytorch, write_csv, write_parquet
from io.metadata import generate_processing_metadata
from core.quality import verify_dataset_quality
from core.alignment import align_raster
from core.cloud_mask import apply_cloud_mask
from core.stacking import stack_monthly_rasters, handle_missing_months
from core.interpolation import gap_fill
from core.normalization import normalize_min_max, normalize_z_score
from core.features import compute_indices, generate_annual_summaries
from core.qc_reports import generate_qc_report

def load_config(config_path: str = "config/settings.yaml") -> dict:
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def run_pipeline():
    config = load_config()
    
    # 1. Initialize
    log_dir = config['directories']['logs']
    logger = setup_logger(log_dir)
    logger.info(f"Starting {config['project']['name']} v{config['project']['version']}")
    
    ckpt_dir = config['directories']['checkpoints']
    ckpt = CheckpointManager(ckpt_dir)
    
    raw_dir = config['directories']['raw_data']
    target_crs = config['spatial']['target_crs']
    target_extent = tuple(config['spatial']['target_extent'])
    target_resolution = tuple(config['spatial']['target_resolution'])
    
    # 2. Find and Validate Files
    if not ckpt.is_completed("quality_assessment"):
        files = find_geotiffs(raw_dir, pattern="S2L2A_NILGIRIS_10m_*_monthly_UTM43N.tif")
        if not files:
            logger.error(f"No Sentinel-2 monthly composites found in {raw_dir}")
            return
            
        valid_files, invalid_files = verify_dataset_quality(files, target_crs)
        if len(valid_files) == 0:
            logger.error("No valid files passed quality assessment. Exiting.")
            return
            
        ckpt.set("valid_files", valid_files)
        ckpt.mark_completed("quality_assessment")
    else:
        valid_files = ckpt.get("valid_files")
        logger.info(f"Loaded {len(valid_files)} valid files from checkpoint.")
        
    # 3. Reading, Masking, and Aligning
    if not ckpt.is_completed("alignment_and_stacking"):
        da_list = []
        dates = []
        
        chunk_opts = config['processing']['dask_chunk_size']
        
        for fp in valid_files:
            date_str = parse_date_from_filename(fp)
            if not date_str:
                logger.warning(f"Could not parse date from {fp}. Skipping.")
                continue
                
            da = read_raster_chunked(fp, chunk_size={'x': chunk_opts['x'], 'y': chunk_opts['y']})
            
            # Cloud mask (convert NoData to NaN)
            da = apply_cloud_mask(da)
            
            # Align
            da = align_raster(da, target_extent, target_resolution)
            
            da_list.append(da)
            dates.append(date_str)
            
        # 4. Stacking
        stacked_da = stack_monthly_rasters(da_list, dates)
        
        # 5. Handle missing months
        start_date = min(dates)
        end_date = max(dates)
        stacked_da = handle_missing_months(stacked_da, start_date, end_date)
        
        ckpt.mark_completed("alignment_and_stacking")
        # In a real heavy pipeline, we might write the intermediate stacked_da to disk here.
    else:
        logger.info("Alignment and stacking already completed. (Skipping intermediate load in this skeleton)")
        # Normally you would load the intermediate Zarr/NetCDF here.
        return 

    # 6. Interpolation
    if not ckpt.is_completed("interpolation"):
        # We need to rechunk to allow interpolation over time
        stacked_da = stacked_da.chunk({'time': -1})
        interp_method = config['processing']['interpolation_method']
        stacked_da = gap_fill(stacked_da, method=interp_method)
        ckpt.mark_completed("interpolation")

    # 7. Feature Generation & Normalization
    if not ckpt.is_completed("features_and_normalization"):
        # We convert DataArray to Dataset to compute multiple indices
        if isinstance(stacked_da, xr.DataArray):
            if 'band' in stacked_da.dims:
                # Map bands if possible or just use band numbers
                # Depending on the dataset, we might need to map them properly.
                # Assuming here the dataset already has data variables or we create them.
                ds = stacked_da.to_dataset(dim='band')
                # Optional: rename bands if they are 1, 2, 3...
                # ds = ds.rename({1: 'B2', 2: 'B3', 3: 'B4', 4: 'B8', 5: 'B11', 6: 'B12'})
            else:
                ds = stacked_da.to_dataset(name="features")
        else:
            ds = stacked_da

        # Compute derived indices
        ds = compute_indices(ds)
        
        # Normalize
        norm_method = config['processing']['normalization_method']
        for var in ds.data_vars:
            if norm_method == "min-max":
                ds[var] = normalize_min_max(ds[var])
            elif norm_method == "z-score":
                ds[var] = normalize_z_score(ds[var])
                
        # Generate annual summaries
        ds_annual = generate_annual_summaries(ds)
        
        ckpt.set("ds", ds)
        ckpt.set("ds_annual", ds_annual)
        ckpt.mark_completed("features_and_normalization")
    else:
        ds = ckpt.get("ds")
        ds_annual = ckpt.get("ds_annual")

    # 8. Export
    if not ckpt.is_completed("export"):
        export_fmt = config['processing']['export_format']
        output_dir_ml = config['directories']['ml_features']
        dataset_name = "s2_monthly_stacked_ml_ready"
        
        # Export as required
        if export_fmt in ["zarr", "both"]:
            zarr_path = os.path.join(output_dir_ml, f"{dataset_name}.zarr")
            write_zarr(ds, zarr_path, chunk_dict={'time': -1, 'x': 512, 'y': 512})
            
        if export_fmt in ["cog", "both"]:
            cog_path = os.path.join(config['directories']['processed_data'], f"{dataset_name}_latest.tif")
            # Write only the first variable's last timestep for demonstration
            if len(ds.data_vars) > 0:
                first_var = list(ds.data_vars.keys())[0]
                write_cog(ds[first_var].isel(time=-1), cog_path)

        # Export ML tensors & Dataframes
        numpy_path = os.path.join(output_dir_ml, f"{dataset_name}.npy")
        write_numpy(ds.to_array(), numpy_path)
        
        pytorch_path = os.path.join(output_dir_ml, f"{dataset_name}.pt")
        write_pytorch(ds.to_array(), pytorch_path)
        
        csv_path = os.path.join(output_dir_ml, f"{dataset_name}.csv")
        write_csv(ds_annual, csv_path)
        
        parquet_path = os.path.join(output_dir_ml, f"{dataset_name}.parquet")
        write_parquet(ds_annual, parquet_path)
            
        # Write metadata
        stats = {
            "time_steps": ds.sizes.get('time', 0),
            "x_size": ds.sizes.get('x', 0),
            "y_size": ds.sizes.get('y', 0),
            "variables": list(ds.data_vars.keys()),
            "normalization": config['processing']['normalization_method']
        }
        generate_processing_metadata(output_dir_ml, dataset_name, stats)
        
        # Write QC report
        qc_dir = config['directories']['logs']
        generate_qc_report(ds, qc_dir, dataset_name)
        
        ckpt.mark_completed("export")

    logger.info("Pipeline execution completed successfully.")

if __name__ == "__main__":
    run_pipeline()
