import os
import json
import logging
import xarray as xr
from datetime import datetime

logger = logging.getLogger("SFII_Preproc.QC")

def generate_qc_report(ds: xr.Dataset, output_dir: str, dataset_name: str) -> None:
    """
    Generates a Quality Control (QC) report summarizing the dataset.
    This includes missing data statistics, spatial extent, and index distributions.
    """
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"Generating QC report for {dataset_name}...")
    
    report = {
        "dataset": dataset_name,
        "generated_at": datetime.now().isoformat(),
        "dimensions": {dim: size for dim, size in ds.sizes.items()},
        "variables": {}
    }
    
    total_pixels = float(ds.sizes.get('x', 1) * ds.sizes.get('y', 1) * ds.sizes.get('time', 1))
    
    for var_name, da in ds.data_vars.items():
        # Using dask to compute these if lazy
        valid_count = float(da.count().compute()) if hasattr(da.data, 'compute') else float(da.count())
        missing_count = total_pixels - valid_count
        missing_pct = (missing_count / total_pixels) * 100 if total_pixels > 0 else 0
        
        vmin = float(da.min().compute()) if hasattr(da.data, 'compute') else float(da.min())
        vmax = float(da.max().compute()) if hasattr(da.data, 'compute') else float(da.max())
        vmean = float(da.mean().compute()) if hasattr(da.data, 'compute') else float(da.mean())
        
        report["variables"][var_name] = {
            "valid_pixels": valid_count,
            "missing_pixels": missing_count,
            "missing_percentage": round(missing_pct, 2),
            "min_value": vmin,
            "max_value": vmax,
            "mean_value": vmean
        }
        
    report_file = os.path.join(output_dir, f"{dataset_name}_qc_report.json")
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=4)
        
    logger.info(f"QC report successfully written to {report_file}")
