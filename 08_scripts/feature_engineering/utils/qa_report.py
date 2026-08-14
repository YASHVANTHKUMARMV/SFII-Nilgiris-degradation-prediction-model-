import os
import json
import logging
from datetime import datetime
import numpy as np
import xarray as xr

# 19. Logging Setup
def setup_logger(log_dir: str, name: str = "FeatureEng") -> logging.Logger:
    os.makedirs(log_dir, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    if not logger.handlers:
        fh = logging.FileHandler(os.path.join(log_dir, f"{name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"))
        ch = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        logger.addHandler(fh)
        logger.addHandler(ch)
        
    return logger

# 20. Quality Reports
def generate_quality_report(ds: xr.Dataset, output_dir: str, filename: str = "qa_report.json"):
    """Generates a JSON report on data health (NaN percentages, min/max)."""
    logger = logging.getLogger("FeatureEng.QA")
    logger.info("Generating Quality Assessment Report...")
    
    os.makedirs(output_dir, exist_ok=True)
    report = {
        "timestamp": datetime.now().isoformat(),
        "dimensions": dict(ds.sizes),
        "variables": {}
    }
    
    for var_name in ds.data_vars:
        da = ds[var_name]
        # Computation on a sample or reduced array to prevent OOM
        total_pixels = float(da.size)
        nan_count = float(da.isnull().sum().compute())
        nan_pct = (nan_count / total_pixels) * 100
        
        report["variables"][var_name] = {
            "nan_percentage": round(nan_pct, 4),
            "total_elements": total_pixels
        }
        
    out_path = os.path.join(output_dir, filename)
    with open(out_path, 'w') as f:
        json.dump(report, f, indent=4)
        
    logger.info(f"QA Report saved to {out_path}")
