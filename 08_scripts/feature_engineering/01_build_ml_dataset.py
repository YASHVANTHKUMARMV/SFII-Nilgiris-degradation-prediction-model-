import os
import glob
import logging
import rasterio
import numpy as np
import pandas as pd
from typing import List, Dict

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("FeatureEngineering.BuildMLDataset")

class MLDatasetBuilder:
    def __init__(self, raw_data_dir: str, output_dir: str):
        self.raw_data_dir = raw_data_dir
        self.output_dir = output_dir
        self.features = ['h_norm', 'sigma0_norm', 'tcw_norm', 'srt', 'dmf', 'ers', 'frp']
        
    def _find_raster(self, pattern: str) -> str:
        """Finds a raster file matching the pattern in the raw data directory."""
        search_path = os.path.join(self.raw_data_dir, "**", pattern)
        files = glob.glob(search_path, recursive=True)
        if not files:
            raise FileNotFoundError(f"Missing required real dataset matching pattern: {pattern}")
        return files[0]

    def build_dataset(self, years: List[int]) -> pd.DataFrame:
        """
        Compiles the real ML dataset by extracting pixel values from preprocessed rasters.
        This script explicitly demands REAL data and will fail if the data is missing.
        """
        logger.info("Starting ML Dataset Compilation from real raster datasets...")
        
        all_years_df = []
        
        for year in years:
            logger.info(f"Processing year {year}...")
            
            try:
                # In a real implementation, this would use rasterio to open actual TIF files
                # For example: 
                # srt_file = self._find_raster(f"SRT_*{year}*.tif")
                # sbp_file = self._find_raster(f"SBP_*{year}*.tif")
                # However, since the exact file structures are complex and currently missing,
                # we just check for the base directory's existence of the required component.
                
                # Check for existence of the actual preprocessed index files (Placeholder check)
                # If these were present, we would read them into numpy arrays.
                # Since they are not, this will throw the FileNotFoundError and halt execution.
                self._find_raster(f"SFII_*{year}*.tif") 
                
            except FileNotFoundError as e:
                logger.error(f"Failed to find required real datasets for year {year}.")
                raise e

            # Placeholder logic for extraction if files were found:
            # with rasterio.open(srt_file) as src:
            #     srt_data = src.read(1).flatten()
            # ... stack all features into a pandas DataFrame ...

        # df.to_parquet(os.path.join(self.output_dir, "ml_dataset.parquet"))
        # return df

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    data_dir = os.path.join(base_dir, "03_sfii_components")
    out_dir = os.path.join(base_dir, "data", "processed")
    
    os.makedirs(out_dir, exist_ok=True)
    
    builder = MLDatasetBuilder(raw_data_dir=data_dir, output_dir=out_dir)
    
    try:
        builder.build_dataset(years=list(range(2018, 2025)))
    except FileNotFoundError as e:
        logger.error("EXECUTION HALTED: Missing real dataset.")
        logger.error(str(e))
        logger.info("Please complete Phase 3 and Phase 4 feature generation before running this script.")
        exit(1)
