import os
import logging
import numpy as np
import pandas as pd
from typing import Tuple, Optional

logger = logging.getLogger("ML_Pipeline.DataLoader")

class SFIIDataLoader:
    def __init__(self, data_dir: str):
        self.data_dir = data_dir
        self.features = ['EVI2', 'NBR', 'TCW', 'SAR_VV', 'SAR_VH', 'DEM_Elevation', 'DEM_Slope', 'Climate_Precip', 'Anthro_RoadDist', 'LandTrendr_Dur', 'SRT', 'SBP', 'DMF', 'ERS', 'FRP']
        
    def _generate_synthetic_dataset(self, num_pixels: int = 50000, num_years: int = 7) -> pd.DataFrame:
        """Generates a highly realistic synthetic dataset if real features are missing."""
        logger.info("Generating synthetic spatial-temporal dataset for fail-safe execution...")
        
        years = np.arange(2018, 2018 + num_years)
        
        # Spatial metadata for stratification
        pixel_ids = np.arange(num_pixels)
        elevations = np.random.uniform(500, 2500, size=num_pixels) # 500m to 2500m
        forest_types = np.random.choice(['Evergreen', 'Deciduous', 'Mixed'], size=num_pixels)
        
        # Expand over time
        df_list = []
        for year in years:
            # Base features
            df_year = pd.DataFrame({
                'pixel_id': pixel_ids,
                'year': year,
                'elevation': elevations,
                'forest_type': forest_types,
                'is_disturbed': np.random.choice([0, 1], size=num_pixels, p=[0.7, 0.3]),
            })
            
            # Simulated structural features based on disturbance state
            dist_mask = df_year['is_disturbed'] == 1
            
            df_year['h_norm'] = np.where(dist_mask, np.random.uniform(0.1, 0.5, num_pixels), np.random.uniform(0.6, 1.0, num_pixels))
            df_year['sigma0_norm'] = np.where(dist_mask, np.random.uniform(0.2, 0.6, num_pixels), np.random.uniform(0.7, 1.0, num_pixels))
            df_year['tcw_norm'] = np.where(dist_mask, np.random.uniform(0.0, 0.4, num_pixels), np.random.uniform(0.5, 0.9, num_pixels))
            df_year['srt'] = np.random.uniform(0, 1.5, num_pixels)
            df_year['dmf'] = np.where(dist_mask, np.random.uniform(0.5, 1.0, num_pixels), np.random.uniform(0.0, 0.2, num_pixels))
            df_year['ers'] = np.random.uniform(0, 1, num_pixels)
            df_year['frp'] = np.maximum(0, df_year['srt'] - (df_year['h_norm']*0.5 + df_year['sigma0_norm']*0.5))
            
            # Target (SFII) - bounded [0, 1]
            sfii_base = (1.0 - df_year['h_norm']) * 0.4 + df_year['dmf'] * 0.3 + df_year['frp'] * 0.3
            df_year['sfii'] = np.clip(sfii_base + np.random.normal(0, 0.05, num_pixels), 0.0, 1.0)
            
            # Create classification target (Degraded vs Intact)
            df_year['target_class'] = (df_year['sfii'] > 0.4).astype(int)
            
            df_list.append(df_year)
            
        return pd.concat(df_list, ignore_index=True)

    def load_dataset(self, validation_mode: bool = False, sample_fraction: float = 0.1, allow_synthetic_for_debugging: bool = False) -> pd.DataFrame:
        """
        Loads the dataset. Uses Stratified Sampling if in architectural validation mode.
        """
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        data_path = os.path.join(base_dir, "04_sfii_outputs", "computed", "sfii_computed_dataset.parquet")
        
        if os.path.exists(data_path):
            logger.info(f"Loading actual dataset from {data_path}")
            df = pd.read_parquet(data_path)
            # Create classification target for testing since the dataset target is continuous SFII
            if 'target_class' not in df.columns:
                df['target_class'] = (df['SFII'] > 0.4).astype(int)
            if 'is_disturbed' not in df.columns:
                df['is_disturbed'] = (df['LandTrendr_Dur'] > 0).astype(int)
            if 'forest_type' not in df.columns:
                df['forest_type'] = 'Mixed'
        else:
            if allow_synthetic_for_debugging and validation_mode:
                logger.warning(f"Dataset not found at {data_path}. Generating Synthetic data ONLY for debugging/verifying software execution.")
                df = self._generate_synthetic_dataset()
            else:
                logger.error(f"Real SFII feature dataset not found at {data_path}.")
                raise FileNotFoundError(
                    f"Real dataset missing at {data_path}. "
                    "Automatic synthetic data generation is strictly prohibited for training, model comparison, "
                    "performance evaluation, SHAP analysis, and publication figures. "
                    "Please run the feature engineering preprocessing pipeline to generate the real dataset."
                )
            
        if validation_mode:
            logger.warning(f"ARCHITECTURAL VALIDATION MODE: Subsetting dataset ({sample_fraction*100}%).")
            logger.info("Applying Stratified Spatial-Temporal Sampling...")
            
            # Stratify by Year, Disturbance Status, and Forest Type
            df = self._stratified_sample(df, sample_fraction)
        else:
            logger.info("FINAL SCIENTIFIC EXPERIMENT MODE: Using the COMPLETE dataset.")
            
        return df

    def _stratified_sample(self, df: pd.DataFrame, frac: float) -> pd.DataFrame:
        """
        Performs stratified sampling preserving class balance, years, and forest types.
        Elevation is preserved implicitly if randomly sampled within the strata.
        """
        # Create a stratification column
        df['strata'] = df['year'].astype(str) + '_' + df['is_disturbed'].astype(str) + '_' + df['forest_type']
        
        sampled = df.groupby('strata', group_keys=False).apply(lambda x: x.sample(frac=frac, random_state=42))
        
        # Drop temporary column
        sampled = sampled.drop(columns=['strata'])
        logger.info(f"Subsampled down to {len(sampled)} rows while preserving spatial-temporal distributions.")
        
        return sampled
