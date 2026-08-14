import numpy as np
import pandas as pd
import logging
from sklearn.model_selection import GroupKFold, TimeSeriesSplit

logger = logging.getLogger("ML_Pipeline.CVSplitter")

class SpatialTemporalSplitter:
    """
    Implements advanced splitting strategies to prevent data leakage.
    - Spatial CV: Ensures pixels in Train are disjoint from Test.
    - Temporal CV: Trains on past years, predicts future years.
    """
    
    @staticmethod
    def get_spatial_split(df: pd.DataFrame, test_size: float = 0.2):
        """
        Splits data so that specific pixels (and all their temporal observations)
        are either entirely in train or entirely in test.
        """
        logger.info(f"Generating Spatial Holdout Split (Test Size = {test_size})...")
        unique_pixels = df['pixel_id'].unique()
        
        np.random.seed(42)
        np.random.shuffle(unique_pixels)
        
        split_idx = int(len(unique_pixels) * (1 - test_size))
        train_pixels = set(unique_pixels[:split_idx])
        
        train_mask = df['pixel_id'].isin(train_pixels)
        
        train_df = df[train_mask]
        test_df = df[~train_mask]
        
        logger.info(f"Spatial Split -> Train: {len(train_df)} rows, Test: {len(test_df)} rows.")
        return train_df, test_df
        
    @staticmethod
    def get_temporal_split(df: pd.DataFrame, holdout_year: int = 2024):
        """
        Splits data chronologically. Train on all years < holdout_year.
        Test on holdout_year.
        """
        logger.info(f"Generating Temporal Holdout Split (Holdout Year = {holdout_year})...")
        
        train_df = df[df['year'] < holdout_year]
        test_df = df[df['year'] == holdout_year]
        
        if len(test_df) == 0:
            logger.warning(f"No data found for holdout year {holdout_year}. Falling back to max year.")
            holdout_year = df['year'].max()
            train_df = df[df['year'] < holdout_year]
            test_df = df[df['year'] == holdout_year]
            
        logger.info(f"Temporal Split -> Train: {len(train_df)} rows, Test: {len(test_df)} rows.")
        return train_df, test_df

    @staticmethod
    def prepare_xy(df: pd.DataFrame, feature_cols: list, target_col: str = 'target_class'):
        """Extracts X and y arrays from DataFrame."""
        X = df[feature_cols].values
        y = df[target_col].values
        return X, y
