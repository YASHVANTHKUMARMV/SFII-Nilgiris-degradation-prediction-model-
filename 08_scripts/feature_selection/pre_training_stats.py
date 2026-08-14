import pandas as pd
import numpy as np
from scipy.stats import spearmanr, kendalltau
from sklearn.feature_selection import mutual_info_regression
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.decomposition import PCA
from scipy.cluster import hierarchy
import logging

logger = logging.getLogger("FeatureSelection.PreTraining")

def compute_correlations(df: pd.DataFrame) -> dict:
    """Computes Pearson, Spearman, and Kendall correlations."""
    logger.info("Computing correlation matrices...")
    correlations = {
        'pearson': df.corr(method='pearson'),
        'spearman': df.corr(method='spearman'),
        'kendall': df.corr(method='kendall')
    }
    return correlations

def compute_vif(df: pd.DataFrame) -> pd.DataFrame:
    """Computes Variance Inflation Factor (VIF) to measure multicollinearity."""
    logger.info("Computing VIF...")
    vif_data = pd.DataFrame()
    vif_data["feature"] = df.columns
    # Add constant for VIF calculation to be accurate
    df_with_const = df.copy()
    df_with_const['const'] = 1.0
    vif_data["VIF"] = [variance_inflation_factor(df_with_const.values, i) 
                       for i in range(len(df.columns))]
    return vif_data.sort_values(by="VIF", ascending=False)

def compute_mutual_information(df_X: pd.DataFrame, y: pd.Series) -> pd.Series:
    """Computes non-linear Mutual Information between features and a target proxy."""
    logger.info("Computing Mutual Information...")
    mi_scores = mutual_info_regression(df_X, y, random_state=42)
    return pd.Series(mi_scores, index=df_X.columns).sort_values(ascending=False)

def compute_pca_variance(df: pd.DataFrame) -> np.ndarray:
    """Evaluates intrinsic dimensionality of the dataset using PCA."""
    logger.info("Computing PCA explained variance...")
    pca = PCA()
    pca.fit(df)
    return pca.explained_variance_ratio_

def compute_hierarchical_clustering(df: pd.DataFrame) -> dict:
    """Groups features based on Spearman correlation distances."""
    logger.info("Computing Hierarchical Clustering based on Spearman distance...")
    corr = df.corr(method='spearman')
    # Convert correlation to distance matrix [0, 2]
    distance = 1 - corr
    linkage = hierarchy.ward(distance)
    return {'distance': distance, 'linkage': linkage}
