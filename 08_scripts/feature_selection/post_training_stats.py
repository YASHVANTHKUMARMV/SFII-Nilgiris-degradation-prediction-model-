import pandas as pd
import logging
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.feature_selection import RFE
from sklearn.inspection import permutation_importance
import shap

logger = logging.getLogger("FeatureSelection.PostTraining")

def compute_rfe(df_X: pd.DataFrame, y: pd.Series, estimator, n_features_to_select: int = 4) -> pd.DataFrame:
    """Executes Recursive Feature Elimination (RFE)."""
    logger.info(f"Running RFE with {estimator.__class__.__name__}...")
    rfe = RFE(estimator=estimator, n_features_to_select=n_features_to_select, step=1)
    rfe.fit(df_X, y)
    
    results = pd.DataFrame({
        'Feature': df_X.columns,
        'Selected': rfe.support_,
        'Rank': rfe.ranking_
    }).sort_values(by='Rank')
    return results

def compute_permutation_importance(df_X: pd.DataFrame, y: pd.Series, estimator) -> pd.DataFrame:
    """Computes Permutation Importance to test absolute model reliance."""
    logger.info(f"Running Permutation Importance with {estimator.__class__.__name__}...")
    estimator.fit(df_X, y)
    result = permutation_importance(estimator, df_X, y, n_repeats=10, random_state=42, n_jobs=-1)
    
    importance = pd.DataFrame({
        'Feature': df_X.columns,
        'Importance_Mean': result.importances_mean,
        'Importance_Std': result.importances_std
    }).sort_values(by='Importance_Mean', ascending=False)
    return importance

def compute_shap_values(df_X: pd.DataFrame, y: pd.Series, estimator_type: str = 'xgboost'):
    """
    Computes SHAP values to determine exact marginal contributions.
    estimator_type: 'rf' or 'xgboost'
    """
    logger.info(f"Computing SHAP values for {estimator_type}...")
    
    if estimator_type == 'xgboost':
        model = XGBRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    else:
        model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
        
    model.fit(df_X, y)
    
    # SHAP Explainer
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(df_X)
    
    return explainer, shap_values
