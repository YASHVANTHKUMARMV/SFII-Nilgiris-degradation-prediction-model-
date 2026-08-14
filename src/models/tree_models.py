import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
import logging

logger = logging.getLogger(__name__)

def build_xgboost(config: dict) -> xgb.XGBRegressor:
    """
    Builds the XGBoost model based on Hydra configuration.
    XGBoost is highly robust to non-linear ecological relationships but sensitive to collinearity.
    """
    logger.info("Initializing XGBoost Regressor...")
    model = xgb.XGBRegressor(
        n_estimators=config['n_estimators'],
        learning_rate=config['learning_rate'],
        max_depth=config['max_depth'],
        subsample=config['subsample'],
        objective='reg:squarederror',
        n_jobs=-1,
        random_state=42
    )
    return model

def build_random_forest(config: dict) -> RandomForestRegressor:
    """
    Builds the Random Forest model based on Hydra configuration.
    Random Forest is highly robust to multicollinearity due to random feature subsetting at nodes.
    """
    logger.info("Initializing Random Forest Regressor...")
    model = RandomForestRegressor(
        n_estimators=config['n_estimators'],
        max_depth=config['max_depth'],
        n_jobs=-1,
        random_state=42
    )
    return model

# -----------------------------------------------------------------------------
# INTERNAL LABORATORY REVIEW: TREE MODEL SELECTION
# -----------------------------------------------------------------------------
# Decision: While Random Forest is robust to collinearity, XGBoost consistently 
# outperforms RF in predicting continuous geospatial variables when the feature 
# space is properly validated and pruned (which we did in Phase 5 Feature Selection).
# XGBoost's gradient descent approach handles the high-dimensional tabular data 
# exported from the Parquet files more efficiently.
# 
# Therefore, for the final baseline tabular model, we will prioritize XGBoost.
# -----------------------------------------------------------------------------
