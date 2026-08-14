"""
Phase 13: Validation Module
"""
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import logging

logger = logging.getLogger("Lab.Validation")

def compute_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Computes rigorous statistical metrics for continuous SFII prediction.
    """
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    # ---------------------------------------------------------
    # INTERNAL LABORATORY REVIEW: UNCERTAINTY QUANTIFICATION
    # ---------------------------------------------------------
    # The laboratory mandates that standard point estimates (RMSE) are insufficient 
    # for remote sensing prediction due to spatial autocorrelation.
    # Decision: We inject Bootstrapping (n=1000) to calculate 95% Confidence Intervals 
    # for R², allowing us to report robust uncertainty bounds in the final paper.
    # ---------------------------------------------------------
    
    # Bootstrap R2
    n_iterations = 1000
    r2_bootstraps = []
    for _ in range(n_iterations):
        indices = np.random.choice(len(y_true), len(y_true), replace=True)
        r2_bootstraps.append(r2_score(y_true[indices], y_pred[indices]))
        
    ci_lower = np.percentile(r2_bootstraps, 2.5)
    ci_upper = np.percentile(r2_bootstraps, 97.5)
    
    metrics = {
        'RMSE': rmse,
        'MAE': mae,
        'R2': r2,
        'R2_95CI': (ci_lower, ci_upper)
    }
    
    return metrics
