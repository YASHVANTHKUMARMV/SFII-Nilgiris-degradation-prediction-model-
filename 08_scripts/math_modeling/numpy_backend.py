import numpy as np
import logging
from typing import Optional

logger = logging.getLogger("SFII_Math.NumPyBackend")

def safe_divide(numerator: np.ndarray, denominator: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Safe division avoiding NaN and Inf."""
    sign = np.sign(denominator)
    sign = np.where(sign == 0, 1.0, sign)
    stable_denom = denominator + eps * sign
    
    result = numerator / stable_denom
    return np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)

def enforce_bounds(arr: np.ndarray, min_val: float = 0.0, max_val: float = 1.0) -> np.ndarray:
    """Clips array to boundaries."""
    return np.clip(arr, min_val, max_val)

def compute_srt(vi_current: np.ndarray, vi_pre: np.ndarray, vi_ref: np.ndarray) -> np.ndarray:
    """Spectral Recovery Trajectory (SRT)."""
    numerator = vi_current - vi_pre
    denominator = vi_ref - vi_pre
    srt = safe_divide(numerator, denominator)
    return enforce_bounds(srt, 0.0, 2.0)

def compute_sbp(h_norm: np.ndarray, sigma0_norm: np.ndarray, tcw_norm: np.ndarray, 
                alpha: float = 0.5, beta: float = 0.3, gamma: float = 0.2) -> np.ndarray:
    """Structural Biomass Proxy (SBP)."""
    sbp = (alpha * h_norm) + (beta * sigma0_norm) + (gamma * tcw_norm)
    return enforce_bounds(sbp, 0.0, 1.0)

def compute_dmf(time_arr: np.ndarray, disturbance_times: np.ndarray, 
                disturbance_mags: np.ndarray, lam: float = 0.08) -> np.ndarray:
    """Disturbance Memory Function (DMF)."""
    delta_t = time_arr - disturbance_times
    mask = (delta_t >= 0).astype(float)
    decay = np.exp(-lam * delta_t) * mask
    dmf = np.sum(disturbance_mags * decay, axis=-1, keepdims=True)
    return dmf

def compute_ers(f_dist: np.ndarray, mu_ref: np.ndarray, cov_ref: np.ndarray, ridge_lambda: float = 1e-4) -> np.ndarray:
    """Ecosystem Resilience Score (ERS). Using Bhattacharyya distance with Ridge Regularization."""
    feat_dim = cov_ref.shape[-1]
    identity = np.eye(feat_dim)
    cov_reg = cov_ref + ridge_lambda * identity
    
    try:
        cov_inv = np.linalg.inv(cov_reg)
    except np.linalg.LinAlgError:
        logger.warning("Falling back to pseudo-inverse for ERS.")
        cov_inv = np.linalg.pinv(cov_ref)
        
    diff = f_dist - mu_ref
    
    # diff: [..., F], cov_inv: [F, F]
    left = np.einsum('...f,fg->...g', diff, cov_inv)
    d_b = 0.125 * np.sum(left * diff, axis=-1, keepdims=True)
    
    ers = 1.0 - np.exp(-d_b)
    return enforce_bounds(ers, 0.0, 1.0)

def compute_frp(srt: np.ndarray, sbp: np.ndarray) -> np.ndarray:
    """False Recovery Penalty (FRP)."""
    diff = srt - sbp
    frp = np.maximum(0, diff)
    return frp

def compute_sfii(sbp: np.ndarray, dmf: np.ndarray, ers: np.ndarray, frp: np.ndarray,
                 dmf_max: float = 1.0, w1: float = 0.30, w2: float = 0.25, 
                 w3: float = 0.25, w4: float = 0.20) -> np.ndarray:
    """Structural Forest Integrity Index (SFII)."""
    dmf_norm = dmf / dmf_max
    
    term1 = w1 * (1.0 - sbp)
    term2 = w2 * dmf_norm
    term3 = w3 * ers
    term4 = w4 * frp
    
    sfii = term1 + term2 + term3 + term4
    return enforce_bounds(sfii, 0.0, 1.0)
