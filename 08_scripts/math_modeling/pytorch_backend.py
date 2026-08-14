import torch
import logging
from typing import Optional
from .utils import safe_divide, enforce_bounds

logger = logging.getLogger("SFII_Math.PyTorchBackend")

def compute_srt(vi_current: torch.Tensor, vi_pre: torch.Tensor, vi_ref: torch.Tensor) -> torch.Tensor:
    """Spectral Recovery Trajectory (SRT)."""
    numerator = vi_current - vi_pre
    denominator = vi_ref - vi_pre
    srt = safe_divide(numerator, denominator, eps=1e-6)
    return enforce_bounds(srt, min_val=0.0, max_val=2.0)

def compute_sbp(h_norm: torch.Tensor, sigma0_norm: torch.Tensor, tcw_norm: torch.Tensor, 
                alpha: float = 0.5, beta: float = 0.3, gamma: float = 0.2) -> torch.Tensor:
    """Structural Biomass Proxy (SBP)."""
    sbp = (alpha * h_norm) + (beta * sigma0_norm) + (gamma * tcw_norm)
    return enforce_bounds(sbp, min_val=0.0, max_val=1.0)

def compute_dmf(time_tensor: torch.Tensor, disturbance_times: torch.Tensor, 
                disturbance_mags: torch.Tensor, lam: float = 0.08) -> torch.Tensor:
    """Disturbance Memory Function (DMF)."""
    delta_t = time_tensor - disturbance_times
    mask = (delta_t >= 0).float()
    decay = torch.exp(-lam * delta_t) * mask
    dmf = torch.sum(disturbance_mags * decay, dim=-1, keepdim=True)
    return dmf

def compute_ers(f_dist: torch.Tensor, mu_ref: torch.Tensor, cov_ref: torch.Tensor, ridge_lambda: float = 1e-4) -> torch.Tensor:
    """Ecosystem Resilience Score (ERS). Using Bhattacharyya distance with Ridge Regularization."""
    feat_dim = cov_ref.size(-1)
    identity = torch.eye(feat_dim, device=cov_ref.device, dtype=cov_ref.dtype)
    cov_reg = cov_ref + ridge_lambda * identity
    
    try:
        cov_inv = torch.linalg.inv(cov_reg)
    except torch.linalg.LinAlgError:
        logger.warning("Falling back to SVD-based pseudo-inverse.")
        cov_inv = torch.linalg.pinv(cov_ref)
        
    diff = f_dist - mu_ref
    left = torch.einsum('...f,fg->...g', diff, cov_inv)
    d_b = 0.125 * torch.sum(left * diff, dim=-1, keepdim=True)
    
    ers = 1.0 - torch.exp(-d_b)
    return enforce_bounds(ers, 0.0, 1.0)

def compute_frp(srt: torch.Tensor, sbp: torch.Tensor) -> torch.Tensor:
    """False Recovery Penalty (FRP)."""
    diff = srt - sbp
    frp = torch.nn.functional.relu(diff)
    return frp

def compute_sfii(sbp: torch.Tensor, dmf: torch.Tensor, ers: torch.Tensor, frp: torch.Tensor,
                 dmf_max: float = 1.0, w1: float = 0.30, w2: float = 0.25, 
                 w3: float = 0.25, w4: float = 0.20) -> torch.Tensor:
    """Structural Forest Integrity Index (SFII)."""
    dmf_norm = dmf / dmf_max
    term1 = w1 * (1.0 - sbp)
    term2 = w2 * dmf_norm
    term3 = w3 * ers
    term4 = w4 * frp
    
    sfii = term1 + term2 + term3 + term4
    return enforce_bounds(sfii, 0.0, 1.0)
