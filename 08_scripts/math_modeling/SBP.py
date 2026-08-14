import torch
import logging
import matplotlib.pyplot as plt
from .utils import enforce_bounds, check_tensor_format

logger = logging.getLogger("SFII_Math.SBP")

def compute(h_norm: torch.Tensor, sigma0_norm: torch.Tensor, tcw_norm: torch.Tensor, 
            alpha=0.5, beta=0.3, gamma=0.2) -> torch.Tensor:
    """
    Computes the Structural Biomass Proxy (SBP).
    Equation: SBP = alpha*H + beta*Sigma0 + gamma*TCW
    
    Args:
        h_norm: Normalized GEDI canopy height
        sigma0_norm: Normalized SAR backscatter
        tcw_norm: Normalized Tasseled Cap Wetness
        alpha, beta, gamma: Weight coefficients
        
    Returns:
        sbp: Bounded proxy [0, 1]
    """
    check_tensor_format(h_norm)
    logger.info("Computing SBP...")
    
    sbp = (alpha * h_norm) + (beta * sigma0_norm) + (gamma * tcw_norm)
    
    # Clip to [0, 1] to prevent floating point overshoot
    sbp_bounded = enforce_bounds(sbp, min_val=0.0, max_val=1.0)
    
    return sbp_bounded

def validate(sbp: torch.Tensor) -> bool:
    if torch.isnan(sbp).any() or torch.isinf(sbp).any():
        logger.error("SBP validation failed: NaN or Inf present.")
        return False
    if (sbp < 0.0).any() or (sbp > 1.0).any():
        logger.error("SBP validation failed: Values outside [0, 1].")
        return False
    return True

def plot(sbp: torch.Tensor, time_steps: list, title="Structural Biomass Proxy"):
    if sbp.dim() == 5:
        y = sbp[0, :, 0, 0, 0].cpu().numpy()
    else:
        y = sbp[0, :, 0].cpu().numpy()
        
    plt.figure(figsize=(10, 4))
    plt.plot(time_steps, y, marker='s', color='g', label='SBP')
    plt.title(title)
    plt.xlabel("Time")
    plt.ylabel("SBP Score")
    plt.ylim(0, 1.1)
    plt.legend()
    plt.show()

def save(tensor: torch.Tensor, path: str):
    torch.save(tensor, path)

def load(path: str) -> torch.Tensor:
    return torch.load(path)
