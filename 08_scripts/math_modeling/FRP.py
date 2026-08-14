import torch
import logging
import matplotlib.pyplot as plt
from .utils import check_tensor_format

logger = logging.getLogger("SFII_Math.FRP")

def compute(srt: torch.Tensor, sbp: torch.Tensor) -> torch.Tensor:
    """
    Computes the False Recovery Penalty (FRP).
    Equation: FRP = max(0, SRT - SBP)
    
    Args:
        srt: Spectral Recovery Trajectory
        sbp: Structural Biomass Proxy
        
    Returns:
        frp: Penalty applied when spectral recovery outpaces structural recovery.
    """
    check_tensor_format(srt)
    logger.info("Computing False Recovery Penalty (FRP)...")
    
    # Calculate difference
    diff = srt - sbp
    
    # ReLU equivalent: max(0, diff)
    frp = torch.nn.functional.relu(diff)
    
    return frp

def validate(frp: torch.Tensor) -> bool:
    if torch.isnan(frp).any() or torch.isinf(frp).any():
        logger.error("FRP validation failed: NaN or Inf present.")
        return False
    if (frp < 0.0).any():
        logger.error("FRP validation failed: Negative values present (violates max(0,x)).")
        return False
    return True

def plot(frp: torch.Tensor, time_steps: list, title="False Recovery Penalty (FRP)"):
    if frp.dim() == 5:
        y = frp[0, :, 0, 0, 0].cpu().numpy()
    else:
        y = frp[0, :, 0].cpu().numpy()
        
    plt.figure(figsize=(10, 4))
    plt.plot(time_steps, y, marker='d', color='red', label='FRP')
    plt.title(title)
    plt.xlabel("Time")
    plt.ylabel("Penalty")
    plt.legend()
    plt.show()

def save(tensor: torch.Tensor, path: str):
    torch.save(tensor, path)

def load(path: str) -> torch.Tensor:
    return torch.load(path)
