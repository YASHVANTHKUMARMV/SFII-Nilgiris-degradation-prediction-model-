import torch
import logging
import matplotlib.pyplot as plt
from .utils import safe_divide, enforce_bounds, check_tensor_format

logger = logging.getLogger("SFII_Math.SRT")

def compute(vi_current: torch.Tensor, vi_pre: torch.Tensor, vi_ref: torch.Tensor) -> torch.Tensor:
    """
    Computes the Spectral Recovery Trajectory (SRT).
    Equation: SRT = (VI_current - VI_pre) / (VI_ref - VI_pre)
    
    Args:
        vi_current: Tensor of shape [B, T, H, W, 1] or [B, T, 1]
        vi_pre: Pre-disturbance baseline (broadcastable shape)
        vi_ref: Reference pristine forest (broadcastable shape)
    
    Returns:
        srt: Bounded spectral recovery trajectory [0, 2]
    """
    check_tensor_format(vi_current)
    logger.info("Computing SRT...")
    
    numerator = vi_current - vi_pre
    denominator = vi_ref - vi_pre
    
    srt = safe_divide(numerator, denominator, eps=1e-6)
    
    # Clip to [0, 2] as per definition (values > 1 indicate overshoot)
    srt_bounded = enforce_bounds(srt, min_val=0.0, max_val=2.0)
    
    return srt_bounded

def validate(srt: torch.Tensor) -> bool:
    """Validates tensor output bounds."""
    if torch.isnan(srt).any() or torch.isinf(srt).any():
        logger.error("SRT validation failed: NaN or Inf present.")
        return False
    if (srt < 0.0).any() or (srt > 2.0).any():
        logger.error("SRT validation failed: Values outside [0, 2].")
        return False
    return True

def plot(srt: torch.Tensor, time_steps: list, title="Spectral Recovery Trajectory"):
    """Visualizes the SRT over time for the first batch/pixel."""
    # Assuming 1D time-series shape for plotting: [T]
    if srt.dim() == 5:
        y = srt[0, :, 0, 0, 0].cpu().numpy()
    else:
        y = srt[0, :, 0].cpu().numpy()
        
    plt.figure(figsize=(10, 4))
    plt.plot(time_steps, y, marker='o', label='SRT')
    plt.axhline(1.0, color='r', linestyle='--', label='Full Spectral Recovery')
    plt.title(title)
    plt.xlabel("Time")
    plt.ylabel("SRT Score")
    plt.legend()
    plt.show()

def save(srt: torch.Tensor, path: str):
    torch.save(srt, path)
    logger.info(f"SRT saved to {path}")

def load(path: str) -> torch.Tensor:
    return torch.load(path)
