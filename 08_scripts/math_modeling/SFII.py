import torch
import logging
import matplotlib.pyplot as plt
from .utils import check_tensor_format, enforce_bounds

logger = logging.getLogger("SFII_Math.SFII")

def compute(sbp: torch.Tensor, dmf: torch.Tensor, ers: torch.Tensor, frp: torch.Tensor,
            dmf_max: float = 1.0, w1=0.30, w2=0.25, w3=0.25, w4=0.20) -> torch.Tensor:
    """
    Computes the Structural Forest Integrity Index (SFII).
    Equation: SFII = w1*(1-SBP) + w2*(DMF/DMF_max) + w3*ERS + w4*FRP
    
    Args:
        sbp: Structural Biomass Proxy
        dmf: Disturbance Memory Function
        ers: Ecosystem Resilience Score
        frp: False Recovery Penalty
        dmf_max: Maximum possible DMF score for normalization
        w1, w2, w3, w4: Weights summing to 1.0
        
    Returns:
        sfii: Bounded degradation index [0, 1]
    """
    check_tensor_format(sbp)
    logger.info("Computing final Structural Forest Integrity Index (SFII)...")
    
    # Normalize DMF
    dmf_norm = dmf / dmf_max
    
    # Compute weighted sum
    term1 = w1 * (1.0 - sbp)
    term2 = w2 * dmf_norm
    term3 = w3 * ers
    term4 = w4 * frp
    
    sfii = term1 + term2 + term3 + term4
    
    # Enforce boundaries due to potential float drift or extreme FRP
    sfii_bounded = enforce_bounds(sfii, 0.0, 1.0)
    
    return sfii_bounded

def validate(sfii: torch.Tensor) -> bool:
    if torch.isnan(sfii).any() or torch.isinf(sfii).any():
        logger.error("SFII validation failed: NaN or Inf present.")
        return False
    if (sfii < 0.0).any() or (sfii > 1.0).any():
        logger.error("SFII validation failed: Values outside [0, 1].")
        return False
    return True

def plot(sfii: torch.Tensor, time_steps: list, title="Structural Forest Integrity Index"):
    if sfii.dim() == 5:
        y = sfii[0, :, 0, 0, 0].cpu().numpy()
    else:
        y = sfii[0, :, 0].cpu().numpy()
        
    plt.figure(figsize=(10, 4))
    plt.plot(time_steps, y, marker='o', color='black', linewidth=2, label='SFII')
    plt.axhline(0.2, color='green', linestyle=':', label='Intact')
    plt.axhline(0.8, color='red', linestyle=':', label='Severe Degradation')
    plt.title(title)
    plt.xlabel("Time")
    plt.ylabel("Degradation Score")
    plt.ylim(0, 1.1)
    plt.legend()
    plt.show()

def save(tensor: torch.Tensor, path: str):
    torch.save(tensor, path)

def load(path: str) -> torch.Tensor:
    return torch.load(path)
