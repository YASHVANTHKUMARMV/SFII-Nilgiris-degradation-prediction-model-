import torch
import logging
import matplotlib.pyplot as plt
from .utils import check_tensor_format, safe_divide

logger = logging.getLogger("SFII_Math.DMF")

def compute(time_tensor: torch.Tensor, disturbance_times: torch.Tensor, 
            disturbance_mags: torch.Tensor, lam: float = 0.08) -> torch.Tensor:
    """
    Computes the Disturbance Memory Function (DMF).
    Equation: DMF = sum_i(m_i * exp(-lambda * (t - t_i))) for t_i <= t
    
    Args:
        time_tensor: Current time `t`
        disturbance_times: `t_i` events
        disturbance_mags: `m_i` magnitudes
        lam: Decay constant lambda
        
    Returns:
        dmf: Unnormalized DMF score
    """
    logger.info(f"Computing DMF (lambda={lam})...")
    
    # Calculate time delta (t - t_i)
    # Ensure we only consider past disturbances (t >= t_i)
    delta_t = time_tensor - disturbance_times
    mask = (delta_t >= 0).float()
    
    decay = torch.exp(-lam * delta_t) * mask
    
    # Sum over disturbance events (assuming event dimension is last)
    dmf = torch.sum(disturbance_mags * decay, dim=-1, keepdim=True)
    
    return dmf

def validate(dmf: torch.Tensor) -> bool:
    if torch.isnan(dmf).any():
        logger.error("DMF contains NaN.")
        return False
    if (dmf < 0.0).any():
        logger.error("DMF contains negative values (impossible for decay sum).")
        return False
    return True

def plot(dmf: torch.Tensor, time_steps: list, title="Disturbance Memory Function (DMF)"):
    if dmf.dim() == 5:
        y = dmf[0, :, 0, 0, 0].cpu().numpy()
    else:
        y = dmf[0, :, 0].cpu().numpy()
        
    plt.figure(figsize=(10, 4))
    plt.plot(time_steps, y, marker='^', color='orange', label='DMF')
    plt.title(title)
    plt.xlabel("Time")
    plt.ylabel("DMF Score")
    plt.legend()
    plt.show()

def save(tensor: torch.Tensor, path: str):
    torch.save(tensor, path)

def load(path: str) -> torch.Tensor:
    return torch.load(path)
