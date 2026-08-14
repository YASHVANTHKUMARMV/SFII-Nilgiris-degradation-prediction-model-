import torch
import logging

logger = logging.getLogger("SFII_Math.Utils")

def safe_divide(numerator: torch.Tensor, denominator: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Performs division with epsilon stabilization to prevent NaN/Inf.
    
    Args:
        numerator: Tensor of any shape.
        denominator: Tensor of matching shape.
        eps: Small value added to the denominator.
        
    Returns:
        Tensor resulting from numerator / (denominator + eps*sign(denominator))
    """
    # Ensure denominator never hits exactly zero
    # We use sign to preserve the direction of the denominator if it's negative
    sign = torch.sign(denominator)
    sign = torch.where(sign == 0, torch.ones_like(sign), sign)
    stable_denom = denominator + eps * sign
    
    result = numerator / stable_denom
    
    # NaN/Inf checks
    if torch.isnan(result).any() or torch.isinf(result).any():
        logger.warning("safe_divide resulted in NaN or Inf despite stabilization.")
        # Replace NaNs with 0 as a strict fallback
        result = torch.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)
        
    return result

def enforce_bounds(tensor: torch.Tensor, min_val: float = 0.0, max_val: float = 1.0) -> torch.Tensor:
    """
    Clips a tensor to the specified boundaries.
    """
    return torch.clamp(tensor, min=min_val, max=max_val)

def check_tensor_format(tensor: torch.Tensor):
    """
    Validates that the tensor follows either:
    [Batch, Time, Height, Width, Features] or [Batch, Time, Features]
    """
    dims = tensor.dim()
    if dims not in [3, 5]:
        logger.error(f"Invalid tensor shape {tensor.shape}. Expected 3 or 5 dimensions.")
        raise ValueError(f"Tensor must be [B, T, F] or [B, T, H, W, F]. Got {dims}D.")
