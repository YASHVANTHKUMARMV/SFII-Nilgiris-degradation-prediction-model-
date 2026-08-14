import torch
import logging
import matplotlib.pyplot as plt
from .utils import check_tensor_format, enforce_bounds

logger = logging.getLogger("SFII_Math.ERS")

def compute(f_dist: torch.Tensor, mu_ref: torch.Tensor, cov_ref: torch.Tensor, ridge_lambda: float = 1e-4) -> torch.Tensor:
    """
    Computes the Ecosystem Resilience Score (ERS) using Bhattacharyya distance with Ridge Regularization.
    
    Args:
        f_dist: Features of disturbed pixel [..., Features]
        mu_ref: Mean vector of reference forest [Features]
        cov_ref: Covariance matrix of reference forest [Features, Features]
        ridge_lambda: Regularization parameter for covariance inversion
        
    Returns:
        ers: Bounded resilience score [0, 1]
    """
    logger.info(f"Computing ERS with Ridge Regularization (lambda={ridge_lambda})...")
    
    # Ridge Regularization: Sigma_reg = Sigma + lambda*I
    feat_dim = cov_ref.size(-1)
    identity = torch.eye(feat_dim, device=cov_ref.device, dtype=cov_ref.dtype)
    cov_reg = cov_ref + ridge_lambda * identity
    
    # Compute inverse
    try:
        cov_inv = torch.linalg.inv(cov_reg)
    except torch.linalg.LinAlgError:
        logger.error("Covariance inversion failed despite ridge regularization.")
        # Fallback to pseudo-inverse for ablation/failsafe
        logger.warning("Falling back to SVD-based pseudo-inverse.")
        cov_inv = torch.linalg.pinv(cov_ref)
        
    # Simplified Bhattacharyya distance (assuming equal covariances for computation)
    diff = f_dist - mu_ref
    
    # Compute Mahalanobis term: diff^T * cov_inv * diff
    # Use torch.einsum for batched compatibility
    # diff: [..., F], cov_inv: [F, F]
    left = torch.einsum('...f,fg->...g', diff, cov_inv)
    d_b = 0.125 * torch.sum(left * diff, dim=-1, keepdim=True)
    
    # ERS = 1 - exp(-D_B)
    ers = 1.0 - torch.exp(-d_b)
    
    return enforce_bounds(ers, 0.0, 1.0)

def validate(ers: torch.Tensor) -> bool:
    if torch.isnan(ers).any() or torch.isinf(ers).any():
        logger.error("ERS validation failed: NaN or Inf present.")
        return False
    if (ers < 0.0).any() or (ers > 1.0).any():
        logger.error("ERS validation failed: Values outside [0, 1].")
        return False
    return True

def plot(ers: torch.Tensor, time_steps: list, title="Ecosystem Resilience Score (ERS)"):
    if ers.dim() == 5:
        y = ers[0, :, 0, 0, 0].cpu().numpy()
    else:
        y = ers[0, :, 0].cpu().numpy()
        
    plt.figure(figsize=(10, 4))
    plt.plot(time_steps, y, marker='x', color='purple', label='ERS')
    plt.title(title)
    plt.xlabel("Time")
    plt.ylabel("ERS Score")
    plt.ylim(0, 1.1)
    plt.legend()
    plt.show()

def save(tensor: torch.Tensor, path: str):
    torch.save(tensor, path)

def load(path: str) -> torch.Tensor:
    return torch.load(path)
