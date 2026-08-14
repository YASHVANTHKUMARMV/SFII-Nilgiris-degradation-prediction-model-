import pytest
import numpy as np
import torch
from engine import SFIIEngine
from validator import SFIIMathValidator

@pytest.fixture
def dummy_inputs():
    """Generates synthetic data for testing."""
    B, T, F = 2, 5, 3
    np.random.seed(42)
    
    inputs = {
        'vi_current': np.random.rand(B, T, 1),
        'vi_pre': np.random.rand(B, T, 1),
        'vi_ref': np.random.rand(B, T, 1) + 1.0, # ensure denominator > 0
        'h_norm': np.random.rand(B, T, 1),
        'sigma0_norm': np.random.rand(B, T, 1),
        'tcw_norm': np.random.rand(B, T, 1),
        'time_arr': np.arange(T).reshape(1, T, 1).repeat(B, axis=0),
        'disturbance_times': np.array([1.0]).reshape(1, 1, 1).repeat(B, axis=0),
        'disturbance_mags': np.array([0.5]).reshape(1, 1, 1).repeat(B, axis=0),
        'f_dist': np.random.rand(B, T, F),
        'mu_ref': np.zeros((F,)),
        'cov_ref': np.eye(F),
        'dmf_max': 1.0
    }
    return inputs

def test_validator_detects_nans():
    inputs = {'h_norm': np.array([np.nan, 0.5])}
    with pytest.raises(ValueError, match="NaN detected"):
        SFIIMathValidator.validate_inputs(inputs)

def test_backend_equivalence(dummy_inputs):
    """Tests if NumPy and PyTorch backends produce identical results."""
    # Convert numpy inputs to torch tensors for PyTorch backend
    torch_inputs = {k: torch.tensor(v, dtype=torch.float32) if isinstance(v, np.ndarray) else v 
                   for k, v in dummy_inputs.items()}
    
    engine_np = SFIIEngine(backend='numpy')
    engine_pt = SFIIEngine(backend='pytorch')
    
    out_np = engine_np.compute_all(dummy_inputs)
    out_pt = engine_pt.compute_all(torch_inputs)
    
    for key in out_np.keys():
        np_val = out_np[key]
        pt_val = out_pt[key].cpu().numpy()
        
        # Assert almost equal (float tolerance)
        np.testing.assert_allclose(np_val, pt_val, rtol=1e-5, atol=1e-5, err_msg=f"Mismatch in {key}")

def test_frp_penalty():
    """Verify FRP is strictly max(0, SRT - SBP)"""
    engine = SFIIEngine(backend='numpy')
    srt = np.array([1.5, 0.5])
    sbp = np.array([1.0, 0.8])
    
    frp = engine.backend_module.compute_frp(srt, sbp)
    
    assert frp[0] == 0.5  # 1.5 - 1.0
    assert frp[1] == 0.0  # max(0, 0.5 - 0.8)
