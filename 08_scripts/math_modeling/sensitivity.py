import numpy as np
import logging
import pandas as pd
from .engine import SFIIEngine

logger = logging.getLogger("SFII_Math.Sensitivity")

def run_sensitivity_analysis(base_inputs: dict, perturbations: list = [-0.1, 0.0, 0.1]) -> pd.DataFrame:
    """
    Runs a sensitivity analysis by perturbing structural features.
    
    Args:
        base_inputs: Base dictionary of inputs (NumPy format expected).
        perturbations: List of percentage perturbations (e.g., -0.1 means -10%).
        
    Returns:
        DataFrame containing the variance in SFII for each perturbation.
    """
    logger.info(f"Running sensitivity analysis with perturbations: {perturbations}")
    engine = SFIIEngine(backend='numpy')
    
    results = []
    
    # Base run
    base_output = engine.compute_all(base_inputs)
    base_sfii = float(np.mean(base_output['sfii']))
    
    # Test variables to perturb
    vars_to_perturb = ['h_norm', 'sigma0_norm', 'tcw_norm']
    
    for var in vars_to_perturb:
        if var not in base_inputs:
            continue
            
        for p in perturbations:
            if p == 0.0:
                continue
                
            # Create perturbed copy
            test_inputs = base_inputs.copy()
            # Perturb by percentage of the base value
            perturbed_val = test_inputs[var] * (1.0 + p)
            # Clip to valid [0, 1] range to avoid validation errors
            test_inputs[var] = np.clip(perturbed_val, 0.0, 1.0)
            
            # Compute new SFII
            test_output = engine.compute_all(test_inputs)
            test_sfii = float(np.mean(test_output['sfii']))
            
            # Record variance
            sfii_change = test_sfii - base_sfii
            
            results.append({
                'Variable': var,
                'Perturbation': f"{p*100}%",
                'Base_SFII': base_sfii,
                'Perturbed_SFII': test_sfii,
                'Absolute_Change': sfii_change
            })
            
    df = pd.DataFrame(results)
    logger.info("Sensitivity Analysis Complete.")
    return df
