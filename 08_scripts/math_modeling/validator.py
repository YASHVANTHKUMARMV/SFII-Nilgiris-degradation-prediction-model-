import logging
import numpy as np

logger = logging.getLogger("SFII_Math.Validator")

class SFIIMathValidator:
    """
    Strict mathematical validator for all SFII inputs.
    Checks boundaries, NaN presence, and dimensionality.
    """
    
    @staticmethod
    def check_is_tensor_or_array(obj, name: str):
        # We accept either numpy array or torch tensor
        if not (hasattr(obj, 'shape') or hasattr(obj, 'size')):
            raise TypeError(f"Input {name} must be a tensor or array. Got {type(obj)}.")
            
    @staticmethod
    def check_no_nans(obj, name: str):
        # Using duck typing for cross-compatibility between numpy and torch
        if hasattr(obj, 'isnan'):
            has_nan = bool(obj.isnan().any())
        else:
            has_nan = bool(np.isnan(obj).any())
            
        if has_nan:
            raise ValueError(f"Mathematical instability: NaN detected in {name}.")
            
    @staticmethod
    def check_no_infs(obj, name: str):
        if hasattr(obj, 'isinf'):
            has_inf = bool(obj.isinf().any())
        else:
            has_inf = bool(np.isinf(obj).any())
            
        if has_inf:
            raise ValueError(f"Mathematical instability: Inf detected in {name}.")
            
    @staticmethod
    def check_bounds(obj, name: str, min_val: float, max_val: float):
        if hasattr(obj, 'min'):
            c_min = float(obj.min())
            c_max = float(obj.max())
        else:
            c_min = float(np.min(obj))
            c_max = float(np.max(obj))
            
        if c_min < min_val or c_max > max_val:
            logger.warning(f"Input {name} out of bounds [{min_val}, {max_val}]. Got [{c_min}, {c_max}]. Clipping will occur.")
            
    @classmethod
    def validate_inputs(cls, inputs: dict):
        """
        Validates the entire input dictionary before passing to the engine.
        """
        logger.info("Running strict mathematical validation on inputs...")
        
        # 1. Structural inputs should be normalized [0, 1]
        for var in ['h_norm', 'sigma0_norm', 'tcw_norm']:
            if var in inputs:
                cls.check_is_tensor_or_array(inputs[var], var)
                cls.check_no_nans(inputs[var], var)
                cls.check_no_infs(inputs[var], var)
                cls.check_bounds(inputs[var], var, 0.0, 1.0)
                
        # 2. Spectral inputs (can be any range, but shouldn't be nan/inf)
        for var in ['vi_current', 'vi_pre', 'vi_ref']:
            if var in inputs:
                cls.check_is_tensor_or_array(inputs[var], var)
                cls.check_no_nans(inputs[var], var)
                cls.check_no_infs(inputs[var], var)
                
        # 3. ERS inputs
        if 'cov_ref' in inputs:
            cls.check_is_tensor_or_array(inputs['cov_ref'], 'cov_ref')
            cls.check_no_nans(inputs['cov_ref'], 'cov_ref')
            # Check square
            shape = inputs['cov_ref'].shape
            if shape[-1] != shape[-2]:
                raise ValueError(f"Covariance matrix must be square. Got shape {shape}.")
                
        logger.info("All mathematical inputs validated successfully.")
        return True
