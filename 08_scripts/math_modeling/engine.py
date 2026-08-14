import logging
from typing import Literal

logger = logging.getLogger("SFII_Math.Engine")

class SFIIEngine:
    def __init__(self, backend: Literal['numpy', 'pytorch'] = 'pytorch'):
        self.backend_type = backend
        
        if backend == 'pytorch':
            import torch
            try:
                from . import pytorch_backend as backend_module
            except ImportError:
                import pytorch_backend as backend_module
            self.backend_module = backend_module
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            logger.info(f"Initialized PyTorch backend on device: {self.device}")
        elif backend == 'numpy':
            try:
                from . import numpy_backend as backend_module
            except ImportError:
                import numpy_backend as backend_module
            self.backend_module = backend_module
            self.device = 'cpu'
            logger.info("Initialized NumPy backend.")
        else:
            raise ValueError(f"Unknown backend: {backend}")

    def compute_all(self, inputs: dict) -> dict:
        """
        Executes the entire SFII pipeline.
        
        Inputs expected:
        - vi_current, vi_pre, vi_ref
        - h_norm, sigma0_norm, tcw_norm
        - time_arr, disturbance_times, disturbance_mags
        - f_dist, mu_ref, cov_ref
        """
        # Ensure inputs are moved to correct device if using PyTorch
        if self.backend_type == 'pytorch':
            for k, v in inputs.items():
                if hasattr(v, 'to'):
                    inputs[k] = v.to(self.device)

        mod = self.backend_module

        # 1. Compute SRT
        srt = mod.compute_srt(inputs['vi_current'], inputs['vi_pre'], inputs['vi_ref'])
        
        # 2. Compute SBP
        sbp = mod.compute_sbp(inputs['h_norm'], inputs['sigma0_norm'], inputs['tcw_norm'])
        
        # 3. Compute DMF
        dmf = mod.compute_dmf(inputs['time_arr'], inputs['disturbance_times'], inputs['disturbance_mags'])
        dmf_max = inputs.get('dmf_max', 1.0)
        
        # 4. Compute ERS
        ers = mod.compute_ers(inputs['f_dist'], inputs['mu_ref'], inputs['cov_ref'])
        
        # 5. Compute FRP
        frp = mod.compute_frp(srt, sbp)
        
        # 6. Compute SFII
        sfii = mod.compute_sfii(sbp, dmf, ers, frp, dmf_max=dmf_max)
        
        return {
            'srt': srt,
            'sbp': sbp,
            'dmf': dmf,
            'ers': ers,
            'frp': frp,
            'sfii': sfii
        }
