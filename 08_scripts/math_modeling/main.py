import os
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from engine import SFIIEngine
from validator import SFIIMathValidator
from sensitivity import run_sensitivity_analysis

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("SFII_Math.Main")

def generate_mock_data(B=1, T=10, F=3):
    """Generates a realistic spatial-temporal mock dataset for demonstration."""
    logger.info("Generating mock spatial-temporal input data...")
    inputs = {
        'vi_current': np.linspace(0.8, 0.4, T).reshape(1, T, 1).repeat(B, axis=0),
        'vi_pre': np.ones((B, T, 1)) * 0.8,
        'vi_ref': np.ones((B, T, 1)) * 0.85,
        'h_norm': np.linspace(1.0, 0.5, T).reshape(1, T, 1).repeat(B, axis=0),
        'sigma0_norm': np.linspace(0.9, 0.6, T).reshape(1, T, 1).repeat(B, axis=0),
        'tcw_norm': np.linspace(0.8, 0.3, T).reshape(1, T, 1).repeat(B, axis=0),
        'time_arr': np.arange(2015, 2015 + T).reshape(1, T, 1).repeat(B, axis=0),
        'disturbance_times': np.array([2018]).reshape(1, 1, 1).repeat(B, axis=0),
        'disturbance_mags': np.array([0.5]).reshape(1, 1, 1).repeat(B, axis=0),
        'f_dist': np.random.rand(B, T, F),
        'mu_ref': np.zeros((F,)),
        'cov_ref': np.eye(F),
        'dmf_max': 1.0
    }
    return inputs

def plot_time_series(sfii_series: np.ndarray, time_steps: np.ndarray, output_path: str):
    """Generates a time-series plot of the SFII."""
    plt.figure(figsize=(10, 5))
    plt.plot(time_steps.flatten(), sfii_series.flatten(), marker='o', color='purple', linewidth=2)
    plt.axhline(0.2, color='green', linestyle='--', label='Intact Forest')
    plt.axhline(0.8, color='red', linestyle='--', label='Severe Degradation')
    plt.title("SFII Time Series (Single Pixel)")
    plt.xlabel("Year")
    plt.ylabel("SFII Score")
    plt.ylim(0, 1.1)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.savefig(output_path)
    plt.close()
    logger.info(f"Time series plot saved to {output_path}")

def main():
    logger.info("Initializing SFII Mathematical Modelling Pipeline...")
    
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    output_dir = os.path.join(base_dir, "04_sfii_outputs", "math_results")
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Load Data
    inputs = generate_mock_data(B=100, T=10, F=5) # 100 pixels, 10 years, 5 features
    
    # 2. Validation
    SFIIMathValidator.validate_inputs(inputs)
    
    # 3. Execution (NumPy by default for stability, PyTorch if GPU available)
    # We use NumPy here to demonstrate the batch logic without requiring CUDA hardware
    engine = SFIIEngine(backend='numpy')
    results = engine.compute_all(inputs)
    
    # 4. Save Outputs
    # A. Time Series for first pixel
    time_arr = inputs['time_arr'][0, :, 0]
    sfii_ts = results['sfii'][0, :, 0]
    ts_path = os.path.join(output_dir, "sfii_time_series.png")
    plot_time_series(sfii_ts, time_arr, ts_path)
    
    # B. Statistics
    sfii_mean = np.mean(results['sfii'])
    sfii_std = np.std(results['sfii'])
    stats_df = pd.DataFrame([{"Mean_SFII": float(sfii_mean), "Std_SFII": float(sfii_std)}])
    stats_path = os.path.join(output_dir, "sfii_statistics.csv")
    stats_df.to_csv(stats_path, index=False)
    logger.info(f"SFII Statistics saved to {stats_path}")
    
    # C. Mock Raster (Saving as NPY since we aren't using GDAL here)
    # Assuming shape [B, T] -> Reshape into a mock 10x10 spatial grid for a specific timestep
    if inputs['vi_current'].shape[0] == 100:
        spatial_sfii = results['sfii'][:, -1, 0].reshape(10, 10)
        raster_path = os.path.join(output_dir, "sfii_raster_mock.npy")
        np.save(raster_path, spatial_sfii)
        logger.info(f"SFII raster array saved to {raster_path}")
        
    # 5. Sensitivity Analysis
    base_pixel = {k: v[0:1] if isinstance(v, np.ndarray) and len(v.shape) >= 1 else v for k, v in inputs.items()}
    sens_df = run_sensitivity_analysis(base_pixel, perturbations=[-0.2, -0.1, 0.1, 0.2])
    sens_path = os.path.join(output_dir, "sfii_sensitivity.csv")
    sens_df.to_csv(sens_path, index=False)
    logger.info(f"Sensitivity Analysis saved to {sens_path}")
    
    logger.info("SFII Mathematical Modelling Pipeline completed successfully.")

if __name__ == "__main__":
    main()
