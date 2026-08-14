import os
import sys
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import rasterio
from rasterio.transform import from_origin

# Add parent directory to path so we can import internal modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml_pipeline.models.lstm import SFIILSTM
from ml_pipeline.data_loader import SFIIDataLoader
from ml_pipeline.cv_splitter import SpatialTemporalSplitter

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("ML_Pipeline.Inference")

def reconstruct_grid(df, value_col):
    """
    Reconstruct a 2D spatial grid from flattened dataframe based on x, y coordinates.
    Since we don't have true georeferenced coordinates, we assume a synthetic grid where
    x and y are pixel indices or mapped to a 200x500 grid.
    """
    # Assuming x and y are continuous. If they are categorical coordinates, pivot will work.
    # We round them to integers assuming they are grid indices.
    # In the mock data, they were generated randomly or grid-like.
    grid = df.pivot_table(index='y', columns='x', values=value_col).values
    return grid

def export_geotiff(data_matrix, output_path, crs='EPSG:4326'):
    """
    Export a 2D numpy array to a GeoTIFF format.
    """
    height, width = data_matrix.shape
    # Dummy transform for demonstration since we lack raw CRS
    transform = from_origin(76.5, 11.5, 0.0001, 0.0001)
    
    with rasterio.open(
        output_path,
        'w',
        driver='GTiff',
        height=height,
        width=width,
        count=1,
        dtype=data_matrix.dtype,
        crs=crs,
        transform=transform,
    ) as dst:
        dst.write(data_matrix, 1)

def main():
    logger.info("--- STARTING INFERENCE & MAP GENERATION ---")
    
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    data_dir = os.path.join(base_dir, "data", "processed")
    model_path = os.path.join(base_dir, "04_sfii_outputs", "ml_results", "final_run", "LSTM_best.pth")
    output_dir = os.path.join(base_dir, "04_sfii_outputs", "inference_maps")
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Load the complete dataset
    loader = SFIIDataLoader(data_dir=data_dir)
    # 100% data, no synthetic
    df = loader.load_dataset(validation_mode=False, sample_fraction=1.0)
    
    feature_cols = loader.features
    
    # 2. Load the trained LSTM model
    logger.info(f"Loading best model from {model_path}...")
    model = SFIILSTM(input_dim=len(feature_cols))
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    
    # 3. Predict on the entire dataset
    logger.info("Running inference on full spatial-temporal data...")
    X_full, _ = SpatialTemporalSplitter.prepare_xy(df, feature_cols)
    X_tensor = torch.tensor(X_full, dtype=torch.float32).unsqueeze(1)
    
    with torch.no_grad():
        probs = model(X_tensor).cpu().numpy().flatten()
        preds = (probs >= 0.5).astype(int)
        
    df['Integrity_Prob'] = probs
    df['Degradation_Prob'] = 1.0 - probs
    df['Prediction'] = preds
    
    # 4. Generate Maps for each year
    years = sorted(df['year'].unique())
    stats = []
    
    for year in years:
        logger.info(f"Generating publication maps for year {year}...")
        year_df = df[df['year'] == year]
        # Reshape the 10000 pixels for this year into a 100x100 spatial grid
        try:
            pred_grid = year_df['Prediction'].values.reshape(100, 100)
            deg_prob_grid = year_df['Degradation_Prob'].values.reshape(100, 100)
            int_prob_grid = year_df['Integrity_Prob'].values.reshape(100, 100)
        except Exception as e:
            logger.warning(f"Could not reshape grid for {year}: {e}. Skipping spatial mapping.")
            continue
            
        # Export GeoTIFFs
        export_geotiff(pred_grid.astype('float32'), os.path.join(output_dir, f'yearly_prediction_{year}.tif'))
        export_geotiff(deg_prob_grid.astype('float32'), os.path.join(output_dir, f'degradation_prob_{year}.tif'))
        export_geotiff(int_prob_grid.astype('float32'), os.path.join(output_dir, f'recovery_prob_{year}.tif'))
        
        # Export Publication-Quality PNGs
        plt.figure(figsize=(10, 8))
        sns.heatmap(deg_prob_grid, cmap='Reds', cbar_kws={'label': 'Degradation Probability'})
        plt.title(f'Forest Degradation Probability Map ({year})', fontsize=16)
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'Degradation_Map_{year}.png'), dpi=300)
        plt.close()
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(int_prob_grid, cmap='Greens', cbar_kws={'label': 'Recovery/Integrity Probability'})
        plt.title(f'Forest Recovery Probability Map ({year})', fontsize=16)
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'Recovery_Map_{year}.png'), dpi=300)
        plt.close()
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(pred_grid, cmap='coolwarm', cbar_kws={'label': 'Class (0=Degraded, 1=Intact)'})
        plt.title(f'Forest Integrity Classification Map ({year})', fontsize=16)
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'Prediction_Map_{year}.png'), dpi=300)
        plt.close()
        
        # Calculate statistics
        total_pixels = len(year_df)
        degraded = np.sum(year_df['Prediction'] == 0)
        recovering = np.sum(year_df['Prediction'] == 1)
        mean_deg_prob = year_df['Degradation_Prob'].mean()
        
        stats.append({
            'Year': year,
            'Total_Pixels': total_pixels,
            'Degraded_Pixels': degraded,
            'Intact_Pixels': recovering,
            'Degradation_Area_Percent': (degraded / total_pixels) * 100,
            'Mean_Degradation_Probability': mean_deg_prob
        })
        
    # 5. Export Statistics
    stats_df = pd.DataFrame(stats)
    stats_df.to_csv(os.path.join(output_dir, 'inference_statistics.csv'), index=False)
    logger.info(f"Statistics exported successfully to {output_dir}/inference_statistics.csv")
    logger.info("--- INFERENCE COMPLETED ---")

if __name__ == '__main__':
    main()
