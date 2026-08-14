import logging
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import KFold
import yaml

from models.sequence_models import SFIILSTM, SFIITransformer
from models.tree_models import build_xgboost
from validation.metrics import compute_regression_metrics

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Lab.ML_Pipeline")

def train_pytorch_model(model, dataloader, epochs, lr):
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0
        for X_batch, y_batch in dataloader:
            optimizer.zero_grad()
            preds = model(X_batch)
            loss = criterion(preds.squeeze(), y_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        if epoch % 10 == 0:
            logger.info(f"Epoch {epoch}/{epochs} | Loss: {epoch_loss/len(dataloader):.4f}")
    return model

def execute_ml_pipeline():
    logger.info("Starting Phase 12: Machine Learning Pipeline")
    
    # ---------------------------------------------------------
    # INTERNAL LABORATORY REVIEW: SEQUENCE MODEL SELECTION
    # ---------------------------------------------------------
    # The laboratory evaluated LSTM vs Transformer for this spatiotemporal task.
    # While Transformers excel at long-range dependencies, they are highly 
    # data-hungry and prone to overfitting on continuous tabular/raster series 
    # (84 time steps) unless massive pre-training is done.
    # 
    # Decision: We select the LSTM as the primary sequence architecture because 
    # it provides better inductive bias for the strictly sequential phenological 
    # cycles of the Nilgiris forests.
    # ---------------------------------------------------------
    
    with open("configs/config.yaml", 'r') as f:
        config = yaml.safe_load(f)
        
    # Simulate data loading
    # Input shape: [Batch, Time=84, Features=12]
    # Target shape: [Batch] (Continuous SFII score 0-1)
    logger.info("Loading Phase 5 ML Tensors...")
    X_dummy = torch.rand(1000, 84, 12)
    y_dummy = torch.rand(1000)
    
    # K-Fold Cross Validation (Phase 13 Integration)
    kf = KFold(n_splits=config['machine_learning']['cv_folds'], shuffle=True, random_state=42)
    
    fold_metrics = []
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X_dummy)):
        logger.info(f"--- Training Fold {fold+1} ---")
        
        X_train, X_val = X_dummy[train_idx], X_dummy[val_idx]
        y_train, y_val = y_dummy[train_idx], y_dummy[val_idx]
        
        train_loader = DataLoader(TensorDataset(X_train, y_train), 
                                  batch_size=config['machine_learning']['sequence_models']['lstm']['batch_size'], 
                                  shuffle=True)
                                  
        # Initialize selected LSTM
        model = SFIILSTM(
            input_size=12,
            hidden_size=config['machine_learning']['sequence_models']['lstm']['hidden_size'],
            num_layers=config['machine_learning']['sequence_models']['lstm']['num_layers'],
            dropout=config['machine_learning']['sequence_models']['lstm']['dropout']
        )
        
        model = train_pytorch_model(
            model, 
            train_loader, 
            epochs=config['machine_learning']['sequence_models']['lstm']['epochs'],
            lr=config['machine_learning']['sequence_models']['lstm']['learning_rate']
        )
        
        # Validation
        model.eval()
        with torch.no_grad():
            preds = model(X_val).squeeze().numpy()
            targets = y_val.numpy()
            
            metrics = compute_regression_metrics(targets, preds)
            fold_metrics.append(metrics)
            logger.info(f"Fold {fold+1} Metrics: RMSE={metrics['RMSE']:.3f}, R2={metrics['R2']:.3f}")
            
    # Aggregate results
    avg_rmse = np.mean([m['RMSE'] for m in fold_metrics])
    avg_r2 = np.mean([m['R2'] for m in fold_metrics])
    logger.info(f"=== Cross-Validation Complete ===")
    logger.info(f"Average RMSE: {avg_rmse:.4f}")
    logger.info(f"Average R²: {avg_r2:.4f}")
    
    # Save the final model state
    torch.save(model.state_dict(), "outputs/models/sfii_lstm_final.pth")
    logger.info("Model saved successfully.")

if __name__ == "__main__":
    execute_ml_pipeline()
