import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import logging

logger = logging.getLogger("ML_Pipeline.Trainer")

class DeepLearningTrainer:
    def __init__(self, model: nn.Module, model_name: str, output_dir: str, 
                 patience: int = 5, learning_rate: float = 1e-3, device: str = 'cpu'):
        self.model = model.to(device)
        self.model_name = model_name
        self.output_dir = output_dir
        self.patience = patience
        self.device = device
        
        self.criterion = nn.BCELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        
        self.train_losses = []
        self.val_losses = []
        
        os.makedirs(self.output_dir, exist_ok=True)
        self.checkpoint_path = os.path.join(self.output_dir, f"{model_name}_best.pth")

    def fit(self, X_train, y_train, X_val, y_val, epochs: int = 50, batch_size: int = 64):
        logger.info(f"Starting training for {self.model_name}...")
        
        # Convert to tensors
        X_train_t = torch.tensor(X_train, dtype=torch.float32)
        y_train_t = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
        X_val_t = torch.tensor(X_val, dtype=torch.float32)
        y_val_t = torch.tensor(y_val, dtype=torch.float32).unsqueeze(1)
        
        # We need 3D tensors for sequence models if they aren't already
        if X_train_t.dim() == 2:
            X_train_t = X_train_t.unsqueeze(1)
            X_val_t = X_val_t.unsqueeze(1)
            
        train_dataset = TensorDataset(X_train_t, y_train_t)
        val_dataset = TensorDataset(X_val_t, y_val_t)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        best_val_loss = float('inf')
        epochs_no_improve = 0
        
        for epoch in range(epochs):
            self.model.train()
            train_loss = 0.0
            
            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                
                self.optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = self.criterion(outputs, batch_y)
                loss.backward()
                self.optimizer.step()
                
                train_loss += loss.item() * batch_X.size(0)
                
            train_loss /= len(train_loader.dataset)
            self.train_losses.append(train_loss)
            
            # Validation
            self.model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                    outputs = self.model(batch_X)
                    loss = self.criterion(outputs, batch_y)
                    val_loss += loss.item() * batch_X.size(0)
                    
            val_loss /= len(val_loader.dataset)
            self.val_losses.append(val_loss)
            
            logger.info(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
            
            # Early Stopping and Checkpointing
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_no_improve = 0
                torch.save(self.model.state_dict(), self.checkpoint_path)
                logger.info(f"Validation loss decreased. Model checkpoint saved to {self.checkpoint_path}")
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= self.patience:
                    logger.info(f"Early stopping triggered after {epoch+1} epochs.")
                    break
                    
        # Load best model
        self.model.load_state_dict(torch.load(self.checkpoint_path))
        return self

    def predict_proba(self, X_test):
        self.model.eval()
        X_test_t = torch.tensor(X_test, dtype=torch.float32)
        if X_test_t.dim() == 2:
            X_test_t = X_test_t.unsqueeze(1)
        X_test_t = X_test_t.to(self.device)
        
        with torch.no_grad():
            preds = self.model(X_test_t).cpu().numpy().flatten()
            
        return preds
        
    def predict(self, X_test, threshold=0.5):
        probs = self.predict_proba(X_test)
        return (probs >= threshold).astype(int)
