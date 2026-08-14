import os
import logging
import numpy as np
import matplotlib.pyplot as plt
import shap
import warnings
# Suppress SHAP warnings for cleaner output
warnings.filterwarnings("ignore")

logger = logging.getLogger("ML_Pipeline.Interpretability")

def generate_feature_importance(model, feature_names: list, output_dir: str, model_name: str):
    """Extracts and plots Gini importance for tree-based models."""
    if not hasattr(model, 'feature_importances_'):
        logger.warning(f"Model {model_name} does not have feature_importances_ attribute.")
        return
        
    logger.info(f"Extracting feature importance for {model_name}...")
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    plt.figure(figsize=(10, 6))
    plt.title(f"{model_name} - Feature Importance")
    plt.bar(range(len(importances)), importances[indices], align="center", color="teal")
    plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=45)
    plt.xlim([-1, len(importances)])
    plt.tight_layout()
    
    out_path = os.path.join(output_dir, f"{model_name}_feature_importance.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    logger.info(f"Saved feature importance plot to {out_path}")

def generate_shap_analysis(model, X_train, feature_names: list, output_dir: str, model_name: str, model_type: str = "tree"):
    """
    Generates SHAP summary plots.
    Uses TreeExplainer for RF/XGB and DeepExplainer for Neural Networks.
    """
    logger.info(f"Generating SHAP analysis for {model_name} ({model_type})...")
    
    try:
        # Sample background data to speed up SHAP calculation
        background = X_train[np.random.choice(X_train.shape[0], min(100, X_train.shape[0]), replace=False)]
        
        if model_type == "tree":
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(background)
            # Handle binary classification lists returned by some tree models
            if isinstance(shap_values, list):
                shap_values = shap_values[1] 
        else:
            # DeepExplainer requires tensors
            import torch
            background_t = torch.tensor(background, dtype=torch.float32)
            if background_t.dim() == 2:
                background_t = background_t.unsqueeze(1)
            explainer = shap.DeepExplainer(model, background_t)
            shap_values = explainer.shap_values(background_t)
            
            # Reformat for plotting
            if isinstance(shap_values, list):
                shap_values = shap_values[0]
            if len(shap_values.shape) == 3:
                # Average over sequence length for LSTM/Transformer
                shap_values = np.mean(shap_values, axis=1)
                background = np.mean(background, axis=1) if len(background.shape) == 3 else background
                
        plt.figure(figsize=(8, 6))
        shap.summary_plot(shap_values, background, feature_names=feature_names, show=False)
        plt.title(f"SHAP Summary - {model_name}")
        
        out_path = os.path.join(output_dir, f"{model_name}_shap_summary.png")
        plt.savefig(out_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved SHAP summary to {out_path}")
        
    except Exception as e:
        logger.error(f"Failed to generate SHAP analysis for {model_name}: {e}")
