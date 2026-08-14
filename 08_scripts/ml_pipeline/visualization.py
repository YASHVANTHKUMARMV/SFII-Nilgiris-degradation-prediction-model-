import os
import logging
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

logger = logging.getLogger("ML_Pipeline.Visualization")

def plot_training_curves(train_losses: list, val_losses: list, output_dir: str, model_name: str):
    logger.info(f"Generating training curves for {model_name}...")
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss', color='blue', linewidth=2)
    plt.plot(val_losses, label='Validation Loss', color='orange', linewidth=2)
    plt.title(f"{model_name} - Training & Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    out_path = os.path.join(output_dir, f"{model_name}_loss_curve.png")
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved loss curve to {out_path}")

def plot_confusion_matrix(cm: np.ndarray, output_dir: str, model_name: str, class_names=['Intact', 'Degraded']):
    logger.info(f"Generating confusion matrix for {model_name}...")
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title(f"{model_name} - Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    
    out_path = os.path.join(output_dir, f"{model_name}_confusion_matrix.png")
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved confusion matrix to {out_path}")

def generate_performance_table(experiments: list, output_dir: str):
    logger.info("Generating final performance benchmark table...")
    
    rows = []
    for exp in experiments:
        row = {
            'Model': exp['model'],
            'Mode': exp['mode'],
            'Accuracy': exp['metrics']['accuracy'],
            'F1_Score': exp['metrics']['f1_score'],
            'Log_Loss': exp['metrics']['log_loss']
        }
        rows.append(row)
        
    df = pd.DataFrame(rows)
    out_path = os.path.join(output_dir, "model_performance_benchmark.csv")
    df.to_csv(out_path, index=False)
    logger.info(f"Saved benchmark table to {out_path}")
    return df
