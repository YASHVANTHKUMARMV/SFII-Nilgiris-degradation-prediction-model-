import matplotlib.pyplot as plt
import seaborn as sns
import shap
import pandas as pd
from scipy.cluster import hierarchy
import logging

logger = logging.getLogger("FeatureSelection.Visualization")

def plot_correlation_heatmap(corr_df: pd.DataFrame, title: str, output_path: str):
    """Generates a publication-quality correlation heatmap."""
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_df, annot=True, cmap='coolwarm', vmin=-1, vmax=1, fmt=".2f", linewidths=0.5)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    logger.info(f"Saved correlation heatmap to {output_path}")

def plot_shap_summary(explainer, shap_values, df_X: pd.DataFrame, output_path: str):
    """Generates SHAP summary plot showing feature importance and directionality."""
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, df_X, show=False)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved SHAP summary plot to {output_path}")

def plot_dendrogram(linkage, labels, output_path: str):
    """Generates a hierarchical clustering dendrogram."""
    plt.figure(figsize=(10, 5))
    hierarchy.dendrogram(linkage, labels=labels, leaf_rotation=90)
    plt.title("Feature Hierarchical Clustering (Spearman Distance)")
    plt.ylabel("Distance")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    logger.info(f"Saved Dendrogram to {output_path}")

def plot_pca_variance(explained_variance_ratio, output_path: str):
    """Generates a scree plot for PCA explained variance."""
    plt.figure(figsize=(8, 5))
    cumulative_variance = explained_variance_ratio.cumsum()
    plt.plot(range(1, len(explained_variance_ratio) + 1), cumulative_variance, marker='o', linestyle='--')
    plt.title("PCA Cumulative Explained Variance")
    plt.xlabel("Number of Components")
    plt.ylabel("Cumulative Explained Variance")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    logger.info(f"Saved PCA variance plot to {output_path}")
