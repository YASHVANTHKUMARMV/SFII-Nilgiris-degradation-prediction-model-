# Feature Selection & Validation Pipeline

This package executes rigorous statistical validation of the feature space prior to Machine Learning. 

It implements both model-agnostic (pre-training) metrics and model-specific (post-training) game-theoretic metrics to objectively determine the optimal subset of features.

## Architecture
- `pre_training_stats.py`: Pearson/Spearman, VIF, Mutual Information, PCA, Hierarchical Clustering.
- `post_training_stats.py`: Recursive Feature Elimination (RFE), Permutation Importance, SHAP.
- `visualization.py`: Correlation heatmaps, PCA scree plots, Dendrograms, and SHAP summary plots.

## Usage
Run these validation scripts on a flattened, tabular subset (e.g., a 50,000-pixel sample exported to Parquet from the Feature Engineering phase). Do not run these on the entire 3D array space simultaneously to prevent OOM errors.
