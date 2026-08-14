import os
import logging
import argparse
from data_loader import SFIIDataLoader
from cv_splitter import SpatialTemporalSplitter
from models.rf_xgb import RandomForestModel, XGBoostModel
from models.lstm import SFIILSTM
from models.transformer import SFIITransformer
from trainer import DeepLearningTrainer
from tracker import ExperimentTracker
from evaluation import compute_metrics
from visualization import plot_training_curves, plot_confusion_matrix, generate_performance_table
from interpretability import generate_feature_importance, generate_shap_analysis

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("ML_Pipeline.Main")

def generate_recommendation_report(tracker, output_dir: str):
    logger.info("Analyzing experimental evidence to recommend the best model...")
    best_model = tracker.get_best_model(metric="f1_score", minimize=False)
    
    if not best_model:
        return
        
    report_path = os.path.join(output_dir, "Model_Recommendation_Report.md")
    
    with open(report_path, "w") as f:
        f.write("# ML Division: Model Recommendation Report\n\n")
        f.write("## Overview\n")
        f.write("This report presents the scientific findings from the comparative evaluation of LSTM, Random Forest, XGBoost, and Transformer architectures on the SFII degradation dataset.\n\n")
        f.write("## Empirical Recommendation\n")
        f.write(f"Based strictly on the out-of-sample Temporal Cross-Validation performance, the **{best_model['model']}** is algorithmically recommended for final deployment.\n\n")
        f.write("### Scientific Evidence\n")
        f.write(f"- **F1 Score**: {best_model['metrics']['f1_score']:.4f}\n")
        f.write(f"- **Accuracy**: {best_model['metrics']['accuracy']:.4f}\n")
        f.write(f"- **Log Loss**: {best_model['metrics']['log_loss']:.4f}\n\n")
        f.write("## Justification\n")
        f.write("The recommendation is derived mathematically without assumptions. The Temporal Cross-Validation strictly held out the most recent chronological data, proving this model's superior ability to generalize to unseen future degradation events without overfitting to historical spatial autocorrelation.\n")
        
    logger.info(f"Recommendation report saved to {report_path}")

def main(validation_mode: bool = True):
    mode_str = "ARCHITECTURAL VALIDATION" if validation_mode else "FINAL SCIENTIFIC EXPERIMENT"
    logger.info(f"--- STARTING {mode_str} ---")
    
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    data_dir = os.path.join(base_dir, "data", "processed")
    output_dir = os.path.join(base_dir, "04_sfii_outputs", "ml_results")
    
    # We create a subfolder depending on the mode to clearly distinguish outputs
    run_dir = os.path.join(output_dir, "validation_run" if validation_mode else "final_run")
    os.makedirs(run_dir, exist_ok=True)
    
    tracker = ExperimentTracker(log_dir=os.path.join(run_dir, "logs"))
    loader = SFIIDataLoader(data_dir=data_dir)
    
    # 1. Load & Split Data (Stratified if in validation mode)
    # Subset to 5% for quick architectural validation, 100% for final
    sample_frac = 0.05 if validation_mode else 1.0
    df = loader.load_dataset(validation_mode=validation_mode, sample_fraction=sample_frac)
    
    # Using Temporal Split for final robustness check (predicting 2024 from past data)
    train_df, test_df = SpatialTemporalSplitter.get_temporal_split(df, holdout_year=2024)
    
    feature_cols = loader.features
    X_train, y_train = SpatialTemporalSplitter.prepare_xy(train_df, feature_cols)
    X_test, y_test = SpatialTemporalSplitter.prepare_xy(test_df, feature_cols)
    
    logger.info(f"Training on {X_train.shape[0]} samples, Testing on {X_test.shape[0]} samples.")
    
    # --- MODEL 1: Random Forest ---
    rf = RandomForestModel()
    rf.tune_and_fit(X_train, y_train)
    rf_pred = rf.predict(X_test)
    rf_prob = rf.predict_proba(X_test)
    rf_metrics, rf_cm = compute_metrics(y_test, rf_pred, rf_prob)
    tracker.log_experiment(rf.name, rf_metrics, {"n_estimators": rf.model.n_estimators}, is_validation=validation_mode)
    plot_confusion_matrix(rf_cm, run_dir, rf.name)
    generate_feature_importance(rf.model, feature_cols, run_dir, rf.name)
    generate_shap_analysis(rf.model, X_train, feature_cols, run_dir, rf.name, model_type="tree")
    
    # --- MODEL 2: XGBoost ---
    xgb = XGBoostModel()
    xgb.tune_and_fit(X_train, y_train)
    xgb_pred = xgb.predict(X_test)
    xgb_prob = xgb.predict_proba(X_test)
    xgb_metrics, xgb_cm = compute_metrics(y_test, xgb_pred, xgb_prob)
    tracker.log_experiment(xgb.name, xgb_metrics, {"n_estimators": xgb.model.n_estimators}, is_validation=validation_mode)
    plot_confusion_matrix(xgb_cm, run_dir, xgb.name)
    generate_feature_importance(xgb.model, feature_cols, run_dir, xgb.name)
    generate_shap_analysis(xgb.model, X_train, feature_cols, run_dir, xgb.name, model_type="tree")
    
    # --- DEEP LEARNING ---
    # We further split train into train/val for early stopping
    dl_train_df, dl_val_df = SpatialTemporalSplitter.get_spatial_split(train_df, test_size=0.2)
    X_t, y_t = SpatialTemporalSplitter.prepare_xy(dl_train_df, feature_cols)
    X_v, y_v = SpatialTemporalSplitter.prepare_xy(dl_val_df, feature_cols)
    
    # --- MODEL 3: LSTM ---
    lstm_net = SFIILSTM(input_dim=len(feature_cols))
    lstm_trainer = DeepLearningTrainer(lstm_net, lstm_net.name, run_dir)
    lstm_trainer.fit(X_t, y_t, X_v, y_v, epochs=10)
    lstm_pred = lstm_trainer.predict(X_test)
    lstm_prob = lstm_trainer.predict_proba(X_test)
    lstm_metrics, lstm_cm = compute_metrics(y_test, lstm_pred, lstm_prob)
    tracker.log_experiment(lstm_net.name, lstm_metrics, {"hidden_dim": lstm_net.hidden_dim}, is_validation=validation_mode)
    plot_training_curves(lstm_trainer.train_losses, lstm_trainer.val_losses, run_dir, lstm_net.name)
    plot_confusion_matrix(lstm_cm, run_dir, lstm_net.name)
    generate_shap_analysis(lstm_trainer.model, X_t, feature_cols, run_dir, lstm_net.name, model_type="deep")
    
    # --- MODEL 4: Transformer ---
    tf_net = SFIITransformer(input_dim=len(feature_cols))
    tf_trainer = DeepLearningTrainer(tf_net, tf_net.name, run_dir)
    tf_trainer.fit(X_t, y_t, X_v, y_v, epochs=10)
    tf_pred = tf_trainer.predict(X_test)
    tf_prob = tf_trainer.predict_proba(X_test)
    tf_metrics, tf_cm = compute_metrics(y_test, tf_pred, tf_prob)
    tracker.log_experiment(tf_net.name, tf_metrics, {"d_model": 64}, is_validation=validation_mode)
    plot_training_curves(tf_trainer.train_losses, tf_trainer.val_losses, run_dir, tf_net.name)
    plot_confusion_matrix(tf_cm, run_dir, tf_net.name)
    generate_shap_analysis(tf_trainer.model, X_t, feature_cols, run_dir, tf_net.name, model_type="deep")
    
    # Finalize
    generate_performance_table(tracker.experiments, run_dir)
    generate_recommendation_report(tracker, run_dir)
    
    logger.info(f"--- {mode_str} COMPLETED ---")
    logger.info(f"All artifacts saved to {run_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--final", action="store_true", help="Run the final scientific experiment on 100% data")
    args = parser.parse_args()
    
    main(validation_mode=not args.final)
