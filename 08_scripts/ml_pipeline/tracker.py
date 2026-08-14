import os
import json
import logging
from datetime import datetime

logger = logging.getLogger("ML_Pipeline.Tracker")

class ExperimentTracker:
    def __init__(self, log_dir: str):
        self.log_dir = log_dir
        os.makedirs(self.log_dir, exist_ok=True)
        self.experiments = []
        
    def log_experiment(self, model_name: str, metrics: dict, params: dict, is_validation: bool = True):
        exp_id = f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{model_name}"
        
        experiment = {
            "experiment_id": exp_id,
            "model": model_name,
            "mode": "Architectural Validation" if is_validation else "Final Scientific Experiment",
            "timestamp": datetime.now().isoformat(),
            "metrics": metrics,
            "hyperparameters": params
        }
        
        self.experiments.append(experiment)
        
        log_file = os.path.join(self.log_dir, f"{exp_id}.json")
        with open(log_file, 'w') as f:
            json.dump(experiment, f, indent=4)
            
        logger.info(f"Experiment logged to {log_file}")
        
    def get_best_model(self, metric: str = "f1", minimize: bool = False):
        if not self.experiments:
            logger.warning("No experiments logged.")
            return None
            
        def get_metric(exp):
            return exp['metrics'].get(metric, float('inf') if minimize else -float('inf'))
            
        best_exp = min(self.experiments, key=get_metric) if minimize else max(self.experiments, key=get_metric)
        logger.info(f"Best model based on {metric}: {best_exp['model']} (Score: {get_metric(best_exp):.4f})")
        return best_exp
