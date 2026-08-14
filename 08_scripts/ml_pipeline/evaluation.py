import logging
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, log_loss

logger = logging.getLogger("ML_Pipeline.Evaluation")

def compute_metrics(y_true, y_pred, y_prob):
    """Computes standard ML classification metrics."""
    logger.info("Computing evaluation metrics...")
    
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='weighted')
    prec = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    rec = recall_score(y_true, y_pred, average='weighted')
    
    loss = None
    if y_prob is not None:
        try:
            loss = log_loss(y_true, y_prob)
        except Exception as e:
            logger.warning(f"Could not compute log_loss: {e}")
            
    cm = confusion_matrix(y_true, y_pred)
    
    metrics = {
        'accuracy': float(acc),
        'f1_score': float(f1),
        'precision': float(prec),
        'recall': float(rec),
        'log_loss': float(loss) if loss is not None else None
    }
    
    logger.info(f"Evaluation completed. F1 Score: {f1:.4f}")
    return metrics, cm
