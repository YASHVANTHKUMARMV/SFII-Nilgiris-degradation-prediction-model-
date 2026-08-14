import logging
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import RandomizedSearchCV

logger = logging.getLogger("ML_Pipeline.Models.TreeBased")

class RandomForestModel:
    def __init__(self, random_state: int = 42):
        self.model = RandomForestClassifier(random_state=random_state, class_weight='balanced', n_jobs=2)
        self.name = "Random Forest"
        
    def tune_and_fit(self, X_train, y_train):
        logger.info(f"Using best known hyperparameters for {self.name} to avoid deadlocks...")
        self.model = RandomForestClassifier(n_estimators=500, max_depth=30, min_samples_split=10, 
                                            random_state=42, class_weight='balanced', n_jobs=2)
        self.model.fit(X_train, y_train)
        return self

    def predict(self, X_test):
        return self.model.predict(X_test)
        
    def predict_proba(self, X_test):
        return self.model.predict_proba(X_test)[:, 1]

class XGBoostModel:
    def __init__(self, random_state: int = 42):
        # scale_pos_weight could be dynamically set for class balance, we use a default
        self.model = XGBClassifier(random_state=random_state, use_label_encoder=False, eval_metric='logloss')
        self.name = "XGBoost"
        
    def tune_and_fit(self, X_train, y_train):
        logger.info(f"Using best known hyperparameters for {self.name} to avoid deadlocks...")
        self.model = XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1,
                                   random_state=42, n_jobs=2, use_label_encoder=False, eval_metric='logloss')
        self.model.fit(X_train, y_train)
        return self

    def predict(self, X_test):
        return self.model.predict(X_test)
        
    def predict_proba(self, X_test):
        return self.model.predict_proba(X_test)[:, 1]
