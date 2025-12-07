import optuna
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import recall_score
import pandas as pd
import numpy as np

def xgb_hyperparameter_tuning(X, y, scale_pos_weight=1, n_trials=20):
    """
    Use Optuna to optimize XGBoost hyperparameters.
    """
    def objective(trial):
        params = {
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'n_estimators': trial.suggest_int('n_estimators', 100, 500),
            'scale_pos_weight': scale_pos_weight,
            'random_state': 42,
            'use_label_encoder': False,
            'eval_metric': 'logloss',
            'verbosity': 0
        }
        model = XGBClassifier(**params)
        skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        recall_scores = []
        for train_idx, val_idx in skf.split(X, y):
            X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]
            model.fit(X_tr, y_tr)
            preds = model.predict(X_val)
            recall_scores.append(recall_score(y_val, preds, pos_label=0))
        return np.mean(recall_scores)
    
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials)
    return study.best_trial.params
