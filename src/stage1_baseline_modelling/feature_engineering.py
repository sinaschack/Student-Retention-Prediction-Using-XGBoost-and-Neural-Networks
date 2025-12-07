import xgboost as xgb
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import recall_score
import optuna

def train_default_xgb(X_train, y_train, scale_pos_weight):
    model = xgb.XGBClassifier(
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        eval_metric='logloss'
    )
    model.fit(X_train, y_train)
    return model

def xgb_hyperparameter_tuning(X_train, y_train, scale_pos_weight, n_trials=10):
    def objective(trial):
        params = {
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'n_estimators': trial.suggest_int('n_estimators', 100, 500),
            'scale_pos_weight': scale_pos_weight,
            'eval_metric': 'logloss',
            'random_state': 42,
            'use_label_encoder': False,
            'verbosity': 0
        }
        model = xgb.XGBClassifier(**params)
        skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        recall_scores = []
        for train_idx, val_idx in skf.split(X_train, y_train):
            X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
            y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
            model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
            preds = model.predict(X_val)
            recall_scores.append(recall_score(y_val, preds, pos_label=0))
        return sum(recall_scores) / len(recall_scores)

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials)
    return study.best_trial.params
