import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, roc_curve, ConfusionMatrixDisplay
from sklearn.model_selection import StratifiedKFold
import optuna
from xgboost import XGBClassifier
from data_preprocessing import X_train, X_test, y_train, y_test

# Initialize XGBoost with scale_pos_weight
num_neg = sum(y_train == 0)
num_pos = sum(y_train == 1)
scale_pos_weight = num_neg / num_pos

model_xgb = XGBClassifier(scale_pos_weight=scale_pos_weight, random_state=42, eval_metric='logloss')
model_xgb.fit(X_train, y_train)

# Feature importance plots
fig, axes = plt.subplots(3, 1, figsize=(15, 18))
xgb.plot_importance(model_xgb, importance_type='gain', ax=axes[0], max_num_features=10, title='Feature Importance (Gain)')
xgb.plot_importance(model_xgb, importance_type='weight', ax=axes[1], max_num_features=10, title='Feature Importance (Weight)')
xgb.plot_importance(model_xgb, importance_type='cover', ax=axes[2], max_num_features=10, title='Feature Importance (Cover)')
plt.tight_layout()
plt.show()

# Predictions
y_pred_xgb = model_xgb.predict(X_test)
y_pred_proba_xgb = model_xgb.predict_proba(X_test)[:, 1]

# Metrics
cm = confusion_matrix(y_test, y_pred_xgb)
print("Confusion Matrix:\n", cm)
print("\nClassification Report:\n", classification_report(y_test, y_pred_xgb))
print(f"Accuracy: {np.round((y_test == y_pred_xgb).mean(), 4)}")
print(f"AUC Score: {roc_auc_score(y_test, y_pred_proba_xgb):.4f}")

# Plot ROC
fpr, tpr, _ = roc_curve(y_test, y_pred_proba_xgb)
plt.figure(figsize=(6, 4))
plt.plot(fpr, tpr, label=f'AUC = {roc_auc_score(y_test, y_pred_proba_xgb):.4f}')
plt.plot([0,1],[0,1],'--',color='gray')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.grid(True)
plt.show()

# Optuna tuning for XGBoost
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
    model = XGBClassifier(**params)
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    recall_scores = []
    for train_idx, val_idx in skf.split(X_train, y_train):
        X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
        model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
        preds = model.predict(X_val)
        recall_scores.append(recall_score(y_val, preds, pos_label=0))
    return np.mean(recall_scores)

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=10)
print("Best params:", study.best_trial.params)
