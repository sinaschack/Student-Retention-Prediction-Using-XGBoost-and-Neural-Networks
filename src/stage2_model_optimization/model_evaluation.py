import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, roc_auc_score, classification_report

def evaluate_model(model, X_test, y_test, model_name='Model'):
    """
    Evaluate a trained model with metrics, confusion matrix, and ROC.
    """
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:,1] if hasattr(model, "predict_proba") else y_pred
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    print(f"\n{model_name} Confusion Matrix:\n", cm)
    
    # Classification Report
    print(f"\n{model_name} Classification Report:\n", classification_report(y_test, y_pred, digits=4))
    
    # ROC and AUC
    auc = roc_auc_score(y_test, y_pred_proba)
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
    
    plt.figure(figsize=(6,4))
    plt.plot(fpr, tpr, label=f"AUC = {auc:.4f}")
    plt.plot([0,1], [0,1], linestyle='--', color='gray')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"{model_name} ROC Curve")
    plt.legend()
    plt.grid(True)
    plt.show()
    
    return {'confusion_matrix': cm, 'auc': auc}
