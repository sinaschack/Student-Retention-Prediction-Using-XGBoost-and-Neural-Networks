from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

def evaluate_model(y_true, y_pred, y_prob=None, title="Confusion Matrix"):
    print("Accuracy:", accuracy_score(y_true, y_pred))
    print("Classification Report:\n", classification_report(y_true, y_pred))
    if y_prob is not None:
        print("AUC:", roc_auc_score(y_true, y_prob))
    disp = ConfusionMatrixDisplay(confusion_matrix(y_true, y_pred))
    disp.plot(cmap=plt.cm.Blues)
    plt.title(title)
    plt.show()
