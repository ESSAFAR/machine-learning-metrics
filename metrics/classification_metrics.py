from utils.dataset import generate_classification_test_dataset
import numpy as np
import pandas as pd

def accuracy(y_true, y_pred):
    return np.sum(y_true == y_pred) / len(y_true)

def recall(y_true, y_pred):
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    return tp / (tp + fn) if (tp + fn) > 0 else 0.0

def precision(y_true, y_pred):
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    return tp / (tp + fp) if (tp + fp) > 0 else 0.0

def f1_score(y_true, y_pred):
    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    return 2 * (p * r) / (p + r) if (p + r) > 0 else 0.0


def roc_auc_score(y_true, y_scores):
    sorted_indices = np.argsort(y_scores)
    y_true = y_true[sorted_indices]
    y_scores = y_scores[sorted_indices]
    pos = np.sum(y_true == 1)
    neg = len(y_true) - pos
    tpr = 0
    fpr = 0
    auc = 0
    for i in range(len(y_true)):
        if y_true[i] == 1:
            tpr += 1 / pos
        else:
            auc += tpr / neg
    return auc

def log_loss(y_true, y_pred_probs):
    epsilon = 1e-15
    y_pred_probs = np.clip(y_pred_probs, epsilon, 1 - epsilon)
    return -np.mean(y_true * np.log(y_pred_probs) + (1 - y_true) * np.log(1 - y_pred_probs))

if __name__ == "__main__":
    df = generate_classification_test_dataset()
    y_true = df['True_Label'].values
    y_pred = df['Predicted_Label'].values
    print("Accuracy:", accuracy(y_true, y_pred))
    print("Recall:", recall(y_true, y_pred))
    print("Precision:", precision(y_true, y_pred))
    print("F1 Score:", f1_score(y_true, y_pred))
    y_scores = df['Predicted_Probs'].values
    print("ROC-AUC Score:", roc_auc_score(y_true, y_scores))
    print("Log Loss:", log_loss(y_true, y_scores))


