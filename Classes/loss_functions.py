import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, auc, precision_score, f1_score, recall_score, accuracy_score

def compute_pr_auc(y_true, predidictions):
    precision, recall, thresholds = precision_recall_curve(y_true, predidictions)
    return auc(recall, precision)

def compute_precision_top_K(y_true, predictions, K):
    top_predictions_indices = sorted(range(len(predictions)), key=lambda i: predictions[i], reverse=True)[:K]
    y_top_K = np.array([y_true[i] for i in top_predictions_indices])
    predictions_top_K = np.array([predictions[i] for i in top_predictions_indices])
    precision_top_K = precision_score(y_top_K, predictions_top_K.round())
    return precision_top_K


def compute_losses(y_test_monodimensional, predictions:dict):
    # metrics = ['F1', 'PR_AUC', 'PK_50', 'PK_100']
    metrics_results = {}
    for key in predictions.keys():
        metrics_results[key] = {}
        metrics_results[key]['F1'] = f1_score(y_test_monodimensional, predictions[key].round())
        metrics_results[key]['PR_AUC'] = compute_pr_auc(y_test_monodimensional, predictions[key])
        metrics_results[key]['PK_50'] = compute_precision_top_K(y_test_monodimensional, predictions[key], 50)
        metrics_results[key]['PK_100'] = compute_precision_top_K(y_test_monodimensional, predictions[key], 100)
        metrics_results[key]['Accuracy'] = accuracy_score(y_test_monodimensional, predictions[key].round())
        metrics_results[key]['Recall'] = recall_score(y_test_monodimensional, predictions[key].round())
    return pd.DataFrame(metrics_results)
