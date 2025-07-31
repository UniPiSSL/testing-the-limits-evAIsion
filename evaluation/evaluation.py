import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix
import torch.nn.functional as F
import torch

def evaluate_metrics(classifier, data, true_labels, adv_data=None, phase=""):
    preds = classifier.predict(data)
    pred_labels = np.argmax(preds, axis=1)
    true_labels = true_labels.numpy()

    acc = np.mean(pred_labels == true_labels)
    misc_rate = 1 - acc
    f1 = f1_score(true_labels, pred_labels, average="macro")
    prec = precision_score(true_labels, pred_labels, average='macro', zero_division=0)
    rec = recall_score(true_labels, pred_labels, average="macro")

    conf_scores = classifier.predict(data)
    if conf_scores.max() > 1:
        conf_scores = F.softmax(torch.tensor(conf_scores), dim=1).numpy()

    true_conf = conf_scores[np.arange(len(true_labels)), true_labels]
    avg_conf = np.mean(true_conf)

    return {
        "accuracy": acc,
        "f1_score": f1,
        "precision": prec,
        "recall": rec,
        "misclassification_rate": misc_rate,
        "mean_confidence": avg_conf,
    }

def display_metrics(before, after):
    print("\n--- Model Performance Metrics ---\n")
    print(f"{'Metric':<25} | {'Before Attack':<15} | {'After Attack':<15} | {'Impact':<15}")
    print("-" * 85)
    
    print(f"{'Accuracy':<25} | {before['accuracy']:<15.4f} | {after['accuracy']:<15.4f} | {after['accuracy'] - before['accuracy']:<15.4f}")
    print(f"{'F1 Score':<25} | {before['f1_score']:<15.4f} | {after['f1_score']:<15.4f} | {after['f1_score'] - before['f1_score']:<15.4f}")
    print(f"{'Precision':<25} | {before['precision']:<15.4f} | {after['precision']:<15.4f} | {after['precision'] - before['precision']:<15.4f}")
    print(f"{'Recall':<25} | {before['recall']:<15.4f} | {after['recall']:<15.4f} | {after['recall'] - before['recall']:<15.4f}")
    print(f"{'Misclassification Rate':<25} | {before['misclassification_rate']:<15.4f} | {after['misclassification_rate']:<15.4f} | {after['misclassification_rate'] - before['misclassification_rate']:<15.4f}")
    print(f"{'Mean Confidence':<25} | {before['mean_confidence']:<15.4f} | {after['mean_confidence']:<15.4f} | {after['mean_confidence'] - before['mean_confidence']:<15.4f}")
    print("-" * 85)