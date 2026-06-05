import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import plotly.io as pio
import plotly.graph_objects as go
from sklearn.manifold import TSNE
from sklearn.metrics import det_curve, roc_auc_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.preprocessing import LabelEncoder
import plotly.express as px


from sklearn.metrics import (
    roc_curve, roc_auc_score,
    precision_score, recall_score, f1_score, accuracy_score,
    confusion_matrix, balanced_accuracy_score
)

def generate_loss_curve_plot(history, output_dir, filename="loss_curve.png"):
    plt.figure(figsize=(10, 6))

    # Plot training loss
    if "train_loss" in history:
        train_loss = history["train_loss"]
        train_epochs = np.arange(1, len(train_loss) + 1)
        plt.plot(train_epochs, train_loss, label="Train Loss", marker='o')

    # Plot validation loss (if available)
    if "val_loss" in history and len(history["val_loss"]) > 0:
        val_loss = np.array(history["val_loss"])
        val_epochs_all = np.arange(1, len(val_loss) + 1)
        valid_mask = ~np.isnan(val_loss)
        val_loss_clean = val_loss[valid_mask]
        val_epochs_clean = val_epochs_all[valid_mask]
        plt.plot(val_epochs_clean, val_loss_clean, label="Validation Loss", marker='o')

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss Curve")
    plt.legend()
    plt.grid(True)

    # Save the figure
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, filename))
    plt.close()


def plot_roc_curve(fpr, tpr, eer_index, eer, eer_threshold, auc, output_path):
    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC = {auc:.4f}")
    plt.plot([0, 1], [0, 1], "k--")
    plt.scatter(fpr[eer_index], tpr[eer_index], color='red', label=f"EER = {eer:.4f}@{eer_threshold:.4f}")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve")
    plt.grid(True)
    plt.legend()
    plt.savefig(output_path)
    plt.close()

def plot_det_curve(labels, scores, det_plot_path):
    # Compute DET curve values
    fpr, fnr, thresholds = det_curve(labels, scores)
    
    # Find EER (where |FPR - FNR| is minimized)
    eer_index = np.nanargmin(np.absolute(fnr - fpr))
    eer = (fpr[eer_index] + fnr[eer_index]) / 2
    eer_threshold = thresholds[eer_index]
    
    # Plot DET curve
    plt.figure()
    plt.plot(fpr, fnr, label='DET Curve')
    
    # Plot EER point
    plt.scatter(fpr[eer_index], fnr[eer_index], color='red',
                label=f"EER = {eer:.4f} @ threshold = {eer_threshold:.4f}")
    
    plt.xlabel("False Positive Rate")
    plt.ylabel("False Negative Rate")
    plt.title("DET Curve with EER")
    plt.legend()
    plt.grid(True)
    plt.savefig(det_plot_path)
    plt.close()

def plot_score_histograms(y_scores, y_true, score_histograms_path, eer_threshold=None):
    genuine_scores = y_scores[y_true == 1]
    impostor_scores = y_scores[y_true == 0]

    auc = roc_auc_score(y_true, y_scores)

    plt.figure()
    plt.hist(genuine_scores, bins=50, alpha=0.6, label="Genuine", density=True)
    plt.hist(impostor_scores, bins=50, alpha=0.6, label="Impostor", density=True)
    if eer_threshold is not None:
        plt.axvline(eer_threshold, color='red', linestyle='--', label=f"EER Threshold: {eer_threshold:.4f}")
    plt.xlabel("Similarity Score")
    plt.ylabel("Density")
    plt.title(f"Score Distribution - AUC: {auc*100:.4f}")
    plt.legend()
    plt.grid(True)
    plt.savefig(score_histograms_path)
    plt.close()


def plot_confusion_matrix(cm, classes, save_path, normalize=True, title='Confusion Matrix'):

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1, keepdims=True)

    fig, ax = plt.subplots(figsize=(5, 5))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues, vmin=0, vmax=cm.max() if not normalize else None)
    ax.figure.colorbar(im, ax=ax)

    ax.set(
        xticks=np.arange(len(classes)),
        yticks=np.arange(len(classes)),
        xticklabels=classes,
        yticklabels=classes,
        ylabel='True label',
        xlabel='Predicted label',
        title=title
    )

    # Rotate tick labels and set alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    fmt = '.3f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")

    fig.tight_layout()
    plt.savefig(save_path)
    plt.close()


def create_all_plots(scores, labels, output_dir, similarity_metric="cosine", postfix=""):
    # ROC & AUC
    fpr, tpr, thresholds = roc_curve(labels, scores)
    auc = roc_auc_score(labels, scores)

    # EER
    fnr = 1 - tpr
    eer_index = np.nanargmin(np.abs(fnr - fpr))
    eer = (fpr[eer_index] + fnr[eer_index]) / 2
    eer_threshold = thresholds[eer_index]

    prediction_threshold = eer_threshold
    preds = (scores >= prediction_threshold).astype(int)

    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds)
    recall = recall_score(labels, preds)
    precision = precision_score(labels, preds)
    cm = confusion_matrix(labels, preds)
    bal_acc = balanced_accuracy_score(labels, preds)

    roc_plot_path = os.path.join(output_dir, f"roc_curve_{similarity_metric}{postfix}.png")
    plot_roc_curve(fpr, tpr, eer_index, eer, eer_threshold, auc, roc_plot_path)

    det_plot_path = os.path.join(output_dir, f"det_curve_{similarity_metric}{postfix}.png")
    plot_det_curve(labels, scores, det_plot_path)

    score_hist_path = os.path.join(output_dir, f"score_histogram_{similarity_metric}{postfix}.png")
    plot_score_histograms(scores, labels, score_histograms_path=score_hist_path, eer_threshold=eer_threshold)

    cm_path = os.path.join(output_dir, f"confusion_matrix_{similarity_metric}{postfix}.png")
    plot_confusion_matrix(cm, classes=["Impostor", "Genuine"], save_path=cm_path, normalize=False)

    return {
        "AUC": auc,
        "EER": eer,
        "EER_threshold": eer_threshold,
        "Accuracy": acc,
        "F1": f1,
        "Recall": recall,
        "Precision": precision,
        "Balanced accuracy": bal_acc
    }