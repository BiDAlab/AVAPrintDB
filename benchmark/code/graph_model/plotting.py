import os
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

def plot_hist(y_true, y_score, out_path):
    """Plot and save histograms of genuine vs. impostor scores.

    Args:
        y_true (array-like of shape (n_samples,)): 
            Ground truth binary labels. 
            1 = genuine pair, 0 = impostor pair.
        y_score (array-like of shape (n_samples,)): 
            Similarity or confidence scores corresponding to `y_true`.
        out_path (str): 
            Path where the histogram figure will be saved (PNG).

    Notes:
        - Uses 50 bins for each histogram.
        - Distributions are normalized (density=True).
    """
    fig = plt.figure(figsize=(10,8))

    plt.hist(y_score[y_true==1], bins=50, label="Genuine", alpha=0.7, density=True)
    plt.hist(y_score[y_true==0], bins=50, label="Impostor", alpha=0.7, density=True)
    
    plt.legend()
    plt.savefig(out_path)
    plt.close(fig)

def plot_roc(y_true, y_score, eer=None, eer_threshold=None, lbl=''):
    """Plot a ROC curve and return the AUC.

    Args:
        y_true (array-like of shape (n_samples,)): 
            Ground truth binary labels. 
            1 = genuine pair, 0 = impostor pair.
        y_score (array-like of shape (n_samples,)): 
            Predicted similarity or confidence scores.
        eer (float, optional): 
            Equal Error Rate to display in the legend. Defaults to None.
        eer_threshold (float, optional): 
            Threshold at which FAR == FRR. Shown in the legend if provided.
        lbl (str, optional): 
            Label for this curve (e.g., "Train", "Val", "Test"). 
            Defaults to an empty string.

    Returns:
        float: 
            Area Under the ROC Curve (AUC).

    Notes:
        - Appends ROC curve to the current matplotlib figure.
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, lw=2, label=f'{lbl}: AUC={roc_auc:.4f}, EER={eer:.4f} @ {eer_threshold:.3f} ')
    plt.plot([0,1],[0,1], linestyle='--', lw=1, color='gray')

    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc='lower right')
    plt.grid(True)


    return roc_auc

def generate_loss_curves(epoch:int, history:dict[str,list[float]], output_dir:str):
    """Generate and save training/validation/test loss curves.

    Args:
        epoch (int): 
            The final epoch number. Used as the x-axis length.
        history (dict[str, list[float]]): 
            Dictionary containing loss values per epoch. 
            Supported keys: "train", "val", "test".
        output_dir (str): 
            Directory where the curve image will be saved.
    """
    loss_curve_path = os.path.join(output_dir, "loss_curve.png")
    plt.figure(figsize=(8,5))
    if "train" in history:
        plt.plot(range(1, epoch+1), history['train'], marker='o', label='Train Loss')
    if "val" in history and history['val'] != []:
        plt.plot(range(1, epoch+1), history['val'], marker='o', label='Validation Loss')
    if "test" in history and history['test'] != []:
        plt.plot(range(1, epoch+1), history['test'], marker='o', label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Learning curves')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(loss_curve_path)
    plt.close()

    # save csv with loss values
    loss_csv_path = os.path.join(output_dir, "loss_values.csv")
    # history has keys: "train", "val", "test" and "elapsed_time" (which is a string)
    with open(loss_csv_path, 'w') as f:
        # write header
        f.write("epoch")
        for key in history:
            if key != "elapsed_time":
                f.write(f",{key}_loss")
        f.write(",elapsed_time")
        f.write("\n")
        # write values
        for ep in range(epoch):
            f.write(f"{ep+1}")
            for key in history:
                if key != "elapsed_time":
                    if history[key] != []:
                        f.write(f",{history[key][ep]:.6f}")
                    else:
                        f.write(",")
            f.write(f",{history['elapsed_time'][ep]}")
            f.write("\n")