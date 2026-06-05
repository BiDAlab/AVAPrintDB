"""Utility functions used throughout the code"""

import torch
from tqdm import tqdm
import numpy as np
import random
from sklearn.metrics import roc_curve
import torch.nn.functional as F


def extract_embeddings(model, loader, device, txt=""):
    """Extract similarity scores and labels from triplet embeddings.

    Runs the model over a DataLoader of triplets (anchor, positive, negative),
    computes cosine similarities, and assigns binary labels:
      - Anchor vs. Positive → label 1 (genuine).
      - Anchor vs. Negative → label 0 (impostor).

    Args:
        model (torch.nn.Module): Model implementing forward(anchor, positive, negative)
            and returning embeddings.
        loader (torch.utils.data.DataLoader): Yields batches of (anchors, positives, negatives).
        device (torch.device): Computation device ('cuda' or 'cpu').
        txt (str, optional): Label appended to tqdm progress bar description.

    Returns:
        tuple[np.ndarray, np.ndarray]:
            - labels: Array of shape (2 * N,) with 1s for genuine and 0s for impostor pairs.
            - sims: Array of shape (2 * N,) with cosine similarity scores.
    """
    model.eval()
    sims, labels = [], []
    with torch.no_grad():
        for a, p, n in tqdm(loader, desc=f"Extracting {txt} embeddings..."):
            a, p, n = a.to(device), p.to(device), n.to(device)
            emb_a, emb_p, emb_n, attn_a, attn_p, attn_n = model(a, p, n)
            sim_ap = F.cosine_similarity(emb_a, emb_p).cpu().numpy()
            sim_an = F.cosine_similarity(emb_a, emb_n).cpu().numpy()
            sims.extend(sim_ap); labels.extend([1]*len(sim_ap))
            sims.extend(sim_an); labels.extend([0]*len(sim_an))
    return np.array(labels), np.array(sims)

def compute_eer(y_true, y_score):
    """Compute Equal Error Rate (EER) and its threshold.

    The EER is the point where false acceptance rate (FPR) equals false
    rejection rate (FNR). It is commonly used in biometric verification.

    Args:
        y_true (array-like of shape (n_samples,)): Binary ground truth labels
            (1 = genuine, 0 = impostor).
        y_score (array-like of shape (n_samples,)): Predicted similarity scores.

    Returns:
        tuple[float, float]:
            - eer: Equal Error Rate value.
            - eer_threshold: Score threshold corresponding to EER.
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    fnr = 1 - tpr
    abs_diffs = np.abs(fpr - fnr)
    idx_eer = np.nanargmin(abs_diffs) # minima dif errores  (FPR = FNR)
    eer = fpr[idx_eer]
    eer_threshold = thresholds[idx_eer]
    return eer, eer_threshold


def set_seed(seed = None):
    """Set random seeds for reproducibility across libraries.

    Sets seeds for Python's `random`, NumPy, PyTorch CPU,
    and PyTorch CUDA RNGs

    Args:
        seed (int, optional): Seed value. Defaults to None.
    """
    if seed is None:
        seed = random.SystemRandom().randint(0, 2**32 - 1)
    print(f"Using SEED: {seed}")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    return seed

def seed_worker(worker_id):
    """Seed a DataLoader worker for reproducibility.

    This ensures that each worker process in a DataLoader has a different,
    but deterministic, random seed.

    Args:
        worker_id (int): Worker process ID provided by PyTorch DataLoader.
    """
    seed = torch.initial_seed() % 2**32
    random.seed(seed + worker_id)
    np.random.seed(seed + worker_id)