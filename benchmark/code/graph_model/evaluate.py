"""
Evaluation script for GCN-based avatar verification.

This module evaluates a trained SpatialTemporalTripletModel on one or more
dataset splits, writing scores and plots (ROC and histograms) to disk.

Key outputs:
- scores.csv            # columns: labels (0/1), scores (similarities)
- ROC.png               # global ROC with EER marker (all splits)
- scores_histogram.png  # similarity histograms
"""
import os
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

from avatar_authentication.constants import AVAILABLE_DATASETS
from avatar_authentication.utils import set_seed, seed_worker, extract_embeddings, compute_eer
from avatar_authentication.plotting import plot_roc, plot_hist
from avatar_authentication.landmarks_dataset import TestingDataset, collate_fn, collate_k
from avatar_authentication.models import SpatialTemporalTripletModel


def _pick_collate(ds):
    return collate_k if getattr(ds, "triplets_per_anchor", 1) > 1 else collate_fn


def evaluate_on_all_dataset_splits(ckpt_path, train_ds, val_ds, test_ds, output_dir, batch_size, hidden_dim, embed_dim, device):
    """
    Evaluate a checkpoint across Train/Val/Test splits and generate plots/files. This function is used during training.

    This function:
      1) Loads the same checkpoint into a fresh model for each split.
      2) Extracts pairwise scores via `extract_embeddings`.
      3) Saves per-split scores to CSV.
      4) Computes EER + threshold, prints them, and updates a single ROC figure.
      5) After all splits, saves a combined ROC.png and per-split histograms.

    Parameters
    ----------
    ckpt_path : str
        Path to a PyTorch checkpoint. Accepts either a raw state_dict or a
        dict with key 'model_state_dict'.
    train_ds, val_ds, test_ds : torch.utils.data.Dataset
        Datasets for each split. Each must expose `edge_index`, `num_nodes`, and
        `D` (input feature dim), and be compatible with `collate_fn`.
    output_dir : str
        Directory where CSVs and figures are written. Created if missing.
    batch_size : int
        Batch size for scoring (DataLoader).
    hidden_dim : int
        Hidden dimension for the backbone GCN.
    embed_dim : int
        Output embedding dimension.
    device : torch.device
        Computation device ('cuda' or 'cpu').

    Outputs
    -------
    - {output_dir}/scores_<Train|Val|Test>.csv
      Columns: 'labels' (0 for impostor, 1 for genuine), 'scores' (similarity).
    - {output_dir}/ROC.png
      Combined ROC curves (one per split) with EER markers.
    - {output_dir}/scores_histogram_<Train|Val|Test>.png
      Score distribution histograms per split.
    """
    os.makedirs(output_dir, exist_ok=True)
    info = {}

    histograms_inputs = []
    plt.figure(figsize=(6,6))
    for ds, lbl in zip([train_ds, val_ds, test_ds], ['Train','Val', 'Test']):
        if ds is None:
            continue
        dataLoader = DataLoader(
            ds, batch_size=batch_size, shuffle=False,
            collate_fn=_pick_collate(ds), worker_init_fn=seed_worker, num_workers=0
        )
        model = SpatialTemporalTripletModel(
            edge_index = ds.edge_index,
            num_nodes  = ds.num_nodes,
            in_dim     = ds.D,
            hidden_dim = hidden_dim,
            embed_dim  = embed_dim
        ).to(device)

        # Load checkpoint
        state = torch.load(ckpt_path, map_location=device)
        if isinstance(state, dict) and 'model_state_dict' in state:
            model.load_state_dict(state['model_state_dict'])
        else:
            model.load_state_dict(state)
        print("Loaded model: ", ckpt_path)

        y_true, y_score = extract_embeddings(model, dataLoader, device, lbl)
        df = pd.DataFrame({"labels":y_true, "scores":y_score})
        df.to_csv(os.path.join(output_dir, f'scores_{lbl}.csv'), index=False)

        hist_path = os.path.join(output_dir, f"scores_histogram_{lbl}.png")

        
        histograms_inputs.append((y_true, y_score, hist_path))

        eer, eer_threshold = compute_eer(y_true, y_score)
        print(f"{lbl}:\tEER = {eer:.4f} @ {eer_threshold:.4f}")
        # Generate ROC curve
        auc=plot_roc(y_true, y_score, eer=eer, eer_threshold=eer_threshold, lbl=lbl)
        info[f"{lbl}_EER"] = eer
        info[f"{lbl}_EER_Threshold"] = eer_threshold
        info[f"{lbl}_AUC"] = auc

    plt.savefig(os.path.join(output_dir, 'ROC.png'))

    # generating score histograms for all splits
    for y_true, y_score, hist_path in histograms_inputs:
        plot_hist(y_true, y_score, hist_path)

    return info


def _embed_windows_in_chunks(model, windows, device, chunk_size=128, use_amp=True, keep_on_gpu=True):
    embs = []
    W = windows.shape[0]

    with torch.no_grad():
        for s in range(0, W, chunk_size):
            x = windows[s:s+chunk_size].to(device, non_blocking=True)

            if use_amp and device.type == "cuda":
                with torch.amp.autocast("cuda", dtype=torch.float16):
                    e, _ = model.predict(x)
            else:
                e, _ = model.predict(x)

            e = F.normalize(e, p=2, dim=1)

            if keep_on_gpu:
                embs.append(e)                 # stays on GPU
            else:
                embs.append(e.detach().cpu())  # old behavior

    return torch.cat(embs, dim=0)


def _pairwise_cosine_in_chunks(E_emb, T_emb, block_e=256, block_t=256):
    """
    E_emb: (E, d) on GPU
    T_emb: (T, d) on GPU
    returns: scores (E*T,) on GPU (flattened)
    """
    E, d = E_emb.shape
    T, _ = T_emb.shape

    rows = []
    with torch.no_grad():
        for i in range(0, E, block_e):
            e = E_emb[i:i+block_e]         # GPU
            row_blocks = []
            for j in range(0, T, block_t):
                t = T_emb[j:j+block_t]     # GPU
                row_blocks.append(e @ t.T) # GPU
            rows.append(torch.cat(row_blocks, dim=1))  # GPU (be, T)
    full = torch.cat(rows, dim=0)  # GPU (E, T)
    return full.reshape(-1)        # GPU (E*T,)



def _pad_ragged_embeddings(embs, lengths, device):
    """
    embs: (sumL, d) on GPU
    lengths: list[int] length B
    returns padded: (B, Lmax, d) on GPU and mask: (B, Lmax) bool on GPU
    """
    B = len(lengths)
    d = embs.shape[1]
    Lmax = max(lengths) if B > 0 else 0
    padded = torch.zeros((B, Lmax, d), device=device, dtype=embs.dtype)
    mask = torch.zeros((B, Lmax), device=device, dtype=torch.bool)

    offset = 0
    for b, L in enumerate(lengths):
        if L > 0:
            padded[b, :L] = embs[offset:offset+L]
            mask[b, :L] = True
            offset += L
    return padded, mask


def _aggregate_scores_batched(scores, e_mask, t_mask, mode="mean", topk=5, temperature=1.0):
    """
    scores: (B, Emax, Tmax) on GPU
    e_mask: (B, Emax) bool
    t_mask: (B, Tmax) bool

    returns: (B,) tensor on GPU
    """
    B, Emax, Tmax = scores.shape

    # valid pairs mask: (B, Emax, Tmax)
    valid = e_mask.unsqueeze(2) & t_mask.unsqueeze(1)

    if mode == "mean":
        # masked sum / count
        s = torch.where(valid, scores, torch.zeros_like(scores))
        denom = valid.sum(dim=(1, 2)).clamp_min(1)
        return s.sum(dim=(1, 2)) / denom

    if mode == "max":
        s = scores.masked_fill(~valid, float("-inf"))
        return s.amax(dim=(1, 2))

    if mode == "min":
        s = scores.masked_fill(~valid, float("inf"))
        return s.amin(dim=(1, 2))

    if mode == "topk_mean":
        # flatten
        flat = scores.view(B, -1)
        vflat = valid.view(B, -1)

        # set invalid to -inf so they don't get picked
        flat = flat.masked_fill(~vflat, float("-inf"))

        k = min(topk, flat.shape[1])
        topv = flat.topk(k, dim=1).values  # (B, k)

        # if a row has <k valid entries, topv may contain -inf; mask those out
        good = torch.isfinite(topv)
        denom = good.sum(dim=1).clamp_min(1)
        topv = torch.where(good, topv, torch.zeros_like(topv))
        return topv.sum(dim=1) / denom

    if mode == "logsumexp":
        x = scores * float(temperature)
        x = x.masked_fill(~valid, float("-inf"))
        # logsumexp over all valid entries
        lse = torch.logsumexp(x.view(B, -1), dim=1)
        return lse / float(temperature)

    raise ValueError(f"Unknown aggregation mode: {mode}")


def evaluate_on_test_dataset(model, dataloader, output_dir, device, **kwargs):
    print("\t\t-------------Evaluating on test dataset with arguments:", kwargs)
    window_embed_chunk = kwargs.get("window_embed_chunk", 1024)
    aggregation = kwargs.get("aggregation", "mean")
    topk = kwargs.get("topk", 5)
    use_amp = kwargs.get("use_amp", True)
    temperature = kwargs.get("temperature", 1.0)

    os.makedirs(output_dir, exist_ok=True)
    model.eval()

    all_scores = []
    all_labels = []
    all_pairidx = []
    all_enrol_paths = []
    all_test_paths  = []

    with torch.no_grad():
        for enrol_ws, test_ws, labels, idxs, enrol_paths, test_paths in tqdm(dataloader, desc="Computing test scores (batched windows)..."):
            # enrol_ws/test_ws: list length B, each element is (E_b, N, V, D) / (T_b, N, V, D) on CPU
            B = len(enrol_ws)
            # print(f"Batch size: {B}, window counts E: {[w.shape[0] for w in enrol_ws]}, T: {[w.shape[0] for w in test_ws]}")
            # print("Using aggregation mode:", aggregation)

            # Fast path only for aggregated scoring
            if aggregation is not None:
                
                # ---- 1) concatenate windows across batch (CPU) ----
                E_lens = [int(w.shape[0]) for w in enrol_ws]
                T_lens = [int(w.shape[0]) for w in test_ws]

                enrol_cat = torch.cat(enrol_ws, dim=0)  # (sumE, N, V, D) CPU
                test_cat  = torch.cat(test_ws,  dim=0)  # (sumT, N, V, D) CPU

                # ---- 2) embed once per side (GPU) ----
                E_emb_cat = _embed_windows_in_chunks(
                    model, enrol_cat, device,
                    chunk_size=window_embed_chunk,
                    use_amp=use_amp,
                    keep_on_gpu=True
                )  # (sumE, d) GPU

                T_emb_cat = _embed_windows_in_chunks(
                    model, test_cat, device,
                    chunk_size=window_embed_chunk,
                    use_amp=use_amp,
                    keep_on_gpu=True
                )  # (sumT, d) GPU

                # ---- 3) pad back to (B, Emax, d) and (B, Tmax, d) (GPU) ----
                E_pad, E_mask = _pad_ragged_embeddings(E_emb_cat, E_lens, device)  # (B, Emax, d), (B, Emax)
                T_pad, T_mask = _pad_ragged_embeddings(T_emb_cat, T_lens, device)  # (B, Tmax, d), (B, Tmax)

                # ---- 4) batched pairwise cosine similarities (GPU) ----
                # (B, Emax, d) x (B, d, Tmax) -> (B, Emax, Tmax)
                scores = torch.bmm(E_pad, T_pad.transpose(1, 2))

                # ---- 5) aggregate (GPU) -> (B,) ----
                s_batch = _aggregate_scores_batched(
                    scores, E_mask, T_mask,
                    mode=aggregation, topk=topk, temperature=temperature
                )  # (B,) GPU

                # ---- 6) move just B scalars to CPU ----
                s_cpu = s_batch.detach().cpu().numpy().astype(np.float32, copy=False)
                y_cpu = labels.detach().cpu().numpy().astype(np.int64, copy=False)
                i_cpu = idxs.detach().cpu().numpy().astype(np.int64, copy=False)

                all_scores.append(s_cpu)
                all_labels.append(y_cpu)
                all_pairidx.append(i_cpu)
                all_enrol_paths.extend(enrol_paths)
                all_test_paths.extend(test_paths)

                # cleanup big tensors
                del enrol_cat, test_cat, E_emb_cat, T_emb_cat, E_pad, E_mask, T_pad, T_mask, scores, s_batch

            else:
                # ---- "no aggregation": keep all E*T scores per pair ----
                # We'll do it batched on GPU and then move results to CPU.

                E0 = enrol_ws[0].shape[0]
                T0 = test_ws[0].shape[0]
                same_E = all(w.shape[0] == E0 for w in enrol_ws)
                same_T = all(w.shape[0] == T0 for w in test_ws)

                # print(f"Batch size {B}, same_E={same_E}, same_T={same_T}, E0={E0}, T0={T0}")

                if same_E and same_T:
                    # print("All samples in batch have same window counts, using fast batched scoring...")
                    # 1) stack into (B,E,N,V,D) and (B,T,N,V,D) on CPU
                    enrol_batch = torch.stack(enrol_ws, dim=0)  # CPU
                    test_batch  = torch.stack(test_ws,  dim=0)  # CPU

                    # 2) flatten windows for embedding: (B*E, N,V,D) and (B*T, N,V,D)
                    enrol_flat = enrol_batch.view(B * E0, *enrol_batch.shape[2:])
                    test_flat  = test_batch.view(B * T0, *test_batch.shape[2:])

                    # 3) embed once per side on GPU
                    E_emb_flat = _embed_windows_in_chunks(
                        model, enrol_flat, device,
                        chunk_size=window_embed_chunk,
                        use_amp=use_amp,
                        keep_on_gpu=True
                    )  # (B*E, d) GPU

                    T_emb_flat = _embed_windows_in_chunks(
                        model, test_flat, device,
                        chunk_size=window_embed_chunk,
                        use_amp=use_amp,
                        keep_on_gpu=True
                    )  # (B*T, d) GPU

                    # 4) reshape to (B,E,d) and (B,T,d)
                    d = E_emb_flat.shape[1]
                    E_emb = E_emb_flat.view(B, E0, d)
                    T_emb = T_emb_flat.view(B, T0, d)

                    # 5) batched cosine similarity matrix per pair: (B,E,T)
                    scores = torch.bmm(E_emb, T_emb.transpose(1, 2))  # GPU

                    # 6) move to CPU and store in your existing format
                    # Your original format stores each pair as a 1D array of length E*T.
                    scores_cpu = scores.detach().cpu().reshape(B, -1).numpy().astype(np.float32, copy=False)

                    labels_cpu = labels.detach().cpu().numpy().astype(np.int64, copy=False)
                    idxs_cpu   = idxs.detach().cpu().numpy().astype(np.int64, copy=False)

                    L = scores_cpu.shape[1]  # number of scores per pair (E*T)

                    # append per sample (still a tiny loop, but it’s only appending arrays, not GPU work)
                    for b in range(B):
                        all_scores.append(scores_cpu[b])  # shape (E*T,)
                        all_labels.append(np.full((L,), int(labels_cpu[b]), dtype=np.int64))
                        all_pairidx.append(np.full((L,), int(idxs_cpu[b]), dtype=np.int64))
                        all_enrol_paths.append(np.full((L,), enrol_paths[b], dtype=object))
                        all_test_paths.append(np.full((L,),  test_paths[b],  dtype=object))

                    del enrol_batch, test_batch, enrol_flat, test_flat, E_emb_flat, T_emb_flat, E_emb, T_emb, scores

                else:
                    # print("Variable window counts per sample in batch, falling back to slower per-sample processing...")
                    
                    # Ragged fallback (your old per-pair code)
                    for b in range(B):
                        enrol_w = enrol_ws[b]
                        test_w  = test_ws[b]
                        label   = int(labels[b].item())
                        pair_id = int(idxs[b].item())

                        E_emb = _embed_windows_in_chunks(model, enrol_w, device, chunk_size=window_embed_chunk, use_amp=use_amp, keep_on_gpu=True)
                        T_emb = _embed_windows_in_chunks(model, test_w,  device, chunk_size=window_embed_chunk, use_amp=use_amp, keep_on_gpu=True)

                        scores_flat = _pairwise_cosine_in_chunks(E_emb, T_emb, block_e=1024, block_t=1024)
                        scores_cpu = scores_flat.float().detach().cpu().numpy()

                        L = scores_cpu.shape[0]  # number of scores per pair (E*T)

                        all_scores.append(scores_cpu.astype(np.float32, copy=False))
                        all_labels.append(np.full((L,), label, dtype=np.int64))
                        all_pairidx.append(np.full((L,), pair_id, dtype=np.int64))
                        all_enrol_paths.append(np.full((L,), enrol_paths[b], dtype=object))
                        all_test_paths.append(np.full((L,),  test_paths[b],  dtype=object))

                        del E_emb, T_emb, scores_flat


    # Concatenate across all batches
    y_score = np.concatenate(all_scores, axis=0).astype(np.float32, copy=False)
    y_true  = np.concatenate(all_labels, axis=0).astype(np.int64, copy=False)
    y_pair  = np.concatenate(all_pairidx, axis=0).astype(np.int64, copy=False)
    if aggregation is not None:
        # list[str] already aligned with y_score/y_true/y_pair
        y_enrol_path = np.array(all_enrol_paths, dtype=object)
        y_test_path  = np.array(all_test_paths,  dtype=object)
    else:
        # list[np.ndarray] -> concatenate
        y_enrol_path = np.concatenate(all_enrol_paths, axis=0)
        y_test_path  = np.concatenate(all_test_paths,  axis=0)

    print("Saving scores to CSV file...")
    df = pd.DataFrame({"labels":y_true, "scores": y_score, "pair_id": y_pair, "enrol_path": y_enrol_path, "test_path": y_test_path,})
    df.to_csv(os.path.join(output_dir, "scores.csv"), index=False)

    print(f"Generating plots and saving them in {output_dir} ...")
    # Plot scores histogram
    hist_path = os.path.join(output_dir, "scores_histogram.png")
    plot_hist(y_true, y_score, hist_path)

    # Compute ROC
    eer, eer_threshold = compute_eer(y_true, y_score)
    plt.figure(figsize=(6,6))
    auc = plot_roc(y_true, y_score, eer=eer, eer_threshold=eer_threshold, lbl='')
    plt.savefig(os.path.join(output_dir, 'ROC.png'))
    return eer, eer_threshold, auc


def main_evaluation(args):
    """
    Orchestrate evaluation from CLI args: data/model loading and test scoring.

    Steps
    -----
    1) Set random seeds (optional).
    2) Prepare output directory and save the used arguments.
    3) Load Delaunay edges and build `edge_index`.
    4) Create the TestingDataset and DataLoader.
    5) Instantiate SpatialTemporalTripletModel and load weights.
    6) Evaluate on the test dataset; print EER, threshold, and AUC.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed CLI arguments. See the `__main__` block for all options.
    """

    print(f"Running evaluation with args: {args}")

    assert args.checkpoint is not None, "You need to indicate the model checkpoint to use for evaluation."

    # Setting random seed
    SEED = args.seed
    g = torch.Generator()
    if SEED:
        set_seed(SEED)
        g.manual_seed(SEED)

    # Setting up output directory
    output_dir = os.path.join(args.output_dir_eval, args.experiment)
    os.makedirs(output_dir, exist_ok=True)

    # Saving arguments used
    with open(os.path.join(output_dir, "arguments_used.txt"), mode="w+") as f:
        f.write(str(vars(args)))
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Loading delaunay edges
    df_edges = pd.read_csv(args.edges_csv)
    edges = list(zip(df_edges["i"], df_edges["j"])) + list(zip(df_edges["j"], df_edges["i"]))
    edge_index = torch.tensor(edges, dtype=torch.long).T.contiguous()

    # Dimensions of preprocessed input data (numpy arrays) V=number of landmarks, D=dimension of each landmark
    V, D = 109, 3

    # Loading test dataset
    dataset = TestingDataset(
        root_dir=args.eval_root_data,
        dataset=args.eval_dataset,
        data_filter_csv=args.csv_data_file,
        num_frames=args.num_frames,
        frame_sampler=args.frame_sampler,
        pad_if_short=args.pad_if_short,
        generators=args.generators,
        slide_window_stride=args.window_stride,
    )
    dataLoader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False,
        collate_fn=dataset.collate_fn, worker_init_fn=seed_worker, num_workers=0, pin_memory=True,
    )

    # Creating model
    model = SpatialTemporalTripletModel(
        edge_index= edge_index,
        num_nodes=  V,
        in_dim=     D,
        hidden_dim=args.hidden_dim,
        embed_dim= args.embedding_dim
    ).to(device)

    # Loading model weights
    ckpt_path = args.checkpoint
    state = torch.load(ckpt_path, map_location=device)
    if isinstance(state, dict) and 'model_state_dict' in state:
        model.load_state_dict(state['model_state_dict'])
    else:
        model.load_state_dict(state)
    print("Loaded weights from file:", ckpt_path)

    eer, eer_threshold, auc = evaluate_on_test_dataset(model, dataLoader, output_dir, device, aggregation="mean")
    print(f"{args.experiment}:\tEER = {eer:.4f} @ {eer_threshold:.4f}, \tAUC = {auc:.4f}, args: {args}")




if __name__ == "__main__":
    # ----------------------------- CLI interface -----------------------------
    parser = argparse.ArgumentParser(description="GCN for Avatar verification")
    parser.add_argument(
        "--experiment",
        type=str,
        default="UndefinedExperiment",
        help="Unique identifier to be used for the output directory where evaluation results will be generated",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        "--output-dir-eval",
        dest="output_dir_eval",
        type=str,
        default="./outputs",
        help="Directory where outputs will be generated"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--root-data",
        "--eval-root-data",
        dest="eval_root_data",
        default="./GAGAvatar-Benchmark-db/preprocessed_data/gagavatar_ravdess_and_cremad/test_landmarks_cut_50_50",
        help="Directory where preprocessed data test_landmarks_cut_50_50 is located. This is the test data used to compute evaluation results"
    )
    parser.add_argument(
        "--dataset",
        "--eval-dataset",
        dest="eval_dataset",
        type=str,
        default="CREMAD",
        choices=AVAILABLE_DATASETS,
        help="Dataset used for evaluation",
    )
    parser.add_argument(
        "--generators",
        nargs="+", 
        default=["GAGA"], 
        help="List of generators to include in the evaluation. Each generator corresponds to a different version of the same avatar video, generated with a specific method. The generator is identified in the filename by a suffix '--GEN.mp4' (e.g., '--GAGA.mp4', '--LIVE.mp4', etc). By default, only videos with the '--GAGA.mp4' suffix are included. If you want to include multiple generators, list them all (e.g., --generators GAGA LIVE HUNY). Note: the script will look for files with the specified generator suffixes in the provided data folder and/or csv file, so make sure they are present."
    )
    parser.add_argument(
        "--edges-csv",
        default="./GAGAvatar-Benchmark-db/delaunay_edges.csv",
        help="Path to csv with delaunay edges"
    )
    parser.add_argument(
        "--csv-data-file",
        type=str,
        default=None,
        help="Path to csv that contains all the avatar files that should be used for evaluation. If None, all files in selected folder are used. If provided, the csv must have a column 'avatar_video_path' with the absolute paths to the avatar videos to use. Any other videos found in the selected folder but not listed in the csv will be ignored.", # TODO change description
    )
    parser.add_argument(
        "--batch-size",
        "-bs",
        type=int,
        default=1024,
        help="Batch size for evaluation"
    )
    parser.add_argument(
        "--hidden-dim",
        type=int,
        default=64,
        help="Hidden dimension for GCN"

    )
    parser.add_argument(
        "--embedding-dim",
        type=int,
        default=256,
        help="Output embedding dimension"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="GAGAvatar-Benchmark-code/checkpoints/pretrained_ravdess_and_cremad.pt",
        help="Path to checkpoint file used to load the model and evaluate"
    )
    parser.add_argument(
        "--num-frames",
        type=int,
        default=50,
        help="Number of consecutive frames to use from each video"
    )
    parser.add_argument(
        "--window-stride", 
        type=int, 
        default=10, 
        help="Stride of the sliding window (in frames)." )
    parser.add_argument(
        "--frame-sampler",
        type=str,
        default="sliding",
        choices=["first", "random", "sliding"],
        help="Strategy to pick consecutive frames: 'first' takes the first num_frames frames; 'random' takes num_frames starting at a random valid index; 'sliding' uses a sliding window with stride window-stride"
    )
    parser.add_argument(
        "--pad-if-short",
        action="store_true",
        help="If active, and the video has less frames than num_frames, it pads by repeating the last frame"
    )
    args = parser.parse_args()

    main_evaluation(args)
