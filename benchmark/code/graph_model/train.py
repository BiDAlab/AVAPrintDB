import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import datetime as dt
from tqdm import tqdm
import argparse

from avatar_authentication.constants import CREMAD_DATASET, RAVDESS_DATASET, CREMAD_AND_RAVDESS_DATASETS, AVAILABLE_DATASETS
from avatar_authentication.constants import VAL_IDS_CREMAD, VAL_IDS_RAVDESS, ALL_VALIDATION_IDS
from avatar_authentication.utils import set_seed, seed_worker
from avatar_authentication.landmarks_dataset import TripletGraphDataset, collate_fn, collate_k
from avatar_authentication.models import SpatialTemporalTripletModel
from avatar_authentication.plotting import generate_loss_curves
from avatar_authentication.evaluate import evaluate_on_all_dataset_splits

torch.backends.cudnn.benchmark = False


def main_train(args):
    """Train a SpatialTemporalTripletModel on the GAGAvatar benchmark dataset.

    This function sets up data, model, optimizer, and training loop for
    triplet-loss-based avatar verification. It supports original vs.
    avatarized videos, different dataset splits (CREMA, RAVDESS, or both),
    and optional evaluation on validation/test splits during training.

    Workflow:
        1. Set random seeds for reproducibility.
        2. Build output directories and save arguments.
        3. Prepare train/val/test datasets and DataLoaders.
        4. Initialize the SpatialTemporalTripletModel, optimizer, and LR scheduler.
        5. Train for `n_epochs` with triplet margin loss
        6. Save checkpoints every epoch and track the best model by
           validation loss.
        7. If `--evaluate` is active, run evaluation (EER, ROC, histograms)
           whenever a new best model is found.

    Args:
        args (argparse.Namespace): Command-line arguments. See the `__main__` block for all options.

    Outputs:
        - Checkpoints saved in `<output_dir>/checkpoints/`:
          * Per-epoch checkpoint with losses in filename.
          * `best_model.pt` always containing the latest best model.
        - Loss curves: `<output_dir>/loss_curve.png` updated each epoch.
        - If `--evaluate`: per-epoch evaluation outputs (CSV scores,
          histograms, ROC plots) in subdirectories of `<output_dir>`.
    """

    print(f"Running training with args: {args}")

    # Setting random seed
    SEED = args.seed
    set_seed(SEED)

    # Selecting training dataset
    dataset = args.dataset.upper()

    # Setting up output directories
    output_dir = args.output_dir_train
    timestamp = dt.datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = os.path.join(output_dir, f"run_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    ckpt_dir = os.path.join(output_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    # Saving arguments used
    with open(os.path.join(output_dir, "arguments_used.txt"), mode="w+") as f:
        f.write(str(vars(args)))
    
    # Preparing datasets and dataloaders
    print("Preparing datasets and dataloaders...")
    train_root = args.dev_root_data
    test_root  = args.test_root_data
    edges_csv_path   = args.edges_csv
    data_filter_csv = args.csv_data_file

    if dataset == CREMAD_DATASET:
        val_list = VAL_IDS_CREMAD
    elif dataset ==RAVDESS_DATASET:
        val_list = VAL_IDS_RAVDESS
    elif dataset == CREMAD_AND_RAVDESS_DATASETS:
        val_list = ALL_VALIDATION_IDS
    else:
        raise Exception(f"Dataset chosen:{dataset} is not one of the valid ones")

    
    val_dataset   = None
    if args.validate:
        train_dataset = TripletGraphDataset(train_root, dataset, data_filter_csv, edges_csv_path, triplets_per_anchor=args.triplets_per_anchor, id_list=val_list, num_frames=args.num_frames, frame_sampler=args.frame_sampler, pad_if_short=args.pad_if_short)
        val_dataset = TripletGraphDataset(train_root, dataset, data_filter_csv, edges_csv_path, triplets_per_anchor=args.triplets_per_anchor, id_list=val_list, validation=True, num_frames=args.num_frames, frame_sampler=args.frame_sampler, pad_if_short=args.pad_if_short)
    else:
        train_dataset = TripletGraphDataset(train_root, dataset, data_filter_csv, edges_csv_path, triplets_per_anchor=args.triplets_per_anchor, id_list=args.id_list, validation=True, num_frames=args.num_frames, frame_sampler=args.frame_sampler, pad_if_short=args.pad_if_short)  # even though validation=True, since we are not actually doing validation, this just means that the dataset will use the selected IDs for training, but it will not create a separate val_dataset and will not be used for validation during training.
    test_dataset  = TripletGraphDataset(test_root, dataset,data_filter_csv, edges_csv_path, triplets_per_anchor=1, num_frames=args.num_frames, frame_sampler=args.frame_sampler, pad_if_short=args.pad_if_short)

    g = torch.Generator()
    g.manual_seed(SEED)

    batch_size = args.batch_size

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_collate = collate_k if args.triplets_per_anchor > 1 else collate_fn

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        collate_fn=train_collate, worker_init_fn=seed_worker,
        generator=g, num_workers=4
    )
    if args.validate:
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False,
            collate_fn=train_collate, worker_init_fn=seed_worker,
            num_workers=4
        )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        collate_fn=collate_fn, worker_init_fn=seed_worker,
        num_workers=4
    )

    # Initializing model
    print("Initializing model...")
    model = SpatialTemporalTripletModel(
        edge_index= train_dataset.edge_index,
        num_nodes=  train_dataset.num_nodes,
        in_dim=     train_dataset.D,
        hidden_dim=args.hidden_dim,
        embed_dim= args.embedding_dim
    ).to(device)

    # Initializing optimizer with scheduler
    lr          = args.lr
    optimizer   = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)

    def lr_lambda(epoch):
        if epoch < 100: #50:  # FIXME
            return 1.0       # lr = 1e-4
        elif epoch < 200: #100:   # FIXME
            return 0.1       # lr = 1e-5
        else:
            return 0.1       # lr = 1e-5

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    # Triplet loss
    criterion = nn.TripletMarginLoss(margin=args.margin)


    # Training loop
    print("================= TRAINING LOOP =================")
    epochs     = args.n_epochs
    patience = args.patience
    min_delta = getattr(args, "min_delta", 0.0)  # safe if you decide not to add the arg
    epochs_no_improve = 0

    best_val_loss = avg_val_loss = best_train_loss = float('inf')
    train_losses, val_losses, test_losses = [], [], []
    avg_train_loss = avg_test_loss = 0.0
    pbar = tqdm(range(1, epochs+1), desc="Training")
    history = {"train":[], "val":[], "test":[], "elapsed_time":[]}
    best_results = None
    
    for epoch in pbar:
        # Train
        model.train()
        total_loss = 0.0
        total_count=0
        for anchors, positives, negatives in tqdm(train_loader, desc=f"Train Epoch {epoch}/{epochs}"):
            anchors, positives, negatives = anchors.to(device), positives.to(device), negatives.to(device)
            emb_a, emb_p, emb_n, attn_a, attn_p, attn_n = model(anchors, positives, negatives)
            loss = criterion(emb_a, emb_p, emb_n)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * anchors.size(0)
            total_count += anchors.size(0)
        avg_train_loss = total_loss / total_count
        train_losses.append(avg_train_loss)
        # print(f"Epoch {epoch}/{epochs} — Train Loss: {avg_train_loss:.4f}")
        pbar.set_postfix({"train_loss": avg_train_loss, "val_loss": avg_val_loss, "test_loss": avg_test_loss, "LR": scheduler.get_last_lr()[0]})
        history['train'].append(avg_train_loss)

        # Validation
        if args.validate:
            model.eval()
            total_val_loss = 0.0
            total_count=0
            with torch.no_grad():
                for anchors, positives, negatives in tqdm(val_loader, desc=f"Validation Epoch {epoch}/{epochs}"):
                    anchors, positives, negatives = anchors.to(device), positives.to(device), negatives.to(device)
                    emb_a, emb_p, emb_n, attn_a, attn_p, attn_n = model(anchors, positives, negatives)
                    loss = criterion(emb_a, emb_p, emb_n)
                    total_val_loss += loss.item() * anchors.size(0)
                    total_count += anchors.size(0)
            avg_val_loss = total_val_loss / total_count
            val_losses.append(avg_val_loss)
            # print(f"Epoch {epoch}/{epochs} — Validation  Loss: {avg_val_loss:.4f}")
            pbar.set_postfix({"train_loss": avg_train_loss, "val_loss": avg_val_loss, "test_loss": avg_test_loss, "LR": scheduler.get_last_lr()[0]})
            history['val'].append(avg_val_loss)

        scheduler.step()

        # Evaluate on test set (just for visualization)
        model.eval()
        total_test_loss = 0.0
        total_count=0
        with torch.no_grad():
            for anchors, positives, negatives in tqdm(test_loader, desc=f"Test Epoch {epoch}/{epochs}"):
                anchors, positives, negatives = anchors.to(device), positives.to(device), negatives.to(device)
                emb_a, emb_p, emb_n, attn_a, attn_p, attn_n = model(anchors, positives, negatives)
                loss = criterion(emb_a, emb_p, emb_n)
                total_test_loss += loss.item() * anchors.size(0)
                total_count += anchors.size(0)
        avg_test_loss = total_test_loss / total_count
        test_losses.append(avg_test_loss)
        # print(f"Epoch {epoch}/{epochs} — Test  Loss: {avg_test_loss:.4f}")
        pbar.set_postfix({"train_loss": avg_train_loss, "val_loss": avg_val_loss, "test_loss": avg_test_loss, "LR": scheduler.get_last_lr()[0]})
        history['test'].append(avg_test_loss)

        history['elapsed_time'].append(dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

        # Generate loss curves
        generate_loss_curves(epoch, history, output_dir)
        
        # Save checkpoint if best validation loss
        train_str = f"{avg_train_loss:.4f}"
        val_str  = f"{avg_val_loss:.4f}"


        # ---------------- Early stopping + best checkpoint logic ----------------
        # Decide what metric we monitor:
        if args.validate:
            current_metric = avg_val_loss
            best_metric = best_val_loss
        else:
            current_metric = avg_train_loss
            best_metric = best_train_loss

        improved = (best_metric - current_metric) > min_delta

        if improved:
            epochs_no_improve = 0

            # Update best metric(s)
            if args.validate:
                best_val_loss = avg_val_loss
            best_train_loss = avg_train_loss

            # Save best checkpoint
            train_str = f"{avg_train_loss:.4f}"
            val_str = f"{avg_val_loss:.4f}"

            best_path = os.path.join(
                ckpt_dir,
                f"best_epoch_{epoch:03d}_trainLoss_{train_str}_valLoss_{val_str}.pt"
            )
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
            }, best_path)

            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
            }, os.path.join(ckpt_dir, "best_model.pt"))  # overwrites previous best model

            print(f"\tNew best model, stored in:{best_path}")

            if args.evaluate:
                best_results = evaluate_on_all_dataset_splits(
                    best_path, train_dataset, val_dataset, test_dataset,
                    os.path.join(output_dir, f"evaluation_epoch_{epoch}"),
                    batch_size, args.hidden_dim, args.embedding_dim, device
                )
                best_results["best_epoch"] = epoch

        else:
            epochs_no_improve += 1
            print(f"\tNo improvement for {epochs_no_improve}/{patience} epoch(s).")

            if patience > 0 and epochs_no_improve >= patience:
                print(f"Early stopping triggered at epoch {epoch}: "
                    f"no improvement in monitored loss for {patience} consecutive epochs.")
                break
    
    # Evaluate final model on all splits if not done already
    if args.evaluate and best_results is None:
        best_path = os.path.join(ckpt_dir, "best_model.pt")
        best_results =evaluate_on_all_dataset_splits(best_path, train_dataset, val_dataset, test_dataset, os.path.join(output_dir, f"evaluation_final_epoch"), batch_size, args.hidden_dim, args.embedding_dim, device)
        best_results["best_epoch"] = epoch

    
    # write best results to a common csv, inlcuding all input arguments in parser as separate columns:
    trainings_csv_path = os.path.join(os.path.dirname(os.path.dirname(output_dir)),"trainings_summary.csv")
    with open(trainings_csv_path, mode="a") as f:
        # write header if file is empty
        if os.stat(trainings_csv_path).st_size == 0:
            f.write("dir,timestamp,dataset,triplets_per_anchor,batch_size,n_epochs,lr,hidden_dim,embedding_dim,margin,validate,evaluate,num_frames,frame_sampler,pad_if_short,seed,best_train_loss,best_val_loss,best_epoch")
            for split in ["Train", "Val", "Test"]:
                f.write(f",{split}_EER,{split}_EER_Threshold,{split}_AUC")
            f.write("\n")
        # write values
        f.write(f"{output_dir},{timestamp},{args.dataset},{args.triplets_per_anchor},{args.batch_size},{args.n_epochs},{args.lr},{args.hidden_dim},{args.embedding_dim},{args.margin},{args.validate},{args.evaluate},{args.num_frames},{args.frame_sampler},{args.pad_if_short},{args.seed},{train_str},{val_str}")
        if best_results is not None:
            f.write(f",{best_results['best_epoch']}")
            for split in ["Train", "Val", "Test"]:
                if split+"_EER" in best_results:
                    f.write(f",{best_results[split+'_EER']},{best_results[split+'_EER_Threshold']},{best_results[split+'_AUC']}")
        else:
            for split in ["Train", "Val", "Test"]:
                f.write(",,,")
        f.write("\n")
    
    return os.path.join(ckpt_dir, "best_model.pt")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="GAGAvatar-Benchmark, for biometric Avatar verification")
    parser.add_argument(
        "--dataset",
        type=str,
        default="CREMAD_AND_RAVDESS",
        choices=AVAILABLE_DATASETS,
        help="Dataset used for training",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        "--output-dir-train",
        dest="output_dir_train",
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
        "--dev-root-data",
        default="./109LANDMARKS/DEV",
        help="Directory where landmark files for training/validation are located"
    )
    parser.add_argument(
        "--test-root-data",
        default="./109LANDMARKS/TEST",
        help="Directory where landmark files for testing are located"
    )
    parser.add_argument(
        "--csv-data-file",
        type=str,
        default=None,
        help="Path to csv that contains all the avatar files that should be used for training/evaluation. If None, all files in dev/test folders are used. If provided, the csv must have a column 'avatar_video_path' with the absolute paths to the avatar videos to use. Any other videos found in the dev/test folders but not listed in the csv will be ignored.",
    )
    parser.add_argument(
        "--triplets-per-anchor", 
        type=int, 
        default=1, 
        help="Number of triplets to generate per anchor sample",
    )
    parser.add_argument(
        "--edges-csv",
        default="./GAGAvatar-Benchmark-db/delaunay_edges.csv",
        help="Path to csv with delaunay edges"
    )
    parser.add_argument(
        "--batch-size",
        "-bs",
        type=int,
        default=128,
        help="Batch size for training"
    )
    parser.add_argument(
        "--n-epochs",
        type=int,
        default=500,
        help="Number of epochs to train"
    )
    parser.add_argument(
        "-lr",
        type=float,
        default=1e-4,
        help="Initial learning rate"
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
        "--margin",
        type=float,
        default=1.0,
        help="Margin for triplet loss"
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="If active, it performs validation at each epoch"
    )
    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="If active, it generates evaluation scores and plots for the 3 splits (train, val, test) each epoch with a lower validation loss"
    )
    parser.add_argument(
        "--num-frames",
        type=int,
        default=50,
        help="Number of consecutive frames to use from each video"
    )
    parser.add_argument(
        "--frame-sampler",
        type=str,
        default="random",
        choices=["first", "random"],
        help="Strategy to pick consecutive frames: 'first' takes the first num_frames frames; 'random' takes num_frames starting at a random valid index"
    )
    parser.add_argument(
        "--pad-if-short",
        action="store_true",
        help="If active, and the video has less frames than num_frames, it pads by repeating the last frame"
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=20,
        help="Early stopping patience (epochs without improvement before stopping). "
            "If --validate is set, monitors validation loss; otherwise monitors train loss."
    )
    parser.add_argument(
        "--min-delta",
        type=float,
        default=0.0,
        help="Minimum decrease in the monitored loss to qualify as an improvement."
    )
    parser.add_argument(
        "--id-list",
        nargs="+",
        default=[],
        help="List of IDs to use for training. If empty, all available IDs are used. This is only considered if --validate is not set, since if --validate is active, the validation IDs defined in constants.py are used for training/validation split."
    )

    args = parser.parse_args()

    main_train(args)
        


        

        