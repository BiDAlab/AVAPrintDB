from xml.parsers.expat import model

import torch
import os
import copy

from tqdm import tqdm
import csv
import numpy as np
import sys

from plotting import (
    create_all_plots
)

from utils import (
    get_dataset_class,
    )

def get_test_dataset(dataset_config, **kwargs):
    dataset_class_name = dataset_config.get("dataset_class", None)
    dataset_kwargs = dataset_config.get("dataset_class_kwargs", {})
    dataset_kwargs = dataset_kwargs if dataset_kwargs is not None else {}

    dataset = get_dataset_class(dataset_name=dataset_class_name, **dataset_kwargs, **kwargs)
    return dataset


def run_test(config_evaluation, model, device, output_dir_plots):
    
    # loading weights of the model:
    checkpoint_path = config_evaluation.get("checkpoint", None)
    if checkpoint_path is None:
        raise ValueError("No checkpoint path provided for evaluation. Please specify a valid path to the model chakpoint you want to evaluate")
    
    model, _, _ = load_checkpoint(model, None, checkpoint_path)
    device = torch.device(device) if not isinstance(device, torch.device) else device
    model = model.to(device)
    
    eer_threshold = config_evaluation.get("score_threshold", None)
    do_evaluation_step(config_evaluation, model, device, output_dir_plots, eer_threshold)


def do_evaluation_step(config_evaluation, model, device, output_dir_plots, eer_threshold=None, postfix=""):
    
    
    evaluation_datasets_csvs = config_evaluation["avatar_dataset"].get("dataset_class_kwargs", {}).get("evaluation_pairs_csv", [])
    for evaluation_dataset_csv in evaluation_datasets_csvs:
        
        c_postfix = postfix + "_" + os.path.splitext(os.path.basename(evaluation_dataset_csv))[0]
        current_config = copy.deepcopy(config_evaluation)
        current_config["avatar_dataset"]["dataset_class_kwargs"]["evaluation_pairs_csv"] = evaluation_dataset_csv
        test_dataset = get_test_dataset(current_config["avatar_dataset"], device=device)
        
        # test_sampler = DistributedSampler(test_dataset, num_replicas=world_size, rank=rank)
        test_dataloader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=current_config.get("batch_size", 32),
            num_workers=config_evaluation.get("num_workers", 4),
            collate_fn=test_dataset.collate_fn,
            shuffle=False,
            pin_memory=True,
            sampler=None,
        )
        
        # test_sampler.set_epoch(0)

        
        model_for_eval = model
        threshold = eer_threshold or current_config.get("score_threshold", None)
        metrics = evaluate_model(model_for_eval, test_dataloader, output_dir_plots, device, current_config.get("metric", "cosine"), threshold, c_postfix)
        


def evaluate_model(input_model, dataloader, output_dir, device="cuda", similarity_metric="cosine", SCORE_THRESHOLD=None, postfix=""):
    
    input_model.to(device)
    input_model.eval()
    model = input_model

    local_scores = []
    local_labels = []
    local_enrol_paths = []
    local_test_paths = []

    pbar = tqdm(dataloader, total=len(dataloader), desc="Evaluating data ...", file=sys.stdout)
    with torch.no_grad():
        for batch in pbar:
            enrol_ranges, test_ranges, batch_labels, batch_enrol_paths, batch_test_paths = batch

            # Construimos offsets + listas de views desde windows_tensor (CPU, barato)
            enrol_list, test_list = [], []
            enrol_offsets = [0]
            test_offsets  = [0]

            # IMPORTANTE: usa la referencia al tensor gigante del dataset
            WT = dataloader.dataset.windows_tensor  # [N_total_windows, F, D]

            for (e0, e1), (t0, t1) in zip(enrol_ranges.tolist(), test_ranges.tolist()):
                e = WT[e0:e1]   # view [Ne, F, D]
                t = WT[t0:t1]   # view [Nt, F, D]

                enrol_list.append(e)
                test_list.append(t)

                enrol_offsets.append(enrol_offsets[-1] + e.shape[0])
                test_offsets.append(test_offsets[-1] + t.shape[0])

            enrol_offsets = torch.tensor(enrol_offsets, device=device, dtype=torch.long)
            test_offsets  = torch.tensor(test_offsets,  device=device, dtype=torch.long)

            # Mover a GPU y concatenar EN GPU (reduce carga CPU)
            enrol_list = [x.to(device, non_blocking=True) for x in enrol_list]
            test_list  = [x.to(device, non_blocking=True) for x in test_list]

            enrol_videos = torch.cat(enrol_list, dim=0)  # [sumNe, F, D] en GPU
            test_videos  = torch.cat(test_list,  dim=0)  # [sumNt, F, D] en GPU

            enrol_embeds = model.predict(enrol_videos)
            test_embeds = model.predict(test_videos)

            enrol_embeds = torch.nn.functional.normalize(enrol_embeds, dim=-1)
            test_embeds  = torch.nn.functional.normalize(test_embeds, dim=-1)

            global_sim = enrol_embeds @ test_embeds.T

            batch_scores = []

            for i in range(len(enrol_offsets) - 1):

                e_start, e_end = enrol_offsets[i], enrol_offsets[i+1]
                t_start, t_end = test_offsets[i], test_offsets[i+1]

                sim_block = global_sim[e_start:e_end, t_start:t_end]
                score = sim_block.mean()

                batch_scores.append(score)

            batch_scores = torch.stack(batch_scores).detach().cpu().numpy()
            batch_labels = batch_labels.cpu().numpy().reshape(-1)

            batch_scores = [float(s) for s in batch_scores]
            batch_labels = [int(l) for l in batch_labels]

            local_scores.extend(batch_scores)
            local_labels.extend(batch_labels)
            local_enrol_paths.extend(batch_enrol_paths)
            local_test_paths.extend(batch_test_paths)

    scores = local_scores
    labels = local_labels
    enrol_paths_all = local_enrol_paths
    test_paths_all = local_test_paths

    csv_path = os.path.join(output_dir, f"scores_{similarity_metric}{postfix}.csv")
    with open(csv_path, mode='w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["enrolment_sample", "test_sample", "similarity_score", "label"])
        for enrol_path, test_path, score, label in zip(enrol_paths_all, test_paths_all, scores, labels):
            writer.writerow([enrol_path, test_path, score, label])

    scores = np.array(scores)
    labels = np.array(labels)

    metrics = create_all_plots(scores, labels, output_dir, similarity_metric, postfix)

    return metrics


def load_checkpoint(model, optimizer, path):
    
    checkpoint = torch.load(path, map_location='cpu')
    state_dict = checkpoint['model_state_dict']

    # Get current model's state_dict keys
    model_keys = model.state_dict().keys()
    first_model_key = list(model_keys)[0]
    first_ckpt_key = list(state_dict.keys())[0]

    # Determine if keys mismatch due to DDP wrapper
    if first_ckpt_key.startswith("module.") and not first_model_key.startswith("module."):
        state_dict = {k.replace("module.", "", 1): v for k, v in state_dict.items()}
    elif not first_ckpt_key.startswith("module.") and first_model_key.startswith("module."):
        state_dict = {f"module.{k}": v for k, v in state_dict.items()}

    # Load state_dict into model
    model.load_state_dict(state_dict, strict=True)

    # Load optimizer state (if applicable)
    if optimizer and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    epoch = checkpoint.get("epoch", None)
    
    return model, optimizer, epoch