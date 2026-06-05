import torch
import os
import copy
from logging import getLogger
from tqdm import tqdm
import csv
import numpy as np
import sys

from plotting import (
    generate_loss_curve_plot,
    create_all_plots
)

from utils import (
    get_dataset_class,
    get_optimizer
    )

from evaluate import (
    get_test_dataset,
    load_checkpoint,
    evaluate_model,
    do_evaluation_step
    )

def get_dataset(dataset_config, split="DEV", **kwargs):
    dataset_class_name = dataset_config.get("dataset_class", None)
    dataset_kwargs = dataset_config.get("dataset_class_kwargs", {})
    dataset_kwargs = dataset_kwargs if dataset_kwargs is not None else {}

    dataset = get_dataset_class(
        dataset_class_name,
        root_dir=dataset_config.get("root_dir", None),
        datasets=dataset_config.get("datasets", None),
        split=split,
        generators=dataset_config.get("generators", None),
        extension=".pt" if dataset_config.get("cache", False) else ".mp4",
        **dataset_kwargs,
        **kwargs,
    )
    return dataset

def run_training_eval(config, model, device, output_dir_run, output_dir_plots):
    logger = getLogger("run_training_eval")

    # Getting configurations
    dataset_config = config.get("avatar_dataset")
    config_training = config.get("training", {})
    config_optimizer = config_training["optimizer"]
    config_validation = config.get("validation", {})
    config_traindataExp = config.get("traindataExp", {})
    config_evaluation = config.get("evaluation", {})
    SEED = config.get("seed", None)

    logger.debug(f"Creating dataset with config {dataset_config}")
    dev_dataset = get_dataset(dataset_config)
    logger.info("Dev dataset generated")
    if config_validation.get("do", False):
        logger.info(f"Generating train/validation split with validation size {config_validation.get('val_size', 0.2)}")
        train_dataset, val_dataset = dev_dataset.get_train_val_dataset_split(val_size=config_validation.get("val_size", 0.2), seed=SEED)
    else:
        if config_traindataExp.get("do", False):
            train_dataset = dev_dataset.get_train_dataset_with_n_ids(num_ids=config_traindataExp.get("num_ids_train", 2), seed=SEED)
            val_dataset = None
        else:
            train_dataset, val_dataset = dev_dataset, None
    logger.info(f"Train dataset generated with {len(train_dataset)} samples, validation dataset generated with {len(val_dataset) if val_dataset else 0} samples")

    logger.info(f"Generating dataloaders...")
    train_sampler = None
    shuffle = config_training.get("shuffle", True)

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config_training.get("batch_size", 32),
        num_workers=config_training.get("num_workers", 4),
        collate_fn=train_dataset.collate_fn,
        pin_memory=True,
        sampler=train_sampler,
        shuffle=shuffle,
    )
    logger.info(f"Train dataloader generated with {len(train_dataloader)} batches")

    if val_dataset is not None:
        val_sampler = None
        shuffle = config_training.get("shuffle", False)

        val_dataloader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=config_validation.get("batch_size", 32),
            num_workers=config_validation.get("num_workers", 4),
            collate_fn=val_dataset.collate_fn,
            sampler=val_sampler,
            pin_memory=True,
            shuffle=shuffle,            
        )
        logger.info(f"Validation dataloader generated with {len(val_dataloader)} batches")

    run_eval_for_dev_data = config_training.get("eval_dev_split", False)

    if run_eval_for_dev_data:
        cfg_dataset_eval = config_evaluation["avatar_dataset"]
        dev_eval_config = copy.deepcopy(cfg_dataset_eval)
        dev_eval_config["dataset_class_kwargs"]["evaluation_pairs_csv"] = config_training["evaluation_pairs_dev_csv"]  # overwrite the csv to use for developmet split eval
        dev_eval_config["dataset_class_kwargs"]["root_dir"] = str(dev_eval_config["dataset_class_kwargs"]["root_dir"]).replace("test", "dev")
        dev_eval_dataset = get_test_dataset(dev_eval_config, device=device)
        dev_evaluation_dataloader = torch.utils.data.DataLoader(
                dev_eval_dataset,
                batch_size=config_evaluation.get("batch_size", 32),
                num_workers=config_evaluation.get("num_workers", 4),
                collate_fn=dev_eval_dataset.collate_fn,
                pin_memory=True,
                sampler=None,
            )

    # Optimizer
    optimizer = get_optimizer(model.parameters(), config_optimizer)
    logger.info(f"Optimizer <{config_optimizer['name']}> configured with kwargs: {config_optimizer['kwargs']}")

    # Loss function
    config_loss = config["loss"]
    # loss_fn = get_loss_function(config_loss)
    logger.info(f"Loss function <{config_loss['name']}> configured with kwargs: {config_loss['kwargs']}")

    
    # TODO: add option to continue training from a checkpoint

    # Configure checkpoints directory
    save_last_checkpoint = config_training.get("checkpoint", {}).get("save_last", False)
    save_best_checkpoint = config_training.get("checkpoint", {}).get("save_best", False)
    checkpoints_dir = os.path.join(output_dir_run, "checkpoints")
    if save_last_checkpoint or save_best_checkpoint:
        os.makedirs(checkpoints_dir, exist_ok=True)
        logger.info(f"Checkpoints will be saved in {checkpoints_dir}")
    else:
        logger.warning("No model checkpoints will be saved during training!")
    
    # Training loop
    epochs = config_training.get("n_epochs", 1)
    pbar = tqdm(range(epochs), file=sys.stdout, desc="Training loop")
    history = {"train_loss":[], "val_loss":[]}
    last_val_loss = np.inf
    validate_every_n_epochs = config_validation.get("every_n_epochs", 1)
    logger.info(f"Validating every {validate_every_n_epochs} epochs")

    logger.info(f"Start training for {epochs} epochs")
    for epoch in pbar:
        val_loss_improved = False

        # Solo set_epoch si el sampler existe y lo requiere
        if isinstance(train_dataloader.sampler, torch.utils.data.DistributedSampler):
            train_dataloader.sampler.set_epoch(epoch)

        model_to_train = model.module if hasattr(model, "module") else model
        train_loss_sum, num_batches = model_to_train.train_one_epoch(train_dataloader, optimizer)
        train_loss_sum_tensor = torch.tensor(train_loss_sum, device=device)
        batch_count_tensor = torch.tensor(num_batches, device=device)

        train_loss = (train_loss_sum_tensor / batch_count_tensor).item()

        pbar.set_postfix({"loss":train_loss})
        history["train_loss"].append(train_loss)
        metrics = []

        # Validation step
        if config_validation.get("do", False) and (epoch == 0 or (epoch+1) % validate_every_n_epochs == 0 or epoch == epochs-1):
            print('Init Validation step')
            val_loss_sum, num_val_batches = model.validate(val_dataloader, output_dir_plots=output_dir_plots)
            val_loss_sum_tensor = torch.tensor(val_loss_sum, device=device)
            batch_val_count_tensor = torch.tensor(num_val_batches, device=device)
            val_loss = (val_loss_sum_tensor / batch_val_count_tensor).item()

            history["val_loss"].append(val_loss)
            if val_loss < last_val_loss: 
                last_val_loss = val_loss
                val_loss_improved = True
            generate_loss_curve_plot(history, output_dir_plots)
            print("Validation steps")

            if run_eval_for_dev_data:
                logger.info(f"Running evaluation metrics on dev data...")
                metrics_epoch = evaluate_model(model, dev_evaluation_dataloader, output_dir_plots, device, config_evaluation.get("metric", "cosine"),SCORE_THRESHOLD=None,postfix=f"_dev_split_epoch{epoch+1}")
                logger.info(f"Metrics for dev dataset at epoch {epoch}: {metrics_epoch}")
                metrics.append(metrics_epoch)
            print('End Validation step')

            pbar.set_postfix({
                "loss": train_loss,
                "val_loss": f"{val_loss:.4f}" if not np.isnan(val_loss) else last_val_loss,
            })

        else:
            val_loss = np.nan
            history["val_loss"].append(val_loss)

            # plotting
            generate_loss_curve_plot(history, output_dir_plots)

        # Save checkpoints
        last_checkpoint_path, best_checkpoint_path = None, None
        if save_last_checkpoint:
            last_checkpoint_path = os.path.join(checkpoints_dir, "last_checkpoint.pth")
            save_checkpoint(model, optimizer, last_checkpoint_path, epoch)
            logger.info(f"Last checkpoint saved at {last_checkpoint_path}")
        if save_best_checkpoint and val_loss_improved:
            best_checkpoint_path = os.path.join(checkpoints_dir, "best_checkpoint.pth")
            save_checkpoint(model, optimizer, best_checkpoint_path, epoch)
            logger.info(f"Validation loss improved from last epoch, best checkpoint saved at {best_checkpoint_path}")

            
    logger.info(f"History of training loss: {history}")
    logger.info("Training finished successfully")

    if config_evaluation.get("do", False):
        logger.info("Evaluating model on test dataset")
        output_dir_evaluation =  os.path.join(output_dir_plots, "eval_on_test_data")
        os.makedirs(output_dir_evaluation, exist_ok=True)
        metrics_last_all_ds = do_evaluation_step(config_evaluation, model, device, output_dir_evaluation, postfix="_last_checkpoint")

        logger.info("Evaluating model on test dataset using BEST checkpoint obtained during training...")
        best_model = copy.deepcopy(model)
        best_model, _, _ = load_checkpoint(model, None, os.path.join(checkpoints_dir, "best_checkpoint.pth"))
        metrics_best_all_ds = do_evaluation_step(config_evaluation, best_model, device, output_dir_evaluation, postfix="_best_checkpoint")

 
def save_checkpoint(model, optimizer, save_path, epoch):
    logger = getLogger("training_utils")
    logger.debug("Saving model.state_dict()")
    model_state = model.state_dict()

    # Save model, optimizer, and epoch
    torch.save({
        'model_state_dict': model_state,
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch
    }, save_path)

    logger.debug(f"Checkpoint saved to {save_path} (epoch {epoch+1})")
