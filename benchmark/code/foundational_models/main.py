import torch
import os
import traceback
import matplotlib
matplotlib.use("Agg")
 
 
def main(config, mode="train"):
    print("===> Starting main(config, mode={})".format(mode), flush=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}", flush=True)

    from utils import (
        get_model,
    )

    from train import (
        run_training_eval
    )

    from evaluate import (
        run_test
    )

    # Configurations
    config_model = config.get("model", {})
    config_evaluation = config.get("evaluation", {})
    SEED = config.get("seed", None)


    # Output directories
    config_outputs = config["outputs"]
    output_dir_run = config_outputs["output_dir_run"]
    output_dir_plots = os.path.join(output_dir_run, config_outputs.get("plots_dirname", "plots"))
    os.makedirs(output_dir_plots, exist_ok=True)


    # Get architecture and model configuration
    architecture_config = config_model.get("architecture", {})
    print(f"Generating model", flush=True)
    model = get_model(
        architecture_config.get("name", "SimpleLSTMModel"),
        **architecture_config.get("kwargs", {}),
        save_model_to_dir=output_dir_run,
    )
    print(f"Moving model to device {device}", flush=True)
    model = model.to(device)
    print(f"Model setup complete", flush=True)


    # Run mode
    if mode == "train":
        run_training_eval(config, model, device, output_dir_run, output_dir_plots)
    elif mode == "test":
        run_test(config_evaluation, model, device, output_dir_plots)


    torch.cuda.empty_cache()
