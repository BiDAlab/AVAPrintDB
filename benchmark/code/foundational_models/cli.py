import argparse
import yaml

import datetime as dt
import shutil
from utils import load_config

from pathlib import Path
from utils import load_config, set_seed


def parse_args():
    parser = argparse.ArgumentParser(description="Main Script for AVAPrintDB Foundational Models Benchmark")

    parser.add_argument(
        "--config_file",
        type=str,
        default="benchmark/code/foundational_models/default_config.yaml",
        help="Path to the configuration file",
    )

    parser.add_argument(
        "--mode",
        type=str,
        default="test",
        choices=["train", "test"],
        help=(
            "Execution mode: 'train' for the full pipeline "
            "or 'test' for evaluation only"
        ),
    )

    parser.add_argument(
        "--maxframes",
        type=int,
        default=60,
        help="Maximum number of frames to process",
    )

    parser.add_argument(
        "--features",
        type=str,
        default="CLIP",
        choices=["CLIP", "DINO"],
        help="Type of features to use: 'CLIP' or 'DINO'",
    )

    parser.add_argument(
        "--ktriplets",
        type=int,
        default=6,
        help="Number of triplets to sample/use in the dataset.",
    )

    parser.add_argument(
        "--traindataSize",
        type=int,
        default=4,
        help="Number of identities/samples to use for training. Set to -1 to use the full training dataset.",
    )

    return parser.parse_args()


def load_seed_from_config(config_path):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config.get("seed", None)

def apply_cli_overrides(config, args):
    feature_dims = {
                        "CLIP": 768,
                        "DINO": 1024,
                    }

    config["model"]["architecture"]["kwargs"]["embedding_dim"] = feature_dims[args.features]

    config["avatar_dataset"]["root_dir"] = ( config["avatar_dataset"]["embedding_dir"] + args.features + "Feats" )

    config["evaluation"]["avatar_dataset"]["dataset_class_kwargs"]["root_dir"] = ( config["avatar_dataset"]["root_dir"] + "/TEST/" )

    config["traindataExp"]["num_ids_train"] = args.traindataSize

    config["avatar_dataset"]["dataset_class_kwargs"]["max_frames"] = args.maxframes
    config["evaluation"]["avatar_dataset"]["dataset_class_kwargs"]["max_frames"] = args.maxframes

    config["avatar_dataset"]["dataset_class_kwargs"]["ktriplets"] = args.ktriplets

    return config

def build_run_name(config, args):
    config_path = Path(args.config_file)
    config_name = config_path.stem

    datasets = "--".join(config["avatar_dataset"]["datasets"])
    generators = "--".join(config["avatar_dataset"]["generators"])
    datasets_generators = f"{datasets}_{generators}"

    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    mode_tag = "test_" if args.mode == "test" else ""

    run_name = (
        f"{mode_tag}"
        f"{config_name}_"
        f"{datasets_generators}_"
        f"{args.maxframes}_"
        f"{args.ktriplets}_"
        f"{args.traindataSize}_"
        f"{args.features}_"
        f"{timestamp}"
    )

    return run_name

def prepare_output_dir(config, args):
    config_path = Path(args.config_file)

    root_output = Path(config["outputs"].get("root_output", "outputs"))
    run_name = build_run_name(config, args)

    output_dir_run = root_output / run_name
    output_dir_run.mkdir(parents=True, exist_ok=True)

    config["outputs"]["output_dir_run"] = str(output_dir_run)

    shutil.copyfile(
        config_path,
        output_dir_run / "config.yaml",
    )

    shutil.copyfile(
        config_path.parent / "default_config.yaml",
        output_dir_run / "base_config.yaml",
    )

    return config

def run():
    args = parse_args()

    seed = load_seed_from_config(args.config_file)
    final_seed = set_seed(seed)

    from main import main

    config = load_config(args.config_file)
    config = apply_cli_overrides(config, args)
    config = prepare_output_dir(config, args)
    config["seed"] = final_seed

    main(config, mode=args.mode)


if __name__ == "__main__":
    run()
    
