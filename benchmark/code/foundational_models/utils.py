import numpy as np
import logging
import random
import torch
import yaml
import os

def set_seed(seed=None):
    if seed is None:
        seed = random.SystemRandom().randint(0, 2**32 - 1)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    return seed


def deep_merge(base, override):
    for k, v in override.items():
        if (
            k in base
            and isinstance(base[k], dict)
            and isinstance(v, dict)
        ):
            base[k] = deep_merge(base[k], v)
        else:
            base[k] = v
    return base

def load_config(path):
    with open(path, 'r') as f:
        cfg = yaml.safe_load(f)

    if "base_config" in cfg:
        base_path = os.path.join(os.path.dirname(path), cfg["base_config"])
        with open(base_path, 'r') as f:
            base_cfg = yaml.safe_load(f)

        cfg.pop("base_config")
        cfg = deep_merge(base_cfg, cfg)

    return cfg

def get_dataset_class(dataset_name, *args, **kwargs):
    import datasets as datasets
    dataset_class = getattr(datasets, dataset_name, None)
    if dataset_class is None:
        raise NotImplementedError(f"Dataset class '{dataset_name}' is not found in datasets package.")
    return dataset_class(*args, **kwargs)

def get_optimizer(model_params, config):
    try:
        optimizer_name = config.get("name", "adam").lower()
        optimizer_kwargs = config.get("kwargs", {})
        if optimizer_name == "adam":
            return torch.optim.Adam(model_params, **optimizer_kwargs)
        elif optimizer_name == "sgd":
            return torch.optim.SGD(model_params, **optimizer_kwargs)
        else:
            raise NotImplementedError(f"Optimizer {optimizer_name} is not implemented.")
    except Exception as e:
        raise Exception(f"Error building optimizer: {e}")
    

def get_model(model_name: str, *args, **kwargs):
    import models as models
    model_class = getattr(models, model_name, None)
    if model_class is None:
        raise NotImplementedError(f"Model class '{model_name}' not found in models package.")
    out_dir = kwargs.get("save_model_to_dir", False)
    if out_dir:
        copy_model_source(model_class, out_dir)
    return model_class(*args, **kwargs)


def copy_model_source(model_cls, dst_dir):
    import inspect
    import shutil

    source_file = inspect.getsourcefile(model_cls) or inspect.getfile(model_cls)
    if not os.path.exists(source_file):
        raise FileNotFoundError(f"Cannot locate source for {model_cls!r}")
    os.makedirs(dst_dir, exist_ok=True)
   
    basename = os.path.basename(source_file)
    dst_path = os.path.join(dst_dir, basename)
    shutil.copy(source_file, dst_path)
    return dst_path