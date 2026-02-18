"""
Shared worker and dataset helpers for scan_dataloader_failures_async.
Defined in a separate module so ProcessPoolExecutor workers can import it
without re-running the main script (required for spawn on Windows).
"""
import os
import sys

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_SCRIPT_DIR)
sys.path.insert(0, _ROOT)

from omegaconf import OmegaConf


def get_obj_from_str(string):
    import importlib
    module, cls = string.rsplit(".", 1)
    return getattr(importlib.import_module(module, package=None), cls)


def instantiate_from_config(config):
    if "target" not in config:
        raise KeyError("Expected key `target` to instantiate.")
    params = config.get("params", dict())
    if not isinstance(params, dict):
        params = OmegaConf.to_container(params, resolve=True)
    return get_obj_from_str(config["target"])(**params)


def get_sample_identifier(dataset, index):
    try:
        if hasattr(dataset, "data") and hasattr(dataset.data, "labels"):
            labels = dataset.data.labels
            if "file_path_" in labels:
                val = labels["file_path_"][index]
                return val if isinstance(val, str) else str(val)
            if "s3_key" in labels:
                val = labels["s3_key"][index]
                return val if isinstance(val, str) else str(val)
    except Exception:
        pass
    return str(index)


def load_train_dataset_from_config(config_path: str):
    conf = OmegaConf.load(config_path)
    data = conf.get("data", {})
    init_args = data.get("init_args", {})
    train_cfg = init_args.get("train")
    if train_cfg is None:
        raise ValueError(
            f"No data.init_args.train in {config_path}. "
            "Ensure the YAML has data.init_args.train (target + params)."
        )
    train_cfg = OmegaConf.to_container(train_cfg, resolve=True)
    if "pretrain" in train_cfg.get("target", ""):
        raise ValueError(
            "Train target is a pretrain (e.g. webdataset) dataset; not supported."
        )
    return instantiate_from_config(train_cfg)


def scan_chunk(args):
    """Load dataset from config and scan indices [start, end). Returns list of failure dicts."""
    config_path, start, end = args
    dataset = load_train_dataset_from_config(config_path)
    failed = []
    for i in range(start, end):
        try:
            dataset[i]
        except Exception as e:
            identifier = get_sample_identifier(dataset, i)
            failed.append({
                "index": i,
                "path": identifier,
                "error": str(e),
            })
    return failed
