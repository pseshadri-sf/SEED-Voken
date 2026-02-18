#!/usr/bin/env python3
"""
Run through the train dataloader for a given config and log every image that errors
to a JSON file. The JSON can be used to skip those samples during training (e.g. by
filtering the dataset or using a custom sampler).

Usage:
  python scripts/scan_dataloader_failures.py --config /workspace/scripts/SEED-Voken/configs/IBQ/gpu/local_ibqgan_256.yaml
  python scripts/scan_dataloader_failures.py --config configs/Open-MAGVIT2/gpu/s3_runpod_opencua_lfqgan_128_L.yaml --max-samples 5000 --output failed_images.json
  python scripts/scan_dataloader_failures.py --config configs/Open-MAGVIT2/gpu/imagenet_lfqgan_128_L.yaml --output failed.json
"""
import argparse
import json
import os
import sys
import traceback

from tqdm import tqdm

# Project root for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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
    """Return a stable identifier for the sample at index (path, S3 key, or index)."""
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
    """Load the train dataset from a training YAML (data.init_args.train)."""
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
            "Train target is a pretrain (e.g. webdataset) dataset; this script supports "
            "indexable image datasets (S3Images, LocalImages, ImageNet, etc.)."
        )
    return instantiate_from_config(train_cfg)


def main():
    parser = argparse.ArgumentParser(
        description="Scan dataloader for failing images and write them to a JSON skip list"
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to training YAML (e.g. configs/Open-MAGVIT2/gpu/s3_runpod_opencua_lfqgan_128_L.yaml)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="failed_samples.json",
        help="Output JSON path (default: failed_samples.json)",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Max number of samples to scan (default: all)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print each failure to stderr",
    )
    args = parser.parse_args()

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    config_path = args.config if os.path.isabs(args.config) else os.path.join(root, args.config)
    if not os.path.isfile(config_path):
        print(f"Error: config file not found: {config_path}", file=sys.stderr)
        sys.exit(1)

    tqdm.write(f"Loading train dataset from config: {config_path}")
    dataset = load_train_dataset_from_config(config_path)
    total = len(dataset)
    tqdm.write(f"  Dataset size: {total}")

    max_samples = args.max_samples if args.max_samples is not None else total
    n = min(max_samples, total)

    failed = []
    pbar = tqdm(range(n), desc="Scanning", unit="samples", dynamic_ncols=True)
    for i in pbar:
        try:
            dataset[i]
        except Exception as e:
            identifier = get_sample_identifier(dataset, i)
            entry = {
                "index": i,
                "path": identifier,
                "error": str(e),
            }
            failed.append(entry)
            pbar.set_postfix(failures=len(failed), refresh=False)
            if args.verbose:
                tqdm.write(f"  [{i}] {identifier}: {e}")
                traceback.print_exc(file=sys.stderr)

    out_path = args.output if os.path.isabs(args.output) else os.path.join(root, args.output)
    result = {
        "config": os.path.abspath(config_path),
        "total_scanned": n,
        "total_dataset": total,
        "failed_count": len(failed),
        "failed": failed,
        "skip_paths": [e["path"] for e in failed],
    }
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)

    tqdm.write(f"Done. Failed: {len(failed)}/{n}. Wrote {out_path}")
    if failed:
        tqdm.write("  Use 'skip_paths' or 'failed' in training to skip these samples.")


if __name__ == "__main__":
    main()
