"""
Shared helpers for file manifest cache: ensure manifest exists (build if needed)
and load (root, paths) for use by LocalImages in IBQ and Open-MAGVIT2.
Also supports loading failed-samples JSON to skip bad paths (skip_files).
"""
import json
import os
import subprocess
import sys
import time


def _repo_root():
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _build_script_path():
    return os.path.join(_repo_root(), "scripts", "build_local_ibqgan_image_paths.py")


def _resolve_manifest_path(manifest_path):
    """Resolve manifest path to a single absolute path so all processes (e.g. DDP) use the same file."""
    manifest_path = os.path.expanduser(manifest_path)
    if not os.path.isabs(manifest_path):
        manifest_path = os.path.join(_repo_root(), manifest_path)
    return os.path.abspath(manifest_path)


def _is_rank_zero():
    """True if we are the main process (only this one should run the build script)."""
    try:
        return int(os.environ.get("LOCAL_RANK", "0")) == 0
    except ValueError:
        return True


def load_manifest(manifest_path):
    """
    Load root and list of absolute image paths from a JSON manifest.
    Manifest format: {"root": "/path", "paths": ["/abs/path", ...] or ["rel/path", ...], "count": N}
    """
    with open(manifest_path) as f:
        data = json.load(f)
    root = os.path.abspath(os.path.expanduser(data["root"]))
    paths = data.get("paths", [])
    if paths and not os.path.isabs(paths[0]):
        paths = [os.path.join(root, p) for p in paths]
    return root, paths


def ensure_manifest(manifest_path, build_config_path=None, root=None):
    """
    Ensure the manifest file exists; if not, run the build script, then load and return (root, paths).

    When building:
      - If root is provided: run the build script with --root <root> and -o <manifest_path>
        (each split can have its own manifest built from its own root).
      - If root is not provided: use build_config_path (default: derive from manifest_path by
        replacing '_image_paths.json' with '.yaml') so the build script reads root from that YAML.

    Uses a single canonical path (resolved from repo root) so DDP processes all
    see the same file. Only rank 0 runs the build script; other ranks wait for
    the file to appear (avoids overwriting the manifest with an empty list).
    """
    manifest_path = _resolve_manifest_path(manifest_path)
    if os.path.isfile(manifest_path):
        return load_manifest(manifest_path)

    script = _build_script_path()
    if not os.path.isfile(script):
        raise FileNotFoundError(
            f"Build script not found: {script}. Cannot generate manifest."
        )
    out_dir = os.path.dirname(manifest_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    build_root = os.path.abspath(os.path.expanduser(root)) if root else None
    if build_config_path is None and not build_root:
        if manifest_path.endswith("_image_paths.json"):
            build_config_path = manifest_path[:-len("_image_paths.json")] + ".yaml"
        else:
            raise ValueError(
                "Manifest does not exist; provide build_config_path or root (from split config)."
            )
    if build_config_path is not None:
        if not os.path.isabs(build_config_path):
            build_config_path = os.path.join(_repo_root(), build_config_path)
        build_config_path = os.path.abspath(os.path.expanduser(build_config_path))
        if not os.path.isfile(build_config_path):
            raise FileNotFoundError(
                f"Manifest missing and build config not found: {build_config_path}"
            )

    # Only rank 0 runs the build script so we never have multiple processes overwriting the manifest.
    if _is_rank_zero():
        cmd = [sys.executable, script, "-o", manifest_path]
        if build_root:
            cmd.extend(["--root", build_root])
        else:
            cmd.extend(["--config", build_config_path])
        subprocess.run(cmd, check=True, cwd=_repo_root())
    else:
        # Other ranks: wait for rank 0 to create the manifest (avoid racing with build script).
        for _ in range(300):
            if os.path.isfile(manifest_path):
                break
            time.sleep(0.5)
        if not os.path.isfile(manifest_path):
            raise FileNotFoundError(
                f"Manifest still missing after waiting: {manifest_path}. "
                "Ensure rank 0 can run the build script and the image root is visible."
            )

    return load_manifest(manifest_path)


def get_failed_samples_path_from_manifest(manifest_path):
    """
    Return the path to the failed-samples JSON for the given manifest.
    Convention: configs/.../local_ibqgan_256_image_paths.json
                -> configs/.../local_ibqgan_256_failed_samples.json
    """
    manifest_path = _resolve_manifest_path(manifest_path)
    if manifest_path.endswith("_image_paths.json"):
        return manifest_path[: -len("_image_paths.json")] + "_failed_samples.json"
    return None


def load_failed_paths(failed_samples_path, normalize=True):
    """
    Load the set of file paths to skip from a failed-samples JSON.
    Paths are taken from "failed"[].path or, if present, "skip_paths".
    If normalize is True (default), paths are normalized for local comparison.
    If False, raw strings are used (e.g. for S3 keys).
    """
    failed_samples_path = os.path.abspath(os.path.expanduser(failed_samples_path))
    if not os.path.isfile(failed_samples_path):
        return set()
    with open(failed_samples_path) as f:
        data = json.load(f)
    paths = set()
    def add(p):
        paths.add(
            os.path.normpath(os.path.abspath(os.path.expanduser(p)))
            if normalize
            else p
        )
    if "skip_paths" in data:
        for p in data["skip_paths"]:
            add(p)
    for entry in data.get("failed", []):
        if "path" in entry:
            add(entry["path"])
    return paths
