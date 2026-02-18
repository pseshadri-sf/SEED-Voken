"""
Dataset for loading images from a local directory (recursive glob).
Used for evaluation on custom image folders. IterableDataset for sequential reads.
If manifest_path is set in config: use that JSON cache (build via build_local_ibqgan_image_paths.py if missing).
Optional skip_files (default True): exclude paths listed in the config's failed_samples JSON if present.
"""
import os
import glob
import logging
from omegaconf import OmegaConf
from torch.utils.data import IterableDataset

from src.Open_MAGVIT2.data.base import IterableImagePaths
from src.Open_MAGVIT2.util import retrieve
from src.manifest_utils import (
    ensure_manifest,
    get_failed_samples_path_from_manifest,
    load_failed_paths,
)

log = logging.getLogger(__name__)

# Supported image extensions (match s3_images)
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}


class LocalImages(IterableDataset):
    """
    Load images from a local directory. Recursively finds all image files.
    IterableDataset for sequential reads (mitigates random I/O).
    Config:
        root: path to directory containing images (required)
        size: resize/crop size (default from config or 128)
        random_crop: use random crop for train (default False for eval)
        original_reso: if True, do not resize/crop (default False)
        skip_files: if True (default), exclude paths from failed_samples JSON when present
        failed_samples_path: optional path to failed_samples JSON (auto-derived from manifest_path when possible)
    """

    def __init__(self, config=None):
        self.config = config or OmegaConf.create()
        if not isinstance(self.config, dict):
            self.config = OmegaConf.to_container(self.config)

        skip_files = retrieve(self.config, "skip_files", default=True)
        failed_samples_path = retrieve(self.config, "failed_samples_path", default=None)
        manifest_path = retrieve(self.config, "manifest_path", default=None)
        # When failed_samples_path is None, default to derived path from manifest so skip_files can apply
        if failed_samples_path is None and manifest_path:
            derived = get_failed_samples_path_from_manifest(manifest_path)
            if derived:
                derived = os.path.abspath(os.path.expanduser(derived))
                if os.path.isfile(derived):
                    failed_samples_path = derived
        paths_to_skip = set()
        if skip_files and failed_samples_path:
            paths_to_skip = load_failed_paths(failed_samples_path, normalize=True)
            if paths_to_skip:
                log.info("Skipping %d paths from failed_samples: %s", len(paths_to_skip), failed_samples_path)

        def filter_paths(paths):
            if not paths_to_skip:
                return paths
            return [p for p in paths if os.path.normpath(os.path.abspath(p)) not in paths_to_skip]

        if manifest_path:
            root_for_build = retrieve(self.config, "root", default=None)
            if root_for_build:
                root_for_build = os.path.abspath(os.path.expanduser(root_for_build))
            root, paths = ensure_manifest(manifest_path, root=root_for_build)
            paths = filter_paths(paths)
            if len(paths) == 0:
                raise ValueError("No image paths in manifest after filtering failed samples.")
            size = retrieve(self.config, "size", default=128)
            random_crop = retrieve(self.config, "random_crop", default=False)
            original_reso = retrieve(self.config, "original_reso", default=False)
            self.data = IterableImagePaths(
                paths,
                original_reso=original_reso,
                size=size,
                random_crop=random_crop,
            )
            print(f"Loaded {len(paths)} images from manifest (root: {root})")
            return

        root = retrieve(self.config, "root", default=None)
        if not root:
            raise ValueError("'root' must be specified in config (path to image directory)")
        root = os.path.expanduser(root)
        if not os.path.isdir(root):
            raise FileNotFoundError(f"Image root directory not found: {root}")

        paths = []
        for ext in IMAGE_EXTENSIONS:
            paths.extend(
                glob.glob(os.path.join(root, "**", f"*{ext}"), recursive=True)
            )
        paths = sorted(paths)
        paths = filter_paths(paths)
        if len(paths) == 0:
            raise ValueError(f"No images found under {root} (extensions: {list(IMAGE_EXTENSIONS)})")
        print(f"Found {len(paths)} images in {root}")

        size = retrieve(self.config, "size", default=128)
        random_crop = retrieve(self.config, "random_crop", default=False)
        original_reso = retrieve(self.config, "original_reso", default=False)

        self.data = IterableImagePaths(
            paths,
            original_reso=original_reso,
            size=size,
            random_crop=random_crop,
        )

    def __len__(self):
        return len(self.data)

    def __iter__(self):
        yield from self.data

    def __getitem__(self, i):
        return self.data[i]
