"""
Dataset for loading images from a local directory (recursive glob).
Used for evaluation on custom image folders.
"""
import os
import glob
from omegaconf import OmegaConf
from torch.utils.data import Dataset

from src.Open_MAGVIT2.data.base import ImagePaths
from src.Open_MAGVIT2.util import retrieve

# Supported image extensions (match s3_images)
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}


class LocalImages(Dataset):
    """
    Load images from a local directory. Recursively finds all image files.
    Config:
        root: path to directory containing images (required)
        size: resize/crop size (default from config or 128)
        random_crop: use random crop for train (default False for eval)
        original_reso: if True, do not resize/crop (default False)
    """

    def __init__(self, config=None):
        self.config = config or OmegaConf.create()
        if not isinstance(self.config, dict):
            self.config = OmegaConf.to_container(self.config)
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
        if len(paths) == 0:
            raise ValueError(f"No images found under {root} (extensions: {list(IMAGE_EXTENSIONS)})")
        print(f"Found {len(paths)} images in {root}")

        size = retrieve(self.config, "size", default=128)
        random_crop = retrieve(self.config, "random_crop", default=False)
        original_reso = retrieve(self.config, "original_reso", default=False)

        self.data = ImagePaths(
            paths,
            original_reso=original_reso,
            size=size,
            random_crop=random_crop,
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        return self.data[i]
