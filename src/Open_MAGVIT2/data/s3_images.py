import os
import numpy as np
from PIL import Image, ImageFile
from omegaconf import OmegaConf
from torch.utils.data import Dataset
import boto3
from botocore.exceptions import ClientError
from io import BytesIO

from src.Open_MAGVIT2.data.base import ImagePaths
from src.Open_MAGVIT2.util import retrieve, KeyNotFoundError

import logging
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

ImageFile.LOAD_TRUNCATED_IMAGES = True


class S3ImagePaths(Dataset):
    """
    Custom ImagePaths that loads images lazily from S3.
    Downloads images on-demand to a temporary location or loads directly into memory.
    """
    def __init__(self, s3_keys, s3_client, bucket, size=None, random_crop=False, 
                 original_reso=False, labels=None, cache_dir=None):
        self.s3_client = s3_client
        self.bucket = bucket
        self.size = size
        self.random_crop = random_crop
        self.original_reso = original_reso
        self.cache_dir = cache_dir
        
        # Create cache directory if specified
        if self.cache_dir is not None:
            os.makedirs(self.cache_dir, exist_ok=True)
        
        self.labels = dict() if labels is None else labels
        self.labels["file_path_"] = s3_keys
        self._length = len(s3_keys)
        
        # Setup preprocessing (same as ImagePaths)
        if self.size is not None and self.size > 0:
            import torchvision.transforms as T
            self.rescaler = T.Resize(self.size)
            if not self.random_crop:
                self.cropper = T.CenterCrop((self.size, self.size))
            else:
                self.cropper = T.RandomCrop((self.size, self.size))
            self.preprocessor = T.Compose([self.rescaler, self.cropper])
        else:
            self.preprocessor = lambda x: x

    def __len__(self):
        return self._length

    def _load_from_s3(self, s3_key):
        """Load image from S3, either from cache or directly."""
        # Check cache first if cache_dir is set
        if self.cache_dir is not None:
            # Preserve directory structure in cache, but make it safe
            # Replace any problematic characters but keep directory structure
            safe_key = s3_key.replace("\\", "/")
            cache_path = os.path.join(self.cache_dir, safe_key)
            
            if os.path.exists(cache_path):
                try:
                    return Image.open(cache_path)
                except Exception as e:
                    # If cached file is corrupted, remove it and re-download
                    print(f"Warning: Cached file {cache_path} is corrupted, re-downloading. Error: {e}")
                    try:
                        os.remove(cache_path)
                    except:
                        pass
        
        # Download from S3
        try:
            response = self.s3_client.get_object(Bucket=self.bucket, Key=s3_key)
            image_data = response['Body'].read()
            image = Image.open(BytesIO(image_data))
            
            # Save to cache if cache_dir is set
            if self.cache_dir is not None:
                safe_key = s3_key.replace("\\", "/")
                cache_path = os.path.join(self.cache_dir, safe_key)
                cache_dir = os.path.dirname(cache_path)
                if cache_dir:
                    os.makedirs(cache_dir, exist_ok=True)
                # Save a copy of the image
                image_copy = image.copy()
                image_copy.save(cache_path)
            
            return image
        except ClientError as e:
            raise FileNotFoundError(f"Failed to load image from S3: s3://{self.bucket}/{s3_key}. Error: {e}")

    def preprocess_image(self, s3_key):
        """Load and preprocess image from S3."""
        image = self._load_from_s3(s3_key)
        
        if not image.mode == "RGB":
            image = image.convert("RGB")
        
        if not self.original_reso:
            image = self.preprocessor(image)
        
        image = np.array(image)
        image = (image/127.5 - 1.0).astype(np.float32)
        return image
    
    def __getitem__(self, i):
        example = dict()
        example["image"] = self.preprocess_image(self.labels["file_path_"][i])
        for k in self.labels:
            example[k] = self.labels[k][i]
        return example


class S3ImagesBase(Dataset):
    """
    Base class for loading images from S3, similar to ImageNetBase.
    """
    # Supported image extensions
    IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp'}
    
    def __init__(self, config=None):
        self.config = config or OmegaConf.create()
        if not type(self.config) == dict:
            self.config = OmegaConf.to_container(self.config)
        
        # Get S3 configuration
        self.bucket = retrieve(self.config, "bucket", default=None)
        self.object_prefix = retrieve(self.config, "object", default="")
        try:
            self.aws_access_key_id = retrieve(self.config, "aws_access_key_id", default=None)
            self.aws_secret_access_key = retrieve(self.config, "aws_secret_access_key", default=None)
            self.aws_session_token = retrieve(self.config, "aws_session_token", default=None)
        except KeyNotFoundError:
            log.info("No AWS credentials found in config, fetching from environment variables.")
            self.aws_access_key_id = os.getenv("AWS_ACCESS_KEY_ID", None)
            self.aws_secret_access_key = os.getenv("AWS_SECRET_ACCESS_KEY", None)
            self.aws_session_token = os.getenv("AWS_SESSION_TOKEN", None)
        self.region_name = retrieve(self.config, "region_name", default=None)
        try:
            self.cache_dir = retrieve(self.config, "cache_dir", default=None)
            os.makedirs(self.cache_dir, exist_ok=True)
        except KeyNotFoundError:
            log.info("No cache directory found in config")
            self.cache_dir = None
        
        if self.bucket is None:
            raise ValueError("'bucket' must be specified in config")
        
        # Initialize S3 client
        s3_kwargs = {}
        if self.aws_access_key_id:
            s3_kwargs['aws_access_key_id'] = self.aws_access_key_id
        if self.aws_secret_access_key:
            s3_kwargs['aws_secret_access_key'] = self.aws_secret_access_key
        if self.aws_session_token:
            s3_kwargs['aws_session_token'] = self.aws_session_token
        if self.region_name:
            s3_kwargs['region_name'] = self.region_name
        
        self.s3_client = boto3.client('s3', **s3_kwargs)
        
        # Get random_crop setting
        self.random_crop = retrieve(self.config, "random_crop", default=False)
        
        # List and load images
        self._load()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        return self.data[i]

    def _list_s3_images(self, bucket, prefix=""):
        """
        Recursively list all image files from S3 bucket with given prefix.
        Returns a list of S3 keys (relative paths within the bucket).
        """
        image_keys = []
        paginator = self.s3_client.get_paginator('list_objects_v2')
        
        try:
            pages = paginator.paginate(Bucket=bucket, Prefix=prefix)
            
            for page in pages:
                if 'Contents' not in page:
                    continue
                
                for obj in page['Contents']:
                    key = obj['Key']
                    # Skip directories (keys ending with /)
                    if key.endswith('/'):
                        continue
                    
                    # Check if file has image extension
                    _, ext = os.path.splitext(key.lower())
                    if ext in self.IMAGE_EXTENSIONS:
                        # Return relative path from prefix
                        if prefix and key.startswith(prefix):
                            rel_key = key[len(prefix):].lstrip('/')
                        else:
                            rel_key = key
                        image_keys.append(key)  # Store full S3 key for access
        
        except ClientError as e:
            raise RuntimeError(f"Failed to list objects from S3 bucket '{bucket}' with prefix '{prefix}'. Error: {e}")
        
        image_keys = sorted(image_keys)
        print(f"Found {len(image_keys)} images in s3://{bucket}/{prefix}")
        return image_keys

    def _filter_keys(self, keys):
        """Filter S3 keys if needed (similar to _filter_relpaths in ImageNetBase)."""
        # Can add filtering logic here if needed
        return keys

    def _load(self):
        """Load all image keys from S3 and create dataset."""
        # List all images recursively from S3
        s3_keys = self._list_s3_images(self.bucket, self.object_prefix)
        
        if len(s3_keys) == 0:
            raise ValueError(f"No images found in s3://{self.bucket}/{self.object_prefix}")
        
        # Filter keys if needed
        l1 = len(s3_keys)
        s3_keys = self._filter_keys(s3_keys)
        if l1 != len(s3_keys):
            print(f"Removed {l1 - len(s3_keys)} files during filtering.")
        
        # Extract relative paths for labels (remove prefix if present)
        if self.object_prefix:
            relpaths = [key[len(self.object_prefix):].lstrip('/') if key.startswith(self.object_prefix) else key 
                       for key in s3_keys]
        else:
            relpaths = s3_keys
        
        # Create labels dictionary
        labels = {
            "relpath": np.array(relpaths),
            "s3_key": np.array(s3_keys),
        }
        
        # Create dataset using custom S3ImagePaths
        self.data = S3ImagePaths(
            s3_keys=s3_keys,
            s3_client=self.s3_client,
            bucket=self.bucket,
            size=retrieve(self.config, "size", default=0),
            random_crop=self.random_crop,
            original_reso=retrieve(self.config, "original_reso", default=False),
            labels=labels,
            cache_dir=self.cache_dir
        )


class S3Images(S3ImagesBase):
    """
    Main dataset class for loading images from S3.
    Usage:
        config = {
            "bucket": "my-bucket",
            "object": "path/to/images/",  # prefix/path in S3
            "size": 256,
            "random_crop": True,
            "aws_access_key_id": "optional",
            "aws_secret_access_key": "optional",
            "region_name": "us-east-1",  # optional
            "cache_dir": "/tmp/s3_cache"  # optional, for caching downloaded images
        }
        dataset = S3Images(config=config)
    """
    pass
