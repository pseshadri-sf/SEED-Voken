#!/usr/bin/env python3
"""
Download up to N images from an S3 bucket prefix into a local directory.
Usage:
  python scripts/download_s3_images.py
  (or set BUCKET, PREFIX, OUT_DIR, MAX_IMAGES via env or edit below)
"""
import os
import sys

# Add project root for imports if needed
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import boto3
from botocore.exceptions import ClientError

BUCKET = os.environ.get("S3_BUCKET", "skyfall-research-models")
PREFIX = os.environ.get("S3_PREFIX", "EWM-DataCollection/images/ScreenAgent/Train/")
OUT_DIR = os.environ.get("OUT_DIR", "test_data")
MAX_IMAGES = int(os.environ.get("MAX_IMAGES", "100"))
REGION = os.environ.get("AWS_REGION", "us-east-2")

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}


def main():
    out_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), OUT_DIR)
    os.makedirs(out_path, exist_ok=True)

    client = boto3.client("s3", region_name=REGION)
    paginator = client.get_paginator("list_objects_v2")
    image_keys = []

    for page in paginator.paginate(Bucket=BUCKET, Prefix=PREFIX):
        if "Contents" not in page:
            continue
        for obj in page["Contents"]:
            key = obj["Key"]
            if key.endswith("/"):
                continue
            _, ext = os.path.splitext(key.lower())
            if ext in IMAGE_EXTENSIONS:
                image_keys.append(key)
                if len(image_keys) >= MAX_IMAGES:
                    break
        if len(image_keys) >= MAX_IMAGES:
            break

    image_keys = image_keys[:MAX_IMAGES]
    print(f"Downloading {len(image_keys)} images from s3://{BUCKET}/{PREFIX} to {out_path}")

    for i, key in enumerate(image_keys):
        # Preserve relative path under OUT_DIR to avoid name collisions
        rel = key[len(PREFIX):].lstrip("/") if key.startswith(PREFIX) else key
        local_path = os.path.join(out_path, rel)
        local_dir = os.path.dirname(local_path)
        if local_dir:
            os.makedirs(local_dir, exist_ok=True)
        try:
            client.download_file(BUCKET, key, local_path)
            print(f"  [{i+1}/{len(image_keys)}] {rel}")
        except ClientError as e:
            print(f"  ERROR {rel}: {e}", file=sys.stderr)

    print(f"Done. Saved {len(image_keys)} images under {out_path}")


if __name__ == "__main__":
    main()
