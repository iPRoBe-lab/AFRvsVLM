"""
Face recognition dataset classes and utilities.

This module provides dataset classes for various face recognition benchmarks
including WebFace42M, LFW, IJB-B, IJB-C, and IJB-S. It handles image loading,
preprocessing, and provides a unified interface for dataset access.

Author: Redwan Sony
PhD Student, iPRoBe Lab
Computer Science and Engineering
Michigan State University
"""

import glob
import os
from typing import Dict, List, Optional, Union, Any

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import Compose, Normalize, ToTensor

from evaluation_verification.config import (
    LFW_DATASET_LOCATION,
    WebFace42M_DATASET_LOCATION,
)


# ============================================================================
# Dataset Classes
# ============================================================================

class WebFace42MDataset(Dataset):
    """
    Dataset class for WebFace42M face recognition dataset.
    
    Args:
        images_dir: Directory containing images
        df_file: CSV file with image metadata
        transform: Optional image transformations
    """
    
    def __init__(self, images_dir: str, df_file: str, transform: Optional[Any] = None):
        self.images_dir = images_dir
        self.df_file = df_file
        self.df = pd.read_csv(df_file)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, Union[torch.Tensor, str]]:
        image_name = self.df.at[idx, "files"]
        image_path = os.path.join(self.images_dir, image_name)
        image = Image.open(image_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return {"pixel_values": image, "image_name": image_name}


class LFWDataset(Dataset):
    """
    Dataset class for Labeled Faces in the Wild (LFW) dataset.
    
    Args:
        images_dir: Directory containing images
        df_file: CSV file with image metadata
        transform: Optional image transformations
    """
    
    def __init__(self, images_dir: str, df_file: str, transform: Optional[Any] = None):
        self.images_dir = images_dir
        self.df_file = df_file
        self.df = pd.read_csv(df_file)
        self.transform = transform
        print(f"Loading LFW dataset from {self.df_file} with {len(self.df)} images from {self.images_dir}.")

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, Union[torch.Tensor, str]]:
        image_name = self.df.at[idx, "files"]
        image_path = os.path.join(self.images_dir, image_name)
        image = Image.open(image_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return {"pixel_values": image, "image_name": image_name}


class IJBDataset(Dataset):
    """
    Dataset class for IJB (IARPA Janus Benchmark) datasets including IJB-B and IJB-C.
    
    Args:
        images_dir: Directory containing images
        df_file: Metadata file with facial landmarks and scores
        transform: Optional image transformations
        flip: Whether to apply horizontal flip augmentation
    """
    
    def __init__(self, images_dir: str, df_file: str, transform: Optional[Any] = None, flip: bool = False):
        self.images_dir = images_dir
        self.df_file = df_file
        self.df = None
        self._load_df()
        self.transform = transform
        self.flip = flip

    def _load_df(self) -> None:
        """Load the metadata dataframe with proper column names."""
        columns = ["image", "left_eye_x", "left_eye_y", "right_eye_x", "right_eye_y", "nose_x", "nose_y", 
                  "mouth_left_x", "mouth_left_y", "mouth_right_x", "mouth_right_y", "face_score"]
        self.df = pd.read_csv(self.df_file, sep=r"\s+", header=None, names=columns)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, Union[torch.Tensor, str]]:
        image_name = self.df.at[idx, "image"]
        image_path = os.path.join(self.images_dir, image_name)
        image = Image.open(image_path).convert("RGB")

        if self.flip:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)

        if self.transform:
            image = self.transform(image)

        return {"pixel_values": image, "image_name": image_name}


class IJBSDataset(Dataset):
    """
    Dataset class for IJB-S (IARPA Janus Benchmark - Surveillance) dataset.
    
    Args:
        images_dir: Directory containing images
        df_file: CSV file with image metadata
        transform: Optional image transformations
        flip: Whether to apply horizontal flip augmentation
    """
    
    def __init__(self, images_dir: str, df_file: str, transform: Optional[Any] = None, flip: bool = False):
        self.images_dir = images_dir
        self.df_file = df_file
        self.df = pd.read_csv(df_file)
        self.transform = transform
        self.flip = flip

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, Union[torch.Tensor, str]]:
        image_name = self.df.at[idx, "files"]
        image_path = os.path.join(self.images_dir, image_name)
        image = Image.open(image_path).convert("RGB")
        
        if self.flip:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)

        if self.transform:
            image = self.transform(image)
            
        return {"pixel_values": image, "image_name": image_name}


# ============================================================================
# Utility Functions
# ============================================================================

def collate_fn(batch: List[Dict[str, Union[torch.Tensor, str]]]) -> Dict[str, Union[torch.Tensor, List[str]]]:
    """
    Collate function for batching dataset items.
    
    Args:
        batch: List of dataset items
        
    Returns:
        Batched dictionary with pixel_values and image_names
    """
    pixel_values = [item["pixel_values"] for item in batch]
    image_names = [item["image_name"] for item in batch]

    # Stack tensors into a batch
    pixel_values = torch.stack(pixel_values)

    return {"pixel_values": pixel_values, "image_names": image_names}


# ============================================================================
# Dataset Factory Functions
# ============================================================================

def get_dataset_by_name(dataset_name: str, split: int = -1, flip: bool = False) -> Dataset:
    """
    Factory function to get dataset by name.
    
    Args:
        dataset_name: Name of the dataset ("webface42m", "LFW-112", "LFW-160", "LFW-250", "ijbb", "ijbc", "ijbs")
        split: Dataset split number (1-5 for WebFace42M, 1 for others)
        flip: Whether to apply horizontal flip augmentation
        
    Returns:
        Dataset instance
        
    Raises:
        ValueError: If dataset name is not supported or split is invalid
    """
    if dataset_name == "webface42m":
        if split < 1 or split > 5:
            raise ValueError("Split must be between 1 and 5 for WebFace42M dataset.")

        return WebFace42MDataset(images_dir=os.path.join(WebFace42M_DATASET_LOCATION, "images"),
                                df_file=os.path.join(".", "metadata/webface42m/verification", f"split_{split}.csv"))
    
    elif dataset_name == "LFW-250":
        assert split == 1, "LFW-250 dataset only has one split."
        images_dir = os.path.join(LFW_DATASET_LOCATION, "deepfunneled")
        print(f"Loading LFW dataset from {images_dir}")
        return LFWDataset(images_dir=images_dir, df_file=os.path.join(".", "metadata", "LFW/verification", "split_1.csv"))
        
    elif dataset_name == "LFW-160":
        assert split == 1, "LFW-160 dataset only has one split."
        images_dir = os.path.join(LFW_DATASET_LOCATION, "deepfunneled_cropped_160")
        print(f"Loading LFW dataset from {images_dir}")
        return LFWDataset(images_dir=images_dir, df_file=os.path.join(".", "metadata", "LFW/verification", "split_1.csv"))
        
    elif dataset_name == "LFW-112":
        assert split == 1, "LFW-112 dataset only has one split."
        images_dir = os.path.join(LFW_DATASET_LOCATION, "deepfunneled_cropped_112")
        print(f"Loading LFW dataset from {images_dir}")
        return LFWDataset(images_dir=images_dir, df_file=os.path.join(".", "metadata", "LFW/verification", "split_1.csv"))

    elif dataset_name == "ijbb":
        assert split == 1, "IJB-B dataset only has one split."
        if os.path.exists("/dev/shm/.data/ijb"):
            images_dir = "/dev/shm/.data/ijb/IJBB/loose_crop"
            df_file = "/dev/shm/.data/ijb/IJBB/meta/ijbb_name_5pts_score.txt"
        else:
            images_dir = "validation_mixed/insightface_ijb_helper/ijb/IJBB/loose_crop"
            df_file = "validation_mixed/insightface_ijb_helper/ijb/IJBB/meta/ijbb_name_5pts_score.txt"
        return IJBDataset(images_dir=images_dir, df_file=df_file, flip=flip)

    elif dataset_name == "ijbc":
        assert split == 1, "IJB-C dataset only has one split."
        if os.path.exists("/dev/shm/.data/ijb"):
            images_dir = "/dev/shm/.data/ijb/IJBC/loose_crop"
            df_file = "/dev/shm/.data/ijb/IJBC/meta/ijbc_name_5pts_score.txt"
        else:
            images_dir = "validation_mixed/insightface_ijb_helper/ijb/IJBC/loose_crop"
            df_file = "validation_mixed/insightface_ijb_helper/ijb/IJBC/meta/ijbc_name_5pts_score.txt"
        return IJBDataset(images_dir=images_dir, df_file=df_file, flip=flip)

    elif dataset_name == "ijbs":
        assert split == 1, "IJB-S dataset only has one split."
        if os.path.exists("/dev/shm/.data/IJBS"):
            images_dir = "/dev/shm/.data/ijb/IJBS/IJBS-Still"
            df_file = "/dev/shm/.data/ijb/IJBS/ijbs_still_metadata.csv"
        else:
            images_dir = ".data/IJBS/IJBS-Still"
            df_file = ".data/IJBS/ijbs_still_metadata.csv"
        return IJBSDataset(images_dir=images_dir, df_file=df_file)
    
    else:
        raise ValueError(f"Dataset {dataset_name} is not supported.")


def get_webface42m_dataloader(batch_size: int = 4, num_workers: int = 0, shuffle: bool = False, split: int = 1) -> DataLoader:
    """
    Get DataLoader for WebFace42M dataset with default transformations.
    
    Args:
        batch_size: Batch size for DataLoader
        num_workers: Number of worker processes for data loading
        shuffle: Whether to shuffle the dataset
        split: Dataset split number (1-5)
        
    Returns:
        DataLoader instance for WebFace42M dataset
    """
    tfmr_pipeline = Compose([ToTensor(), Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])

    webface_ds = WebFace42MDataset(images_dir=os.path.join(WebFace42M_DATASET_LOCATION, "images"),
                                  df_file=f"/research/iprobe-sonymd/AFRvsVLM/metadata/verification/split_{split}.csv",
                                  transform=tfmr_pipeline)

    dl = DataLoader(webface_ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, 
                   collate_fn=collate_fn, pin_memory=True)
    return dl


# ============================================================================
# Main Execution
# ============================================================================

if __name__ == "__main__":
    ds = get_dataset_by_name("ijbb", split=1)
