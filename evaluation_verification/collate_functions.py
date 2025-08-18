
import argparse
import os
import random
import sys
import time

import torch
from torch.nn.functional import adaptive_avg_pool2d, normalize
from torch.utils.data import DataLoader
from tqdm import tqdm

from .common import compute_tmr_at_fmr, create_ground_truth_matrix
from .fr_datasets import get_dataset_by_name
from .model_hub import load_model_by_name


def create_dino_collate_function(processor):
    # Here is the collate function for DINO
    def collate_fn(batch):
        loaded_images = [item["pixel_values"] for item in batch]
        processed_pixel_values = processor(images=loaded_images, return_tensors="pt")
        filenames = [item["image_name"] for item in batch]
        return {"pixel_values": processed_pixel_values['pixel_values'], "filenames": filenames}
    return collate_fn

def create_deepseekvl2_collate_function(processor):
    def collate_fn(batch):
        processed_pixel_values = torch.stack([processor(item["pixel_values"]) for item in batch])
        filenames = [item["image_name"] for item in batch]
        return {"pixel_values": processed_pixel_values, "filenames": filenames}
    return collate_fn



def create_vit_collate_function(processor):
    def collate_fn(batch):
        loaded_images = [item["pixel_values"] for item in batch]
        processed_pixel_values = processor(images=loaded_images, return_tensors="pt")
        filenames = [item["image_name"] for item in batch]
        return {"pixel_values": processed_pixel_values['pixel_values'], "filenames": filenames}
    return collate_fn


def create_collate_function(processor):
    def collate_fn(batch):
        loaded_images = [processor(item["pixel_values"]) for item in batch]
        processed_pixel_values = torch.stack(loaded_images, dim=0)
        filenames = [item["image_name"] for item in batch]
        return {"pixel_values": processed_pixel_values, "filenames": filenames}
    return collate_fn

def create_kosmos_collate_function(processor):
    def collate_fn(batch):
        loaded_images = [item["pixel_values"] for item in batch]
        # print(f"Loaded {len(loaded_images)} images")
        processed_pixel_values = processor(images=loaded_images, text= [" ",] *len(loaded_images),  return_tensors="pt")
        filenames = [item["image_name"] for item in batch]
        return {"pixel_values": processed_pixel_values['pixel_values'], "filenames": filenames}
    return collate_fn


def create_clip_collate_function(processor):
    def collate_fn(batch):
        loaded_images = [item["pixel_values"] for item in batch]
        processed_pixel_values = processor(images=loaded_images, text= " "*len(loaded_images), return_tensors="pt")
        filenames = [item["image_name"] for item in batch]
        return {"pixel_values": processed_pixel_values['pixel_values'], "filenames": filenames}
    return collate_fn

def create_llava_next_collate_function(processor):
    def collate_fn(batch):
        loaded_images = [item["pixel_values"] for item in batch]
        processed_input = processor(images=loaded_images, 
                                           text= "What do you see?"*len(loaded_images), 
                                           return_tensors="pt")
        filenames = [item["image_name"] for item in batch]
        return {'processed_input': processed_input, 'filenames': filenames}
    return collate_fn


def create_sam_collate_function(processor):
    def collate_fn_sam(batch):
        raw_images = [item["pixel_values"] for item in batch]
        inputs = processor(raw_images, return_tensors="pt")
        filenames = [item["image_name"] for item in batch]
        return {"pixel_values": inputs['pixel_values'], 
                "filenames": filenames}
    return collate_fn_sam
