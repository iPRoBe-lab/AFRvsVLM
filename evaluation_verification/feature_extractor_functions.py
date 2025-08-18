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


def extract_dino_features(model, dataloader, use_cls=True):
    all_features, all_filenames = [], []
    num_batches = len(dataloader)

    model.eval()
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Extracting DINO Features", total=num_batches, file=sys.stdout):
            images = batch["pixel_values"].to(model.device)
            filenames = batch["filenames"]
            
            outputs = model(pixel_values=images)
            hidden_states = outputs.last_hidden_state.detach()  # [B, N, D]

            if use_cls:
                features = hidden_states[:, 0, :]  # CLS token
            else:
                features = hidden_states[:, 1:, :].mean(dim=1)  # Mean of patch embeddings

            all_features.append(features.cpu())
            all_filenames.extend(filenames)

    all_features = torch.cat(all_features, dim=0)
    return all_features, all_filenames



def extract_internvl3_features(vision_model, dataloader, dtype=torch.bfloat16):
    all_features, all_filenames = [], []
    num_batches = len(dataloader)

    vision_model.eval()
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Extracting Features", total=num_batches):
            pixel_values = batch["pixel_values"].to(vision_model.device).to(dtype)
            outputs = vision_model(pixel_values)
            features = outputs.last_hidden_state[:, 0, :]  # CLS token
            all_features.append(features.cpu())
            all_filenames.extend(batch["filenames"])

    all_features = torch.cat(all_features, dim=0)
    return all_features, all_filenames


def extract_deepseekvl2_features(vision_model, dataloader, device):
    """
    Extracts global features from images using the vision encoder and MLP projector of DeepSeek-VL2.

    Args:
        vision_model: The visual encoder (e.g., model.vision).
        projector: The MLP projector (e.g., model.projector).
        dataloader: Dataloader that returns image tensors and filenames.
        device: CUDA device (e.g., torch.device("cuda:0"))

    Returns:
        all_features: Tensor of shape [N, D] with extracted feature vectors (CLS token or projected).
        all_filenames: List of filenames corresponding to each feature vector.
    """
    all_features, all_filenames = [], []
    num_batches = len(dataloader)

    vision_model.eval()

    
    mean_pooling = True  
            

    with torch.no_grad(), torch.autocast(device_type=device.type, dtype=torch.float16 if device.type == "cuda" else torch.float32):
        for batch in tqdm(dataloader, desc="Extracting features", total=num_batches, file=sys.stdout):
            images = batch['pixel_values'].to(device)
            filenames = batch['filenames']

            
            visual_features = vision_model(images).detach()       # [B, T, D]
            if mean_pooling:
                visual_features = visual_features.mean(dim=1)
            else:
                visual_features = visual_features[:, 0, :] 
                
            all_features.append(visual_features.cpu())
            all_filenames.extend(filenames)

    all_features = torch.cat(all_features, dim=0)
    return all_features, all_filenames


def extract_vit_features(model, dataloader, use_cls=True):
    all_features, all_filenames = [], []
    num_batches = len(dataloader)

    model.eval()
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Extracting features", total=num_batches, file=sys.stdout):
            images = batch['pixel_values'].to(model.device) # [B, 3, H, W]
            all_filenames.extend(batch['filenames'])
            
            # Extract hidden states from the vision model
            vision_outputs = model(pixel_values=images)
            last_hidden_states = vision_outputs.hidden_states[-1]  # [B, N, D]

            if use_cls:
                features = last_hidden_states[:, 0, :]  # CLS token
            else:
                features = last_hidden_states[:, 1:, :].mean(dim=1)  # Mean of patch embeddings
            all_features.append(features.cpu())
            
        all_features = torch.cat(all_features, dim=0)  # shape: [num_images, dim]
        
        return all_features, all_filenames


def extract_finetuned_vlm_features(model, dataloader, device):
    all_features, all_filenames = [], []
    num_batches = len(dataloader)

    model.eval()
    model = model.to(device)
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Extracting features", total=num_batches, file=sys.stdout):
            images = batch['pixel_values'].to(device) # [B, 3, H, W]
            all_filenames.extend(batch['filenames'])
            
            # Extract hidden states from the vision model
            features, norms = model(images)
            all_features.append(features.detach().cpu())
            
        all_features = torch.cat(all_features, dim=0)  # shape: [num_images, dim]
        
        return all_features, all_filenames
    
    
    
def extract_openclip_features(model, dataloader, device):
    all_features, all_filenames = [], []
    num_batches = len(dataloader)

    model.eval()
    with torch.no_grad(), torch.autocast(device_type=device.type, 
                                         dtype=torch.float16 if device.type == "cuda" else torch.float32):
        for batch in tqdm(dataloader, desc="Extracting OpenCLIP features", total=num_batches, file=sys.stdout):
            pixel_values = batch["pixel_values"].to(device)

            features = model.encode_image(pixel_values).detach()
            features = features / features.norm(dim=-1, keepdim=True)

            all_features.append(features.cpu())
            all_filenames.extend(batch["filenames"])

    return torch.cat(all_features, dim=0), all_filenames


def extract_clip_features(clip_model, dataloader, use_cls=True):
    all_features, all_filenames = [], []
    num_batches = len(dataloader)

    clip_model.eval()
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Extracting features", total=num_batches, file=sys.stdout):
            images = batch['pixel_values'].to(clip_model.device) # [B, 3, H, W]
            all_filenames.extend(batch['filenames'])
                        
            vision_outputs = clip_model.vision_model(images)
            hidden_states = vision_outputs.last_hidden_state  # [B, N, D]

            if use_cls:
                features = hidden_states[:, 0, :]  # CLS token
            else:
                features = hidden_states[:, 1:, :].mean(dim=1)  # Mean of patch embeddings
            all_features.append(features.cpu())
            
        all_features = torch.cat(all_features, dim=0)  # shape: [num_images, dim]
        
        return all_features, all_filenames


def extract_adaface_features(model, dataloader, device):
    all_features, all_filenames = [], []
    num_batches = len(dataloader)

    model.eval()
    model = model.to(device)
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Extracting Adaface features", total=num_batches, file=sys.stdout):
            pixel_values = batch["pixel_values"].to(device)

            features = model(pixel_values).detach()
            all_features.append(features.cpu())
            all_filenames.extend(batch["filenames"])

    return torch.cat(all_features, dim=0), all_filenames
    
def extract_align_features(model, dataloader):
    all_features, all_filenames = [], []
    num_batches = len(dataloader)

    model.eval()
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Extracting Align Features", total=num_batches, file=sys.stdout):
            images = batch["pixel_values"].to(model.device)
            filenames = batch["filenames"]
            
            image_embeds = model.get_image_features(pixel_values=images).detach()  # [B, N, D]

            all_features.append(image_embeds.cpu())
            all_filenames.extend(filenames)

    all_features = torch.cat(all_features, dim=0)
    
    return all_features, all_filenames

def extract_llava_next_features(model, dataloader):
    all_features, all_filenames = [], []
    num_batches = len(dataloader)
    model.eval()
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Extracting LLaVA Features", total=num_batches, file=sys.stdout):
            processed_inputs, filenames = batch['processed_input'], batch['filenames']
            pixel_values = processed_inputs["pixel_values"].to(model.device)
            image_sizes = processed_inputs["image_sizes"].to(model.device)
            
            image_features_list = model.get_image_features(
                pixel_values=pixel_values,
                image_sizes=image_sizes,
                vision_feature_layer=model.config.vision_feature_layer,
                vision_feature_select_strategy='full',
                )
            
            image_features_mean_list = [x[:, 0, :].mean(dim=0) for x in image_features_list]
            image_features_mean = torch.stack(image_features_mean_list).detach()
            all_features.append(image_features_mean.cpu())
            all_filenames.extend(filenames)

    all_features = torch.cat(all_features, dim=0)
    return all_features, all_filenames

def extract_llava_features(model, dataloader):
    all_features, all_filenames = [], []
    num_batches = len(dataloader)

    model.eval()
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Extracting LLaVA Features", total=num_batches, file=sys.stdout):
            pixel_values = batch["pixel_values"].to(model.device)
            vision_outputs = model.vision_tower(pixel_values=pixel_values, output_hidden_states=True, return_dict=True)
            outputs = vision_outputs.hidden_states[-1][:, 0, :].detach()

            all_features.append(outputs.cpu())
            all_filenames.extend(batch["filenames"])

    all_features = torch.cat(all_features, dim=0)
    return all_features, all_filenames


def extract_sam_features(model, dataloader):
    all_features, all_filenames = [], []
    total_batches = len(dataloader) 

    model.eval()
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Extracting SAM features", total=total_batches, file=sys.stdout):
            pixel_inputs = batch["pixel_values"].to(model.device)
            filenames = batch["filenames"]

            image_embeddings = model.get_image_embeddings(pixel_inputs).detach()
            image_feature_vectors = adaptive_avg_pool2d(image_embeddings, output_size=1)
            image_feature_vectors = image_feature_vectors.view(image_feature_vectors.size(0), -1)
            all_features.append(image_feature_vectors.cpu())
            all_filenames.extend(filenames)

    return torch.cat(all_features, dim=0), all_filenames


def extract_kosmos_features(model, dataloader, use_cls=True):
    all_features, all_filenames = [], []
    num_batches = len(dataloader)

    model.eval()
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Extracting features", total=num_batches, file=sys.stdout):
            images = batch['pixel_values'].to(model.device) # [B, 3, H, W]
            all_filenames.extend(batch['filenames'])
                        
            vision_outputs = model.vision_model(images)
            hidden_states = vision_outputs.last_hidden_state  # [B, N, D]

            if use_cls:
                features = hidden_states[:, 0, :]  # CLS token
            else:
                features = hidden_states[:, 1:, :].mean(dim=1)  # Mean of patch embeddings
            all_features.append(features.cpu())
            
        all_features = torch.cat(all_features, dim=0)  # shape: [num_images, dim]
        
        return all_features, all_filenames
