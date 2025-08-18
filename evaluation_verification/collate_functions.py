
"""
Collate functions for different model types in face recognition evaluation.

This module provides collate functions that prepare batches of data for various
vision models including DINO, ViT, CLIP, LLaVA, SAM, and others.

Author: Redwan Sony
PhD Student, iPRoBe Lab
Computer Science and Engineering
Michigan State University
"""

from typing import Any, Callable, Dict, List

import torch



# ============================================================================
# Vision Transformer (ViT) and DINO Model Collate Functions
# ============================================================================

def create_dino_collate_function(processor) -> Callable:
    """
    Create a collate function for DINO models.
    
    Args:
        processor: The DINO image processor
        
    Returns:
        A collate function that processes batches for DINO models
    """
    def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        loaded_images = [item["pixel_values"] for item in batch]
        processed_pixel_values = processor(images=loaded_images, return_tensors="pt")
        filenames = [item["image_name"] for item in batch]
        return {
            "pixel_values": processed_pixel_values['pixel_values'], 
            "filenames": filenames
        }
    return collate_fn


def create_vit_collate_function(processor) -> Callable:
    """
    Create a collate function for Vision Transformer (ViT) models.
    
    Args:
        processor: The ViT image processor
        
    Returns:
        A collate function that processes batches for ViT models
    """
    def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        loaded_images = [item["pixel_values"] for item in batch]
        processed_pixel_values = processor(images=loaded_images, return_tensors="pt")
        filenames = [item["image_name"] for item in batch]
        return {
            "pixel_values": processed_pixel_values['pixel_values'], 
            "filenames": filenames
        }
    return collate_fn


# ============================================================================
# Multi-Modal Model Collate Functions (CLIP, LLaVA, Kosmos)
# ============================================================================

def create_clip_collate_function(processor) -> Callable:
    """
    Create a collate function for CLIP models.
    
    Args:
        processor: The CLIP processor
        
    Returns:
        A collate function that processes batches for CLIP models
    """
    def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        loaded_images = [item["pixel_values"] for item in batch]
        processed_pixel_values = processor(
            images=loaded_images, 
            text=" " * len(loaded_images), 
            return_tensors="pt"
        )
        filenames = [item["image_name"] for item in batch]
        return {
            "pixel_values": processed_pixel_values['pixel_values'], 
            "filenames": filenames
        }
    return collate_fn


def create_kosmos_collate_function(processor) -> Callable:
    """
    Create a collate function for Kosmos models.
    
    Args:
        processor: The Kosmos processor
        
    Returns:
        A collate function that processes batches for Kosmos models
    """
    def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        loaded_images = [item["pixel_values"] for item in batch]
        processed_pixel_values = processor(
            images=loaded_images, 
            text=[" "] * len(loaded_images),  
            return_tensors="pt"
        )
        filenames = [item["image_name"] for item in batch]
        return {
            "pixel_values": processed_pixel_values['pixel_values'], 
            "filenames": filenames
        }
    return collate_fn


def create_llava_next_collate_function(processor) -> Callable:
    """
    Create a collate function for LLaVA-Next models.
    
    Args:
        processor: The LLaVA-Next processor
        
    Returns:
        A collate function that processes batches for LLaVA-Next models
    """
    def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        loaded_images = [item["pixel_values"] for item in batch]
        processed_input = processor(
            images=loaded_images, 
            text="What do you see?" * len(loaded_images), 
            return_tensors="pt"
        )
        filenames = [item["image_name"] for item in batch]
        return {
            'processed_input': processed_input, 
            'filenames': filenames
        }
    return collate_fn


# ============================================================================
# Specialized Model Collate Functions
# ============================================================================

def create_deepseekvl2_collate_function(processor) -> Callable:
    """
    Create a collate function for DeepSeek-VL2 models.
    
    Args:
        processor: The DeepSeek-VL2 processor
        
    Returns:
        A collate function that processes batches for DeepSeek-VL2 models
    """
    def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        processed_pixel_values = torch.stack([
            processor(item["pixel_values"]) for item in batch
        ])
        filenames = [item["image_name"] for item in batch]
        return {
            "pixel_values": processed_pixel_values, 
            "filenames": filenames
        }
    return collate_fn


def create_sam_collate_function(processor) -> Callable:
    """
    Create a collate function for Segment Anything Model (SAM).
    
    Args:
        processor: The SAM processor
        
    Returns:
        A collate function that processes batches for SAM models
    """
    def collate_fn_sam(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        raw_images = [item["pixel_values"] for item in batch]
        inputs = processor(raw_images, return_tensors="pt")
        filenames = [item["image_name"] for item in batch]
        return {
            "pixel_values": inputs['pixel_values'], 
            "filenames": filenames
        }
    return collate_fn_sam


# ============================================================================
# Generic Collate Function
# ============================================================================

def create_collate_function(processor) -> Callable:
    """
    Create a generic collate function for models with standard processors.
    
    Args:
        processor: The image processor
        
    Returns:
        A generic collate function that processes batches
    """
    def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        loaded_images = [processor(item["pixel_values"]) for item in batch]
        processed_pixel_values = torch.stack(loaded_images, dim=0)
        filenames = [item["image_name"] for item in batch]
        return {
            "pixel_values": processed_pixel_values, 
            "filenames": filenames
        }
    return collate_fn
