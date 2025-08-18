"""
Feature extraction and evaluation pipeline for AFR vs VLM comparison.

This module provides the main pipeline for extracting features from various
vision models and evaluating their performance on face recognition datasets.
It supports multiple model architectures including OpenCLIP, CLIP, BLIP,
LLaVA, SAM, ViT, DINO, and custom fine-tuned models.

Author: Redwan Sony
PhD Student, iPRoBe Lab
Computer Science and Engineering
Michigan State University
"""

import argparse
import os
import random
import sys
import time
from typing import Dict, List, Tuple, Union

import torch
from torch.nn.functional import adaptive_avg_pool2d, normalize
from torch.utils.data import DataLoader
from tqdm import tqdm

from .collate_functions import *
from .common import compute_tmr_at_fmr, create_ground_truth_matrix
from .feature_extractor_functions import *
from .fr_datasets import get_dataset_by_name
from .model_hub import load_model_by_name


# ============================================================================
# Model Configuration Mapping
# ============================================================================

def get_model_config(model_name: str) -> Dict[str, Union[str, bool]]:
    """
    Get the appropriate collate function and feature extractor for a model.
    
    Args:
        model_name: Name of the model
        
    Returns:
        Dictionary containing collate function name and feature extractor name
        
    Raises:
        ValueError: If model is not supported
    """
    model_configs = {
        # OpenCLIP models
        "LAION/OpenCLIP": {"collate_fn": "create_collate_function", "extractor": "extract_openclip_features"},
        
        # Face recognition models
        "adaface": {"collate_fn": "create_collate_function", "extractor": "extract_adaface_features"},
        "arcface": {"collate_fn": "create_collate_function", "extractor": "extract_adaface_features"},
        
        # CLIP models
        "openai/clip": {"collate_fn": "create_clip_collate_function", "extractor": "extract_clip_features", "use_cls": True},
        
        # BLIP models
        "Salesforce/blip2": {"collate_fn": "create_clip_collate_function", "extractor": "extract_clip_features", "use_cls": False},
        "Salesforce/blip": {"collate_fn": "create_clip_collate_function", "extractor": "extract_clip_features", "use_cls": False},
        
        # Align model
        "kakaobrain/align-base": {"collate_fn": "create_clip_collate_function", "extractor": "extract_align_features"},
        
        # LLaVA models
        "llava-hf/llava-1.5-7b-hf": {"collate_fn": "create_clip_collate_function", "extractor": "extract_llava_features"},
        "llava-hf/llava-v1.6-mistral-7b-hf": {"collate_fn": "create_llava_next_collate_function", "extractor": "extract_llava_next_features"},
        
        # SAM models
        "facebook/sam-vit": {"collate_fn": "create_sam_collate_function", "extractor": "extract_sam_features"},
        
        # Kosmos model
        "microsoft/kosmos-2-patch14-224": {"collate_fn": "create_kosmos_collate_function", "extractor": "extract_kosmos_features", "use_cls": True},
        
        # ViT models
        "google/vit-": {"collate_fn": "create_vit_collate_function", "extractor": "extract_vit_features", "use_cls": True},
        
        # DeepSeek models
        "deepseek-ai/deepseek-vl2": {"collate_fn": "create_deepseekvl2_collate_function", "extractor": "extract_deepseekvl2_features"},
        
        # InternVL models
        "OpenGVLab/InternVL3": {"collate_fn": "create_collate_function", "extractor": "extract_internvl3_features"},
        
        # DINO models
        "facebook/dino": {"collate_fn": "create_dino_collate_function", "extractor": "extract_dino_features"},
        
        # Fine-tuned models
        "FineTuned": {"collate_fn": "create_collate_function", "extractor": "extract_finetuned_vlm_features"}
    }
    
    # Find matching configuration
    for key, config in model_configs.items():
        if key in model_name:
            return config
    
    raise ValueError(f"Unsupported model: {model_name}. Please check the model name or add support for it.")


# ============================================================================
# Feature Extraction Pipeline
# ============================================================================

def setup_output_paths(args: argparse.Namespace) -> Tuple[str, str]:
    """
    Set up output directory and feature save path.
    
    Args:
        args: Command line arguments
        
    Returns:
        Tuple of (output_dir, feature_save_path)
    """
    output_dir = os.path.join(args.output_dir, "verification", args.model_name)
    feature_save_path = os.path.join(output_dir, f"split_{args.split}_features.pt")
    os.makedirs(os.path.dirname(feature_save_path), exist_ok=True)
    
    return output_dir, feature_save_path


def check_existing_features(feature_save_path: str) -> bool:
    """
    Check if features already exist at the specified path.
    
    Args:
        feature_save_path: Path to the feature file
        
    Returns:
        True if features exist, False otherwise
    """
    if os.path.exists(feature_save_path):
        print(f"📁 Feature file already exists at: {feature_save_path}")
        return True
    else:
        print(f"📁 Feature file does not exist at: {feature_save_path} -- extracting features.")
        return False


def extract_features_for_model(args: argparse.Namespace, model, processor) -> Tuple[torch.Tensor, List[str]]:
    """
    Extract features for a specific model.
    
    Args:
        args: Command line arguments
        model: The loaded model
        processor: The model's processor
        
    Returns:
        Tuple of (features, filenames)
    """
    print(f"🔍 Extracting features using model: {args.model_name} on dataset: {args.dataset_name}, split: {args.split}")
    
    # Load dataset
    ds = get_dataset_by_name(args.dataset_name, split=args.split, flip=args.flip)
    
    # Get model configuration
    config = get_model_config(args.model_name)
    
    # Create appropriate collate function
    collate_fn_name = config["collate_fn"]
    if collate_fn_name == "create_collate_function":
        collate_fn = create_collate_function(processor)
    elif collate_fn_name == "create_clip_collate_function":
        collate_fn = create_clip_collate_function(processor)
    elif collate_fn_name == "create_llava_next_collate_function":
        collate_fn = create_llava_next_collate_function(processor)
    elif collate_fn_name == "create_sam_collate_function":
        collate_fn = create_sam_collate_function(processor)
    elif collate_fn_name == "create_kosmos_collate_function":
        collate_fn = create_kosmos_collate_function(processor)
    elif collate_fn_name == "create_vit_collate_function":
        collate_fn = create_vit_collate_function(processor)
    elif collate_fn_name == "create_deepseekvl2_collate_function":
        collate_fn = create_deepseekvl2_collate_function(processor)
    elif collate_fn_name == "create_dino_collate_function":
        collate_fn = create_dino_collate_function(processor)
    else:
        raise ValueError(f"Unknown collate function: {collate_fn_name}")
    
    # Create DataLoader
    dl = DataLoader(ds, batch_size=args.batch_size, collate_fn=collate_fn, shuffle=False, num_workers=args.num_workers)
    
    # Extract features using appropriate extractor
    extractor_name = config["extractor"]
    if extractor_name == "extract_openclip_features":
        features, filenames = extract_openclip_features(model, dl, args.device)
    elif extractor_name == "extract_adaface_features":
        features, filenames = extract_adaface_features(model, dl, args.device)
    elif extractor_name == "extract_clip_features":
        use_cls = config.get("use_cls", True)
        features, filenames = extract_clip_features(model, dl, use_cls=use_cls)
    elif extractor_name == "extract_align_features":
        features, filenames = extract_align_features(model, dl)
    elif extractor_name == "extract_llava_features":
        features, filenames = extract_llava_features(model, dl)
    elif extractor_name == "extract_llava_next_features":
        features, filenames = extract_llava_next_features(model, dl)
    elif extractor_name == "extract_sam_features":
        features, filenames = extract_sam_features(model, dl)
    elif extractor_name == "extract_kosmos_features":
        use_cls = config.get("use_cls", True)
        features, filenames = extract_kosmos_features(model, dl, use_cls=use_cls)
    elif extractor_name == "extract_vit_features":
        use_cls = config.get("use_cls", True)
        features, filenames = extract_vit_features(model, dl, use_cls=use_cls)
    elif extractor_name == "extract_deepseekvl2_features":
        features, filenames = extract_deepseekvl2_features(model, dl, args.device)
    elif extractor_name == "extract_internvl3_features":
        features, filenames = extract_internvl3_features(model, dl)
    elif extractor_name == "extract_dino_features":
        features, filenames = extract_dino_features(model, dl, args.device)
    elif extractor_name == "extract_finetuned_vlm_features":
        features, filenames = extract_finetuned_vlm_features(model, dl, args.device)
    else:
        raise ValueError(f"Unknown feature extractor: {extractor_name}")
    
    return features, filenames


def save_features_and_similarity(features: torch.Tensor, feature_save_path: str, output_dir: str, args: argparse.Namespace) -> None:
    """
    Save features and compute similarity matrix if needed.
    
    Args:
        features: Extracted features
        feature_save_path: Path to save features
        output_dir: Output directory
        args: Command line arguments
    """
    # Save feature vectors
    print(f"💾 Saving features to: {feature_save_path}")
    
    # Wait random amount of time
    time_to_wait = random.uniform(3, 10)
    print(f"⏳ Waiting {time_to_wait:.2f} seconds before saving features...")
    time.sleep(time_to_wait)
    
    torch.save(features, feature_save_path)
    print(f"💾 Features saved to: {feature_save_path}")

    # Skip similarity computation for certain datasets
    if args.dataset_name in ["ijbb", "ijbc"]:
        return
    
    # Normalize and compute similarity matrix
    features = normalize(features, p=2, dim=1).to(args.device)
    cosine_sim_matrix = (features @ features.T).cpu()
    similarity_save_path = os.path.join(output_dir, f"split_{args.split}_score_matrix.pt")

    # Wait random amount of time before saving similarity matrix
    time_to_wait = random.uniform(3, 10)
    print(f"⏳ Waiting {time_to_wait:.2f} seconds before saving similarity matrix...")
    time.sleep(time_to_wait)

    torch.save(cosine_sim_matrix.half(), similarity_save_path)
    print(f"📐 Cosine similarity matrix saved to: {similarity_save_path}")


# ============================================================================
# Main Pipeline
# ============================================================================

def main(args: argparse.Namespace) -> None:
    """
    Main feature extraction and evaluation pipeline.
    
    Args:
        args: Command line arguments containing model, dataset, and processing parameters
    """
    # Setup output paths
    output_dir, feature_save_path = setup_output_paths(args)
    
    # Check if features already exist
    if check_existing_features(feature_save_path):
        return
    
    # Load model and processor
    model, processor = load_model_by_name(model_name=args.model_name, device=args.device)
    
    # Extract features
    features, filenames = extract_features_for_model(args, model, processor)
    
    # Save features and compute similarity matrix
    save_features_and_similarity(features, feature_save_path, output_dir, args)


# ============================================================================
# Command Line Interface
# ============================================================================

def parse_arguments() -> argparse.Namespace:
    """
    Parse command line arguments.
    
    Returns:
        Parsed arguments
    """
    parser = argparse.ArgumentParser(description="Evaluate model on dataset")
    
    # Model and dataset configuration
    parser.add_argument("--model_name", type=str, default="OpenCLIP-Huge", help="Name of the model to evaluate")
    parser.add_argument("--dataset_name", type=str, default="LFW-112", help="Name of the dataset to evaluate on")
    parser.add_argument("--split", type=int, default=1, help="Dataset split to use (1-5 for WebFace42M)")
    
    # Processing configuration
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for evaluation")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of workers for DataLoader")
    parser.add_argument("--flip", action="store_true", help="Whether to flip the images horizontally during evaluation")
    
    # Output configuration
    parser.add_argument("--output_dir", type=str, default="extracted_features", help="Directory to save extracted features")
    parser.add_argument("--verification_dir", type=str, default="extracted_features/LFW", help="Directory to save verification results")
    
    # Hardware configuration
    parser.add_argument("--device", type=str, default="cuda:0", help="Device to run the model on (e.g., 'cuda' or 'cpu')")
    
    return parser.parse_args()


def configure_output_directory(args: argparse.Namespace) -> None:
    """
    Configure output directory based on flip setting.
    
    Args:
        args: Command line arguments (modified in place)
    """
    if args.flip:
        print("🔄 Flipping images horizontally during evaluation.")
        args.output_dir = os.path.join(f"extracted_features/{args.dataset_name}_flipped")
    else:
        print("❌ Not flipping images during evaluation.")
        args.output_dir = os.path.join(f"extracted_features/{args.dataset_name}_unflipped")


if __name__ == "__main__":
    # Parse arguments
    args = parse_arguments()
    
    # Configure output directory
    configure_output_directory(args)
    
    # Set device
    args.device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    
    # Run main pipeline
    main(args)
