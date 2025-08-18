"""
Configuration file for AFR vs VLM evaluation and verification.

This module contains all configuration settings including dataset paths,
model configurations, and system-specific settings for the face recognition
and vision-language model evaluation pipeline.

Author: Redwan Sony
PhD Student, iPRoBe Lab
Computer Science and Engineering
Michigan State University
"""

import os
from typing import List, Optional


# ============================================================================
# Base Directory Configuration
# ============================================================================

RSEARCH_DIR = "/mnt/research/iPRoBeLab/sonymd"
RESULTS_DIR = os.path.join(RSEARCH_DIR, "VLM-Benchmarking-Results")
MODEL_INFO_FILE = os.path.join(RESULTS_DIR, "model_param_counts.json")


# ============================================================================
# Dataset Root Directories
# ============================================================================

FR_DATASET_LOCATION = os.path.join(RSEARCH_DIR, "FR-Datasets")
IRIS_DATSET_LOCATION = os.path.join(RSEARCH_DIR, "Iris-Dataset/")


# ============================================================================
# Feature Extraction Output Directories
# ============================================================================

FEATURE_SAVE_DIR = os.path.join(RSEARCH_DIR, "extracted_features", "face_recognition")
FEATURE_SAVE_DIR_RACE = os.path.join(RSEARCH_DIR, "extracted_features", "race_classification")
FEATURE_SAVE_DIR_IRIS = os.path.join(RSEARCH_DIR, "extracted_features", "iris_recognition")


# ============================================================================
# Face Recognition Dataset Locations
# ============================================================================

LFW_DATASET_LOCATION = os.path.join(FR_DATASET_LOCATION, "lfw-dataset")
AGE_DB_DATASET_LOCATION = os.path.join(FR_DATASET_LOCATION, "agedb-dataset")
CFP_FP_DATASET_LOCATION = os.path.join(FR_DATASET_LOCATION, "cfp-dataset")
CFP_FP_DATASET_LOCATION_1024 = os.path.join(FR_DATASET_LOCATION, "cfp-dataset_1024")
CPLFW_DATASET_LOCATION = os.path.join(FR_DATASET_LOCATION, "cplfw-dataset")
VGGFACE2_DATASET_LOCATION = os.path.join(FR_DATASET_LOCATION, "VGGFace2")
AGEDB_CR_DATASET_LOCATION = os.path.join("/mnt/scratch/sonymd/FR-Datasets", "agedbcr-dataset")


# ============================================================================
# Dataset Choices Configuration
# ============================================================================

DATASET_CHOICES: List[str] = [
    "agedb_cr", 
    "lfw", 
    "cfp_fp_frontal", 
    "cfp_fp_profile", 
    "cplfw", 
    "agedb", 
    "vggface2", 
    "iris"
]


# ============================================================================
# Dynamic Dataset Path Configuration (Performance Optimization)
# ============================================================================

def _get_webface42m_location() -> str:
    """
    Determine the WebFace42M dataset location based on available paths.
    
    Returns:
        str: Path to WebFace42M dataset
        
    Raises:
        ValueError: If dataset location is not found
    """
    candidate_paths = [
        '/dev/shm/.data/webface42m',
        './.data/webface42m'
    ]
    
    for path in candidate_paths:
        if os.path.exists(path):
            return path
    
    raise ValueError("WebFace42M dataset location not found. Please set the correct path.")


def _get_lfw_location() -> str:
    """
    Determine the LFW dataset location based on available paths.
    
    Returns:
        str: Path to LFW dataset
        
    Raises:
        ValueError: If dataset location is not found
    """
    candidate_paths = [
        '/dev/shm/.data/LFW',
        './.data/LFW'
    ]
    
    for path in candidate_paths:
        if os.path.exists(path):
            return path
    
    raise ValueError("LFW dataset location not found. Please set the correct path.")


# Set dynamic dataset locations
WebFace42M_DATASET_LOCATION = _get_webface42m_location()
LFW_DATASET_LOCATION = _get_lfw_location()


# ============================================================================
# Hugging Face Authentication Configuration
# ============================================================================

def _load_huggingface_token() -> Optional[str]:
    """
    Load Hugging Face token from file.
    
    Returns:
        str or None: Hugging Face token if found, None otherwise
    """
    hf_token_file = os.path.expanduser("~/.huggingface_token")
    
    if os.path.exists(hf_token_file):
        try:
            with open(hf_token_file, "r") as f:
                line = f.readlines()[0]
                return line.strip().split()[-1]
        except (IndexError, IOError):
            return None
    
    return None


# Load Hugging Face token
HF_TOKEN: Optional[str] = _load_huggingface_token()
