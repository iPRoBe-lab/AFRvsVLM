"""
Common utility functions for AFR vs VLM evaluation and verification.

This module provides shared functionality for feature extraction, ground truth
matrix creation, and evaluation metrics computation including TMR at FMR
and ROC curve analysis.

Author: Redwan Sony
PhD Student, iPRoBe Lab
Computer Science and Engineering
Michigan State University
"""

from typing import List, Tuple, Union
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import auc, roc_curve
from tqdm import tqdm


# ============================================================================
# Feature Extraction Utilities
# ============================================================================

@torch.no_grad()
def extract_features(model: torch.nn.Module, dataloader: torch.utils.data.DataLoader) -> torch.Tensor:
    """
    Extract features from a model using a dataloader.

    Args:
        model: The model to extract features from
        dataloader: A PyTorch DataLoader providing the input data

    Returns:
        Concatenated tensor of extracted features
    """
    model.eval()
    features = []
    device = next(model.parameters()).device

    for batch in tqdm(dataloader, total=len(dataloader), desc="Extracting features"):
        inputs = batch["pixel_values"].to(device)
        outputs = model(inputs).detach()  # Extract features without gradients
        features.append(outputs.cpu())

    return torch.cat(features, dim=0)


# ============================================================================
# Ground Truth Matrix Creation
# ============================================================================

def create_ground_truth_matrix(split_file: str) -> torch.Tensor:
    """
    Create a ground truth matrix from a split file.
    
    Args:
        split_file: Path to CSV file containing image files and ID columns
        
    Returns:
        Boolean tensor matrix where True indicates matching IDs
    """
    # Load the CSV
    df = pd.read_csv(split_file)

    # Total number of images
    image_paths = df["files"].tolist()
    ids = df["ID"].tolist()
    N = len(image_paths)

    # Create ground truth label matrix
    gt_matrix = np.zeros((N, N), dtype=int)

    # Fill with 1 where ID matches
    for i in tqdm(range(N), desc="Creating ground truth matrix"):
        for j in range(N):
            if ids[i] == ids[j]:
                gt_matrix[i, j] = 1

    gt_matrix = torch.tensor(gt_matrix, dtype=torch.bool)
    return gt_matrix


# ============================================================================
# Evaluation Metrics
# ============================================================================

@torch.no_grad()
def compute_tmr_at_fmr(score_matrix: torch.Tensor, gt_matrix: torch.Tensor, fmr_targets: List[float] = [0.01, 0.001, 0.0001]) -> List[Tuple[float, float, float]]:
    """
    Compute TMR (True Match Rate) at given FMR (False Match Rate) targets.
    
    Args:
        score_matrix: Similarity score matrix of shape (N, N)
        gt_matrix: Ground truth matrix of shape (N, N) with 1 for genuine pairs, 0 for impostor pairs
        fmr_targets: List of FMR thresholds to compute TMR at
        
    Returns:
        List of tuples containing (FMR, TMR, threshold) for each FMR target
    """
    # Flatten matrices
    scores = score_matrix.view(-1)
    labels = gt_matrix.view(-1)

    # Separate genuine and impostor scores
    genuine_scores = scores[labels == 1]
    impostor_scores = scores[labels == 0]

    # Sort impostor scores descending
    sorted_impostor_scores, _ = torch.sort(impostor_scores, descending=True)
    num_impostors = len(impostor_scores)

    # Precompute thresholds for each FMR
    thresholds = []
    for fmr in fmr_targets:
        idx = int(fmr * num_impostors)
        idx = max(idx, 1)  # Avoid index 0
        thresholds.append(sorted_impostor_scores[idx - 1])

    thresholds = torch.tensor(thresholds, device=scores.device)

    # Broadcast to compute TMRs at all thresholds efficiently
    tmr_values = [(genuine_scores >= t).float().mean().item() for t in thresholds]

    return list(zip(fmr_targets, tmr_values, thresholds.cpu().tolist()))


def compute_roc(score_matrix: torch.Tensor, gt_matrix: torch.Tensor) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute ROC curve metrics from score and ground truth matrices.
    
    Args:
        score_matrix: Similarity score matrix of shape (N, N)
        gt_matrix: Ground truth matrix of shape (N, N)
        
    Returns:
        Tuple of (false_positive_rates, true_positive_rates, thresholds)
    """
    # Flatten matrices
    scores = score_matrix.view(-1).cpu().numpy()
    labels = gt_matrix.view(-1).cpu().numpy()

    # Compute FPR, TPR, thresholds
    fprs, tprs, thresholds = roc_curve(labels, scores)
    
    return fprs, tprs, thresholds
