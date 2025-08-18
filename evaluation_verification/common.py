import numpy as np
import pandas as pd
import torch
from sklearn.metrics import auc, roc_curve
from tqdm import tqdm


@torch.no_grad()
def extract_features(model, dataloader):
    """
    Extract features from a model using a dataloader.

    Args:
        model: The model to extract features from.
        dataloader: A PyTorch DataLoader providing the input data.

    Returns:
        A list of extracted features.
    """
    model.eval()
    features = []

    device = next(model.parameters()).device

    for batch in tqdm(dataloader, total=len(dataloader), desc="Extracting features"):
        inputs = batch["pixel_values"].to(device)
        outputs = model(inputs).detach()  # Extract features without gradients
        features.append(outputs.cpu())

    return torch.cat(features, dim=0)


def create_ground_truth_matrix(split_file):
    # Load the CSV
    df = pd.read_csv(split_file)

    # Total number of images
    image_paths = df["files"].tolist()
    ids = df["ID"].tolist()
    N = len(image_paths)

    # Create ground truth label matrix
    gt_matrix = np.zeros((N, N), dtype=int)

    # Fill with 1 where ID matches
    for i in tqdm(range(N), desc="Ceating ground truth matrix"):
        for j in range(N):
            if ids[i] == ids[j]:
                gt_matrix[i, j] = 1

    gt_matrix = torch.tensor(gt_matrix, dtype=torch.bool)
    return gt_matrix


# @torch.no_grad()
# def compute_tmr_at_fmr(score_matrix, gt_matrix, fmr_targets=[0.01, 0.001, 0.0001]):
#     # Flatten
#     scores = score_matrix.view(-1)
#     labels = gt_matrix.view(-1)

#     # Separate genuine and impostor scores
#     genuine_scores = scores[labels == 1]
#     impostor_scores = scores[labels == 0]

#     # Sort impostor scores descending
#     sorted_impostor_scores, _ = torch.sort(impostor_scores, descending=True)
#     num_impostors = len(impostor_scores)

#     # Precompute thresholds for each FMR
#     thresholds = []
#     for fmr in fmr_targets:
#         idx = int(fmr * num_impostors)
#         idx = max(idx, 1)  # Avoid index 0
#         thresholds.append(sorted_impostor_scores[idx - 1])

#     thresholds = torch.tensor(thresholds, device=scores.device)

#     # Broadcast to compute TMRs at all thresholds efficiently
#     tmr_values = [(genuine_scores >= t).float().mean().item() for t in thresholds]

#     return list(zip(fmr_targets, tmr_values, thresholds.cpu().tolist()))


@torch.no_grad()
# def compute_tmr_at_fmr(score_matrix, gt_matrix, fmr_targets=[0.01, 0.001, 0.0001]):
#     """
#     Compute TMR at given FMR targets from pairwise score and ground truth matrices.
#     Only unique off-diagonal pairs are considered (upper triangle, excluding diagonal).

#     Args:
#         score_matrix: (N, N) similarity score matrix (symmetric).
#         gt_matrix: (N, N) ground truth matrix (1: genuine, 0: impostor).
#         fmr_targets: list of FMR thresholds to compute TMR at.

#     Returns:
#         List of tuples: (FMR, TMR, threshold) for each FMR target.
#     """
#     N = score_matrix.shape[0]
#     device = score_matrix.device

#     # Extract upper triangular (excluding diagonal)
#     tri_upper = torch.triu_indices(N, N, offset=1, device=device)

#     scores = score_matrix[tri_upper[0], tri_upper[1]]
#     labels = gt_matrix[tri_upper[0], tri_upper[1]]

#     # Separate genuine and impostor scores
#     genuine_scores = scores[labels == 1]
#     impostor_scores = scores[labels == 0]

#     # Sort impostor scores in descending order
#     sorted_impostor_scores, _ = torch.sort(impostor_scores, descending=True)
#     num_impostors = len(impostor_scores)

#     # Compute thresholds for each FMR target
#     thresholds = []
#     for fmr in fmr_targets:
#         idx = int(fmr * num_impostors)
#         idx = max(idx, 1)  # Ensure at least one sample
#         thresholds.append(sorted_impostor_scores[idx - 1])

#     thresholds = torch.stack(thresholds)

#     # Compute TMR at each threshold
#     tmr_values = (genuine_scores[:, None] >= thresholds[None, :]).float().mean(dim=0).tolist()

#     # Prepare output
#     results = []
#     for fmr, tmr, thresh in zip(fmr_targets, tmr_values, thresholds.cpu().tolist()):
#         results.append((fmr, tmr, thresh))

#     return results


def compute_tmr_at_fmr(score_matrix, gt_matrix, fmr_targets=[0.01, 0.001, 0.0001]):
    # Flatten
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


def compute_roc(score_matrix, gt_matrix):
    # Flatten matrices
    scores = score_matrix.view(-1).cpu().numpy()
    labels = gt_matrix.view(-1).cpu().numpy()

    # Compute FPR, TPR, thresholds
    fprs, tprs, thresholds = roc_curve(labels, scores)
    roc_auc = auc(fprs, tprs)

    return fprs, tprs, thresholds
