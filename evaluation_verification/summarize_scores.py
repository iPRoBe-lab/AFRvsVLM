"""
Summarize Verification Results Module

This module provides functionality to summarize face verification results across multiple models and dataset splits.
It processes TMR (True Match Rate) results at various FMR (False Match Rate) thresholds to generate comprehensive
performance summaries with statistical aggregation across splits.

Key Components:
- Result aggregation across dataset splits
- Statistical computation (mean and standard deviation)
- CSV report generation for performance analysis
- Support for both single-split (LFW) and multi-split datasets

Author: Redwan Sony
PhD Student, iPRoBe Lab
Computer Science and Engineering
Michigan State University
Last Modified: 2025
"""

# =================================================================================================
# IMPORTS
# =================================================================================================

import argparse
import os
from glob import glob
from typing import Dict, List, Any, Tuple

import pandas as pd


# =================================================================================================
# CORE EVALUATION FUNCTIONS
# =================================================================================================

def determine_dataset_splits(dataset_name: str) -> List[int]:
    """
    Determine the number of splits based on dataset name.
    
    Args:
        dataset_name (str): Name of the dataset to process
        
    Returns:
        List[int]: List of split numbers to process
    """
    if dataset_name.startswith("LFW"):
        return [1]  # LFW datasets use only split 1
    else:
        return [1, 2, 3, 4, 5]  # Standard 5-fold cross-validation


def discover_model_directories(dataset_name: str) -> List[str]:
    """
    Discover all model directories for the specified dataset.
    
    Args:
        dataset_name (str): Name of the dataset to scan for
        
    Returns:
        List[str]: List of model directory paths
    """
    base_path = f"extracted_features/{dataset_name}/verification/*/*"
    model_dirs = [x for x in glob(base_path) if os.path.isdir(x)]
    print(f"Found {len(model_dirs)} model directories.")
    return model_dirs


def extract_model_name(model_dir: str) -> str:
    """
    Extract readable model name from directory path.
    
    Args:
        model_dir (str): Full path to model directory
        
    Returns:
        str: Formatted model name
    """
    return os.path.join(os.path.basename(os.path.dirname(model_dir)), os.path.basename(model_dir))


def process_split_results(model_dir: str, split: int, tmr_by_fmr: Dict[float, List[float]]) -> None:
    """
    Process TMR results for a specific split and update the aggregation dictionary.
    
    Args:
        model_dir (str): Path to model directory
        split (int): Split number to process
        tmr_by_fmr (Dict[float, List[float]]): Dictionary to accumulate TMR values by FMR threshold
    """
    file_path = os.path.join(model_dir, f"split_{split}_tmr_results.csv")
    
    if not os.path.exists(file_path):
        return
        
    df = pd.read_csv(file_path)
    
    # Validate required columns
    if "FMR" not in df.columns or "TMR" not in df.columns:
        print(f"Skipping {file_path} as it does not contain required columns.")
        return
    
    # Clean FMR column - handle percentage format if present
    if df["FMR"].dtype == "object" and "%" in str(df["FMR"].values[0]):
        df["FMR"] = df["FMR"].astype(str).str.replace("%", "").astype(float) / 100
    else:
        df["FMR"] = df["FMR"].astype(float)
    
    # Extract TMR values for each FMR threshold
    for fmr in tmr_by_fmr:
        row = df[df["FMR"] == fmr]
        if not row.empty:
            tmr_by_fmr[fmr].append(row["TMR"].values[0])


def compute_statistics(values: List[float]) -> Tuple[float, float]:
    """
    Compute mean and standard deviation for TMR values.
    
    Args:
        values (List[float]): List of TMR values across splits
        
    Returns:
        Tuple[float, float]: Mean and standard deviation as percentages
    """
    mean = sum(values) / len(values) * 100
    std = pd.Series(values).std() * 100
    return mean, std


def format_tmr_result(values: List[float]) -> str:
    """
    Format TMR results for display with appropriate statistics.
    
    Args:
        values (List[float]): List of TMR values across splits
        
    Returns:
        str: Formatted result string
    """
    if not values:
        return "N/A"
    
    if len(values) == 1:
        tmr_pct = values[0] * 100
        return f"{tmr_pct:.2f}"
    else:
        mean, std = compute_statistics(values)
        return f"{mean:.2f} ± {std:.2f}"


def process_model_results(model_dir: str, splits: List[int]) -> Dict[str, Any]:
    """
    Process all results for a single model across specified splits.
    
    Args:
        model_dir (str): Path to model directory
        splits (List[int]): List of split numbers to process
        
    Returns:
        Dict[str, Any]: Dictionary containing model results
    """
    model_name = extract_model_name(model_dir)
    tmr_by_fmr = {0.001: [], 0.0001: [], 0.00001: [], 0.000001: []}
    
    # Process each split
    for split in splits:
        process_split_results(model_dir, split, tmr_by_fmr)
    
    # Compile results
    result = {"Model": model_name}
    for fmr, values in tmr_by_fmr.items():
        result[f"TMR@FMR={fmr}"] = format_tmr_result(values)
    
    return result


def create_summary_dataframe(summary: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Create and sort summary DataFrame with family grouping.
    
    Args:
        summary (List[Dict[str, Any]]): List of model result dictionaries
        
    Returns:
        pd.DataFrame: Sorted summary DataFrame
    """
    summary_df = pd.DataFrame(summary)
    
    # Add family column for logical grouping
    summary_df["family"] = summary_df["Model"].apply(lambda x: x.split("/")[0])
    summary_df = summary_df.sort_values(by=["family"], ascending=[True])
    
    return summary_df


def save_results(summary_df: pd.DataFrame, dataset_name: str) -> None:
    """
    Save summary results to CSV file.
    
    Args:
        summary_df (pd.DataFrame): Summary DataFrame to save
        dataset_name (str): Name of the dataset for filename
    """
    os.makedirs("results", exist_ok=True)
    output_path = f"results/verification_summary_{dataset_name}.csv"
    summary_df.to_csv(output_path, index=False)
    print(f"Results saved to: {output_path}")


# =================================================================================================
# MAIN EVALUATION PIPELINE
# =================================================================================================

def evaluate(args: argparse.Namespace) -> None:
    """
    Main evaluation function to summarize verification results across models and splits.
    
    This function:
    1. Discovers all model directories for the specified dataset
    2. Processes TMR results across all splits
    3. Computes statistical summaries (mean ± std) for multi-split datasets
    4. Generates formatted output and saves results to CSV
    
    Args:
        args (argparse.Namespace): Command line arguments containing dataset_name
    """
    dataset_name = args.dataset_name
    print(f"Processing verification results for dataset: {dataset_name}")
    
    # Determine splits based on dataset type
    splits = determine_dataset_splits(dataset_name)
    print(f"Processing splits: {splits}")
    
    # Discover model directories
    model_dirs = discover_model_directories(dataset_name)
    
    if not model_dirs:
        print(f"No model directories found for dataset: {dataset_name}")
        return
    
    # Process results for each model
    summary = []
    for model_dir in model_dirs:
        model_result = process_model_results(model_dir, splits)
        summary.append(model_result)
    
    # Create and display summary
    summary_df = create_summary_dataframe(summary)
    print("\n" + "="*100)
    print("VERIFICATION RESULTS SUMMARY")
    print("="*100)
    print(summary_df.to_string(index=False))
    print("="*100)
    
    # Save results
    save_results(summary_df, dataset_name)


# =================================================================================================
# COMMAND LINE INTERFACE
# =================================================================================================

def create_argument_parser() -> argparse.ArgumentParser:
    """
    Create and configure argument parser for command line interface.
    
    Returns:
        argparse.ArgumentParser: Configured argument parser
    """
    parser = argparse.ArgumentParser(
        description="Summarize verification results across models and splits",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python summarize_scores.py --dataset_name webface42m
    python summarize_scores.py --dataset_name LFW
    python summarize_scores.py --dataset_name IJB-B
        """
    )
    
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="webface42m",
        help="Name of the dataset to summarize results for (default: webface42m)"
    )
    
    return parser


# =================================================================================================
# MAIN EXECUTION
# =================================================================================================

if __name__ == "__main__":
    parser = create_argument_parser()
    args = parser.parse_args()
    evaluate(args)
