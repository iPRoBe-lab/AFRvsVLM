import argparse
import os
from glob import glob

import pandas as pd


def evaluate(args):
    DATASET_NAME = args.dataset_name
    # Identify if dataset uses only split 1 or multiple splits
    if DATASET_NAME.startswith("LFW"):
        splits = [1]  # Only split 1
    else:
        splits = [1, 2, 3, 4, 5]  # 5 splits

    # Base directory containing model folders
    base_path = f"extracted_features/{DATASET_NAME}/verification/*/*"
    model_dirs = [x for x in glob(base_path) if os.path.isdir(x)]

    print(f"Found {len(model_dirs)} model directories.")

    summary = []

    for model_dir in model_dirs:
        model_name = os.path.join(
            os.path.basename(os.path.dirname(model_dir)), os.path.basename(model_dir)
        )
        tmr_by_fmr = {0.001: [], 0.0001: [], 0.00001: [], 0.000001: []}

        # Read result for each split
        for split in splits:
            file_path = os.path.join(model_dir, f"split_{split}_tmr_results.csv")
            if not os.path.exists(file_path):
                continue
            df = pd.read_csv(file_path)
            # Clean up FMR column if it contains percentage signs in FMR column values
            if "FMR" not in df.columns or "TMR" not in df.columns:
                print(f"Skipping {file_path} as it does not contain required columns.")
                continue
            if df["FMR"].dtype == "object" and "%" in df["FMR"].values[0]:
                df["FMR"] = (
                    df["FMR"].astype(str).str.replace("%", "").astype(float) / 100
                )
            else:
                df["FMR"] = df["FMR"].astype(float)
            for fmr in tmr_by_fmr:
                row = df[df["FMR"] == fmr]
                if not row.empty:
                    tmr_by_fmr[fmr].append(row["TMR"].values[0])

        # Compute results
        result = {"Model": model_name}
        for fmr, values in tmr_by_fmr.items():
            if values:
                if len(values) == 1:
                    tmr_pct = values[0] * 100
                    result[f"TMR@FMR={fmr}"] = f"{tmr_pct:.2f}"
                else:
                    mean = sum(values) / len(values) * 100
                    std = pd.Series(values).std() * 100
                    result[f"TMR@FMR={fmr}"] = f"{mean:.2f} ± {std:.2f}"
            else:
                result[f"TMR@FMR={fmr}"] = "N/A"
        summary.append(result)

    # Convert to DataFrame
    summary_df = pd.DataFrame(summary)

    # Add family column for sorting
    summary_df["family"] = summary_df["Model"].apply(lambda x: x.split("/")[0])
    summary_df = summary_df.sort_values(
        by=["family"],
        ascending=[
            True,
        ],
    )

    # Output
    print(summary_df)

    # Save CSV
    os.makedirs("results", exist_ok=True)
    summary_df.to_csv(f"results/verification_summary_{DATASET_NAME}.csv", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Summarize verification results across models and splits"
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="webface42m",
        help="Name of the dataset to summarize results for",
    )
    args = parser.parse_args()

    evaluate(args)
