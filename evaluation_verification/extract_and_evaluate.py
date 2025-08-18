import argparse
import os
import random
import sys
import time

import torch
from torch.nn.functional import adaptive_avg_pool2d, normalize
from torch.utils.data import DataLoader
from tqdm import tqdm

from .collate_functions import *
from .common import compute_tmr_at_fmr, create_ground_truth_matrix
from .feature_extractor_functions import *
from .fr_datasets import get_dataset_by_name
from .model_hub import load_model_by_name


def main(args):
    # wait random amount of time before starting
    # time_to_wait = random.uniform(10, 3)
    # print(f"⏳ Waiting {time_to_wait:.2f} seconds before starting...")
    # time.sleep(time_to_wait)

    # model, processor = load_model_by_name(model_name=args.model_name, device=args.device)

    # ds = get_dataset_by_name(args.dataset_name, split=args.split)

    # Define output paths
    output_dir = os.path.join(args.output_dir, "verification", args.model_name)
    feature_save_path = os.path.join(output_dir, f"split_{args.split}_features.pt")
    os.makedirs(os.path.dirname(feature_save_path), exist_ok=True)

    if os.path.exists(feature_save_path):
        print(f"📁 Feature file already exists at: {feature_save_path}")
        # try:
        #     # features = torch.load(feature_save_path)
        #     print(f"📁 Features already exist at: {feature_save_path}   -- loading them.")
        # except Exception as e:
        #     print(f"❌ Error loading features: {e}")
        #     features = None
        features = True
    else:
        print(
            f"📁 Feature file does not exist at: {feature_save_path}   -- extracting features."
        )
        features = None

    if features is None:
        print(
            f"🔍 Extracting features using model: {args.model_name} on dataset: {args.dataset_name}, split: {args.split}"
        )

        ds = get_dataset_by_name(args.dataset_name, split=args.split, flip=args.flip)
        model, processor = load_model_by_name(
            model_name=args.model_name, device=args.device
        )

        if args.model_name.startswith("LAION/OpenCLIP"):
            dl = DataLoader(
                ds,
                batch_size=args.batch_size,
                collate_fn=create_collate_function(processor),
                shuffle=False,
                num_workers=args.num_workers,
            )
            features, filenames = extract_openclip_features(model, dl, args.device)

        elif "adaface" in args.model_name:
            dl = DataLoader(
                ds,
                batch_size=args.batch_size,
                collate_fn=create_collate_function(processor),
                shuffle=False,
                num_workers=args.num_workers,
            )
            features, filenames = extract_adaface_features(model, dl, args.device)

        elif "arcface" in args.model_name:
            dl = DataLoader(
                ds,
                batch_size=args.batch_size,
                collate_fn=create_collate_function(processor),
                shuffle=False,
                num_workers=args.num_workers,
            )
            features, filenames = extract_adaface_features(model, dl, args.device)

        elif "openai/clip" in args.model_name:
            dl = DataLoader(
                ds,
                batch_size=args.batch_size,
                collate_fn=create_clip_collate_function(processor),
                shuffle=False,
                num_workers=args.num_workers,
            )
            features, filenames = extract_clip_features(model, dl, use_cls=True)

        elif (
            "Salesforce/blip2" in args.model_name
            or "Salesforce/blip" in args.model_name
        ):
            dl = DataLoader(
                ds,
                batch_size=args.batch_size,
                collate_fn=create_clip_collate_function(processor),
                shuffle=False,
                num_workers=args.num_workers,
            )
            features, filenames = extract_clip_features(model, dl, use_cls=False)

        elif args.model_name == "kakaobrain/align-base":
            dl = DataLoader(
                ds,
                batch_size=args.batch_size,
                collate_fn=create_clip_collate_function(processor),
                shuffle=False,
                num_workers=args.num_workers,
            )
            features, filenames = extract_align_features(model, dl)

        elif args.model_name == "llava-hf/llava-1.5-7b-hf":
            dl = DataLoader(
                ds,
                batch_size=args.batch_size,
                collate_fn=create_clip_collate_function(processor),
                shuffle=False,
                num_workers=args.num_workers,
            )
            features, filenames = extract_llava_features(model, dl)

        elif args.model_name == "llava-hf/llava-v1.6-mistral-7b-hf":
            dl = DataLoader(
                ds,
                batch_size=args.batch_size,
                collate_fn=create_llava_next_collate_function(processor),
                shuffle=False,
                num_workers=args.num_workers,
            )
            features, filenames = extract_llava_next_features(model, dl)

        elif args.model_name.startswith("facebook/sam-vit"):
            dl = DataLoader(
                ds,
                batch_size=args.batch_size,
                collate_fn=create_sam_collate_function(processor),
                shuffle=False,
                num_workers=args.num_workers,
            )
            features, filenames = extract_sam_features(model, dl)

        elif args.model_name == "microsoft/kosmos-2-patch14-224":
            dl = DataLoader(
                ds,
                batch_size=args.batch_size,
                collate_fn=create_kosmos_collate_function(processor),
                shuffle=False,
                num_workers=args.num_workers,
            )
            features, filenames = extract_kosmos_features(model, dl, use_cls=True)

        elif args.model_name.startswith("google/vit-"):
            dl = DataLoader(
                ds,
                batch_size=args.batch_size,
                collate_fn=create_vit_collate_function(processor),
                shuffle=False,
                num_workers=args.num_workers,
            )
            features, filenames = extract_vit_features(model, dl, use_cls=True)

        elif args.model_name.startswith("deepseek-ai/deepseek-vl2"):
            dl = DataLoader(
                ds,
                batch_size=args.batch_size,
                collate_fn=create_deepseekvl2_collate_function(processor),
                shuffle=False,
                num_workers=args.num_workers,
            )
            features, filenames = extract_deepseekvl2_features(model, dl, args.device)

        elif args.model_name.startswith("OpenGVLab/InternVL3"):
            dl = DataLoader(
                ds,
                batch_size=args.batch_size,
                collate_fn=create_collate_function(processor),
                shuffle=False,
                num_workers=args.num_workers,
            )
            features, filenames = extract_internvl3_features(model, dl)
        elif args.model_name.startswith("facebook/dino"):
            dl = DataLoader(
                ds,
                batch_size=args.batch_size,
                collate_fn=create_dino_collate_function(processor),
                shuffle=False,
                num_workers=args.num_workers,
            )
            features, filenames = extract_dino_features(model, dl, args.device)

        elif args.model_name.startswith("FineTuned"):
            dl = DataLoader(
                ds,
                batch_size=args.batch_size,
                collate_fn=create_collate_function(processor),
                shuffle=False,
                num_workers=args.num_workers,
            )
            features, filenames = extract_finetuned_vlm_features(model, dl, args.device)

        else:
            raise ValueError(
                f"Unsupported model: {args.model_name}. Please check the model name or add support for it."
            )

        # Save feature vectors
        print(f"💾 Saving features to: {feature_save_path}")
        # Wait random amount of time
        time_to_wait = random.uniform(10, 3)
        print(f"⏳ Waiting {time_to_wait:.2f} seconds before saving features...")
        time.sleep(time_to_wait)
        torch.save(features, feature_save_path)
        print(f"💾 Features saved to: {feature_save_path}")

        if args.dataset_name in ["ijbb", "ijbc"]:
            return
        else:
            # Normalize and compute similarity matrix
            features = normalize(features, p=2, dim=1).to(args.device)
            cosine_sim_matrix = (features @ features.T).cpu()
            similarity_save_path = os.path.join(
                output_dir, f"split_{args.split}_score_matrix.pt"
            )

            # wait random amount of time before saving similarity matrix
            time_to_wait = random.uniform(10, 3)
            print(
                f"⏳ Waiting {time_to_wait:.2f} seconds before saving similarity matrix..."
            )
            time.sleep(time_to_wait)

            torch.save(cosine_sim_matrix.half(), similarity_save_path)
            print(f"📐 Cosine similarity matrix saved to: {similarity_save_path}")

    # # Create ground truth matrix once, in parent directory
    # parent_dir = args.verification_dir
    # gt_matrix_save_path = os.path.join(parent_dir, f'split_{args.split}_gt_matrix.pt')

    # if not os.path.exists(gt_matrix_save_path):
    #     print(f"🧩 Creating ground truth matrix...")
    #     gt_matrix = create_ground_truth_matrix(ds.df_file)
    #     torch.save(gt_matrix, gt_matrix_save_path)
    #     print(f"✅ Ground truth matrix saved to: {gt_matrix_save_path}")
    # else:
    #     print(f"📁 Ground truth matrix already exists at: {gt_matrix_save_path} — loading it.")
    #     gt_matrix = torch.load(gt_matrix_save_path)

    # # For completeness, optionally copy to output_dir
    # gt_matrix_save_copy = os.path.join(output_dir, f'split_{args.split}_gt_matrix.pt')
    # if not os.path.exists(gt_matrix_save_copy):
    #     torch.save(gt_matrix, gt_matrix_save_copy)

    # # Compute TMR at FMR thresholds
    # print(f"📊 Computing TMR at selected FMR thresholds...")
    # results = compute_tmr_at_fmr(cosine_sim_matrix, gt_matrix, fmr_targets=[0.01, 0.001, 0.0001])
    # results_output_path = os.path.join(output_dir, f'split_{args.split}_tmr_results.csv')

    # # Save results
    # with open(results_output_path, "w") as f:
    #     f.write("FMR,TMR,Threshold\n")
    #     for fmr, tmr, threshold in results:
    #         print(f"📈 TMR @ FMR={fmr:.4%} → {tmr:.4f} (threshold={threshold:.4f})")
    #         f.write(f"{fmr},{tmr:.4f},{threshold:.4f}\n")

    # print(f"✅ TMR results saved to: {results_output_path}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Evaluate model on dataset")
    parser.add_argument("--model_name", type=str, default="OpenCLIP-Huge", help="Name of the model to evaluate")
    parser.add_argument("--dataset_name", type=str, default="LFW-112", help="Name of the dataset to evaluate on")
    parser.add_argument("--split", type=int, default=1, help="Dataset split to use (1-5 for WebFace42M)")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for evaluation")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of workers for DataLoader")
    parser.add_argument("--output_dir", type=str, default="extracted_features", help="Directory to save extracted features")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device to run the model on (e.g., 'cuda' or 'cpu')")
    parser.add_argument("--verification_dir", type=str, default="extracted_features/LFW", help="Directory to save verification results")
    parser.add_argument("--flip", action="store_true", help="Whether to flip the images horizontally during evaluation")
    args = parser.parse_args()
    
    if args.flip:
        print("🔄 Flipping images horizontally during evaluation.")
        args.output_dir = os.path.join(f"extracted_features/{args.dataset_name}_flipped")
    else:
        print("❌ Not flipping images during evaluation.")
        args.output_dir = os.path.join(f"extracted_features/{args.dataset_name}_unflipped")

    args.device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    main(args)
