"""
Model hub utilities for loading and managing various vision models.

This module provides utilities for downloading and loading models from Hugging Face,
including helper functions for model management and fine-tuned model loading.

Author: Redwan Sony
PhD Student, iPRoBe Lab
Computer Science and Engineering
Michigan State University
"""

import os
import shutil
import sys

import torch
from huggingface_hub import hf_hub_download
from PIL import Image
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Normalize, Resize, ToTensor
from transformers import AutoModel

from evaluation_verification.config import WebFace42M_DATASET_LOCATION

from .common import compute_tmr_at_fmr, create_ground_truth_matrix, extract_features
from .config import HF_TOKEN
from .fr_datasets import get_webface42m_dataloader


# helpfer function to download huggingface repo and use model
def download(repo_id, path, HF_TOKEN=None):
    os.makedirs(path, exist_ok=True)
    files_path = os.path.join(path, 'files.txt')
    if not os.path.exists(files_path):
        hf_hub_download(repo_id, 'files.txt', token=HF_TOKEN,
                        local_dir=path, local_dir_use_symlinks=False)
    with open(os.path.join(path, 'files.txt'), 'r') as f:
        files = f.read().split('\n')
    for file in [f for f in files if f] + ['config.json', 'wrapper.py', 'model.safetensors']:
        full_path = os.path.join(path, file)
        if not os.path.exists(full_path):
            hf_hub_download(repo_id, file, token=HF_TOKEN,
                            local_dir=path, local_dir_use_symlinks=False)


# helpfer function to download huggingface repo and use model
def load_model_from_local_path(path, HF_TOKEN=None):
    cwd = os.getcwd()
    os.chdir(path)
    sys.path.insert(0, path)
    model = AutoModel.from_pretrained(
        path, trust_remote_code=True, token=HF_TOKEN)
    os.chdir(cwd)
    sys.path.pop(0)
    return model


# helpfer function to download huggingface repo and use model
def load_model_by_repo_id(repo_id, save_path, HF_TOKEN=None, force_download=False):
    if force_download:
        if os.path.exists(save_path):
            shutil.rmtree(save_path)
    download(repo_id, save_path, HF_TOKEN)
    return load_model_from_local_path(save_path, HF_TOKEN)


def load_model_by_name(model_name, device: torch.device):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if "adaface" in model_name:
        repo_id = f"minchul/cvlface_{model_name}"
        path = os.path.abspath(f'cvlface_cache/minchul/cvlface_{model_name}')
        model = load_model_by_repo_id(repo_id, path, HF_TOKEN)
        processor = Compose([ToTensor(),
                             Resize((112, 112), interpolation=Image.BICUBIC),
                             Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])
        return model, processor

    if 'arcface' in model_name:
        repo_id = f"minchul/cvlface_{model_name}"
        path = os.path.abspath(f'cvlface_cache/minchul/cvlface_{model_name}')
        model = load_model_by_repo_id(repo_id, path, HF_TOKEN)
        processor = Compose([ToTensor(),
                             Resize((112, 112), interpolation=Image.BICUBIC),
                             Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])
        return model, processor

    elif model_name.startswith("openai/clip-vit"):
        from transformers import CLIPModel, CLIPProcessor
        model = CLIPModel.from_pretrained(model_name).to(device)
        processor = CLIPProcessor.from_pretrained(model_name)
        return model, processor

    elif model_name == "LAION/OpenCLIP-Huge":
        import open_clip
        model_id = "hf-hub:laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
        model, _, processor = open_clip.create_model_and_transforms(
            model_id, device=device)
        return model, processor

    elif model_name == "LAION/OpenCLIP-Base":
        import open_clip
        model_id = "hf-hub:laion/CLIP-ViT-B-32-laion2B-s34B-b79K"
        model, _, processor = open_clip.create_model_and_transforms(
            model_id, device=device)
        return model, processor

    elif model_name == "LAION/OpenCLIP-Giant":
        import open_clip
        model_id = "hf-hub:laion/CLIP-ViT-g-14-laion2B-s12B-b42K"
        model, _, processor = open_clip.create_model_and_transforms(
            model_id, device=device)
        return model, processor

    elif model_name.startswith('Salesforce/blip2'):
        print(f"Loading BLIP2 model: {model_name}")
        from transformers import Blip2ForConditionalGeneration, Blip2Processor
        model = Blip2ForConditionalGeneration.from_pretrained( model_name).to(device)
        processor = Blip2Processor.from_pretrained(model_name)
        return model, processor

    elif model_name.startswith('Salesforce/blip'):
        print(f"Loading BLIP model: {model_name}")
        from transformers import BlipForConditionalGeneration, BlipProcessor
        model = BlipForConditionalGeneration.from_pretrained(
            model_name).to(device)
        processor = BlipProcessor.from_pretrained(model_name)
        return model, processor

    elif model_name == "kakaobrain/align-base":
        from transformers import AutoModel, AutoProcessor
        model = AutoModel.from_pretrained(model_name).to(device)
        processor = AutoProcessor.from_pretrained(model_name)
        return model, processor

    elif model_name == "llava-hf/llava-1.5-7b-hf":
        from transformers import AutoProcessor, LlavaForConditionalGeneration
        processor = AutoProcessor.from_pretrained(model_name)
        model = LlavaForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto")
        model = model.to(device)

        return model, processor

    elif model_name == "llava-hf/llava-v1.6-mistral-7b-hf":
        from transformers import (
            LlavaNextForConditionalGeneration,
            LlavaNextPreTrainedModel,
            LlavaNextProcessor,
        )
        processor = LlavaNextProcessor.from_pretrained(model_name)
        model = LlavaNextForConditionalGeneration.from_pretrained(model_name,
                                                                  torch_dtype=torch.float16,
                                                                  device_map="auto")

        model = model.to(device)
        return model, processor

    elif model_name.startswith("facebook/sam-vit"):
        from transformers import SamModel, SamProcessor
        model = SamModel.from_pretrained(model_name).to(device)
        processor = SamProcessor.from_pretrained(model_name)
        return model, processor

    elif model_name == "microsoft/kosmos-2-patch14-224":
        from transformers import AutoModel, AutoProcessor, Kosmos2Model
        model = Kosmos2Model.from_pretrained(model_name).to(device)
        processor = AutoProcessor.from_pretrained(model_name, use_fast=True)

        return model, processor

    elif model_name.startswith("google/vit-"):
        from transformers import (
            ViTForImageClassification,
            ViTHybridForImageClassification,
            ViTHybridImageProcessor,
            ViTImageProcessor,
        )

        if "hybrid" in model_name:
            model = ViTHybridForImageClassification.from_pretrained(
                model_name, output_hidden_states=True).to(device).eval()
            processor = ViTHybridImageProcessor.from_pretrained(
                model_name, use_fast=True)
        else:
            model = ViTForImageClassification.from_pretrained(
                model_name, output_hidden_states=True).to(device).eval()
            processor = ViTImageProcessor.from_pretrained(
                model_name, use_fast=True)
        return model, processor

    elif model_name.startswith("deepseek-ai/deepseek-vl2"):
        from deepseek_vl2.models import DeepseekVLV2ForCausalLM
        vl_gpt = DeepseekVLV2ForCausalLM.from_pretrained(
            model_name, device_map="auto",  torch_dtype=torch.bfloat16).eval()

        vision_model = vl_gpt.vision

        def get_deepseek_transform(input_size):
            return Compose([
                Resize((input_size, input_size), interpolation=Image.BICUBIC),
                ToTensor(),
                Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ])

        processor = get_deepseek_transform(384)

        return vision_model,  processor

    elif model_name.startswith("OpenGVLab/InternVL3"):
        from torchvision import transforms as T
        from torchvision.transforms.functional import InterpolationMode
        from transformers import AutoConfig, AutoModel, AutoTokenizer

        def build_internvl3_transform(input_size=448):
            IMAGENET_MEAN = (0.485, 0.456, 0.406)
            IMAGENET_STD = (0.229, 0.224, 0.225)
            return T.Compose([
                T.Lambda(lambda img: img.convert('RGB')
                         if img.mode != 'RGB' else img),
                T.Resize((input_size, input_size),
                         interpolation=InterpolationMode.BICUBIC),
                T.ToTensor(),
                T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
            ])
        processor = build_internvl3_transform(input_size=448)

        model = AutoModel.from_pretrained(model_name,
                                          torch_dtype=torch.bfloat16,
                                          load_in_8bit=False,
                                          low_cpu_mem_usage=True,
                                          use_flash_attn=True,
                                          trust_remote_code=True)

        del model.language_model
        del model.mlp1

        model = model.vision_model
        model.eval()

        model = model.to(device)

        for param in model.parameters():
            param.requires_grad = False
        return model, processor

    elif model_name.startswith("facebook/dino"):
        from transformers import AutoImageProcessor, AutoModel
        model = AutoModel.from_pretrained(
            model_name, trust_remote_code=True).to(device).eval()
        processor = AutoImageProcessor.from_pretrained(model_name)
        return model, processor

    elif model_name.startswith("FineTuned"):
        from evaluation_verification.fine_tuned_vlms import load_model_by_name
        model = load_model_by_name(model_name).eval()
        processor = Compose([
            Resize((224, 224), interpolation=Image.BICUBIC),
            ToTensor(),
            Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        return model, processor

    else:
        raise ValueError(
            f"Model {model_name} is not supported or not implemented yet.")


