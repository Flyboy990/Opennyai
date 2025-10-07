import os
import subprocess
import sys
from pathlib import Path
import torch
from huggingface_hub import hf_hub_download

"""Functions for downloading opennyai ner models."""

PIP_INSTALLER_URLS = {
    "en_legal_ner_trf": "https://huggingface.co/opennyaiorg/en_legal_ner_trf/resolve/main/en_legal_ner_trf-any-py3-none-any.whl",
    "en_legal_ner_sm": "https://huggingface.co/opennyaiorg/en_legal_ner_sm/resolve/main/en_legal_ner_sm-any-py3-none-any.whl",
    "en_core_web_md": "https://huggingface.co/opennyaiorg/en_legal_ner_trf/resolve/main/STOCK_SPACY_MODELS/en_core_web_md-3.2.0-py3-none-any.whl",
    "en_core_web_sm": "https://huggingface.co/opennyaiorg/en_legal_ner_trf/resolve/main/STOCK_SPACY_MODELS/en_core_web_sm-3.2.0-py3-none-any.whl",
    "en_core_web_trf": "https://huggingface.co/opennyaiorg/en_legal_ner_trf/resolve/main/STOCK_SPACY_MODELS/en_core_web_trf-3.2.0-py3-none-any.whl"
}

TORCH_PT_MODEL_URLS = {
    "RhetoricalRole": "https://huggingface.co/opennyaiorg/InRhetoricalRoles/resolve/main/InRhetoricalRoleModel.pt",
    "ExtractiveSummarizer": "https://huggingface.co/opennyaiorg/InExtractiveSummarizer/resolve/main/InExtractiveSummarizerModel.pt"
}

# Model repo info for new hf_hub_download API
HF_MODEL_REPOS = {
    "RhetoricalRole": {
        "repo_id": "opennyaiorg/InRhetoricalRoles",
        "filename": "InRhetoricalRoleModel.pt"
    },
    "ExtractiveSummarizer": {
        "repo_id": "opennyaiorg/InExtractiveSummarizer",
        "filename": "InExtractiveSummarizerModel.pt"
    }
}

CACHE_DIR = os.path.join(str(Path.home()), '.opennyai')


def install(package: str):
    """
    It is used for installing pip wheel file for model supported
    Args:
        package (string): wheel file url
    """
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", package, "--no-deps"], 
        stdout=subprocess.DEVNULL
    )


def load_model_from_cache(model_name: str):
    """
    It is used for downloading model.pt files supported and developed by Opennyai
    Args:
        model_name (string): model name to download and save
    """
    if model_name not in HF_MODEL_REPOS:
        raise RuntimeError(f'{model_name} is not supported by opennyai, please check the name!')
    
    model_info = HF_MODEL_REPOS[model_name]
    cache_dir = os.path.join(CACHE_DIR, model_name.lower())
    os.makedirs(cache_dir, exist_ok=True)
    
    # Download using new huggingface_hub API
    print(f'Downloading: "{model_info["repo_id"]}/{model_info["filename"]}" to {cache_dir}')
    
    model_path = hf_hub_download(
        repo_id=model_info["repo_id"],
        filename=model_info["filename"],
        cache_dir=cache_dir
    )
    
    # Load the model
    state_dict = torch.load(model_path, map_location=torch.device('cpu'))
    
    return state_dict
