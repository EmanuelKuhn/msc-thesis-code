import dataclasses
import os
from pathlib import Path
import typing

from omegaconf import OmegaConf, MISSING

import omegaconf

from enum import Enum, auto

import rplanpy

from pathlib import Path
import torch

from tqdm.contrib.concurrent import process_map
from dataset.dataset import OriginalRow, SingleDataset
from dataset.dataset_utils import collate_dicts
from inference.model_context import WandbModelContext

from preprocessing.preprocessess_rplanpy import PreprocessingCategory, PreprocessingRplanpy

import datasets

from preprocessing.preprocess_floorplan import PreprocessingFloorplan
from run_preprocessing import PreprocessingStyle

import pytorch_lightning as pl

from torch.utils.data import DataLoader

# class PreprocessingStyle(Enum):
#     CONSISTENT_WALL_THICKNESS = "ds_rplan_processed_geometry_rgb"
#     CATEGORY_CHANNEL = "ds_rplan_category"
#     RPLANPY = "ds_rplanpy_rgb"

@dataclasses.dataclass
class Config:
        
    split: str =  "debugging"

    # method: PreprocessingStyle=MISSING

    save_datasets_path: str = "data/processed/"

    wandb_prefix: str = "emanuel/msc_thesis_models"

    # Reference to the model artifact that was logged on wandb
    wandb_model_ref: str = "model-uc18eq89:best"

    features_cache_folder: str = "data/predicted/"



def model_huggingface_url_to_preprocessing_style(huggingface_url) -> PreprocessingStyle:
    """Convert the huggingface url used for training to the local dataset type that should be used."""
    if huggingface_url == "ekuhn/ds_rplanpy_floorplan_to_color":
        return PreprocessingStyle.RPLANPY
    elif huggingface_url == "ekuhn/ds_rplanpy_category":
        return PreprocessingStyle.CATEGORY_CHANNEL
    else:
        raise ValueError(f"Unmapped {huggingface_url=}")

def save_features(wandb_prefix, wandb_model_ref, split, save_datasets_path, features_cache_folder):
    # Load model
    model_context = WandbModelContext(wandb_prefix, wandb_model_ref)

    method = model_huggingface_url_to_preprocessing_style(model_context.huggingface_dataset)

    # Load dataset
    try:
        ds = datasets.load_from_disk(str(Path(save_datasets_path) / method.value / split))

        assert isinstance(ds, datasets.Dataset), f"Expexted a Dataset, but found: {type(ds)=}"
    except FileNotFoundError:
        raise FileNotFoundError(f"\n\nYou should first run the dataset preprocessing script for {method.name=} and {split=}")

    features_cache_folder = Path(features_cache_folder) / wandb_prefix / wandb_model_ref / method.value / split

    dataset = SingleDataset(OriginalRow(ds))
            
    dataloader = DataLoader(dataset, batch_size=128, num_workers=4)

    # Predict features
    trainer = pl.Trainer(accelerator="cuda")

    predicted = trainer.predict(model_context.model, dataloader)
    features = collate_dicts(predicted)

    os.makedirs(features_cache_folder, exist_ok=True)

    torch.save(features, features_cache_folder / "feats.pth")


def main():

    conf = OmegaConf.structured(Config)
    conf.merge_with_cli()
    conf = typing.cast(Config, conf)


    for field in dataclasses.fields(Config):
        getattr(conf, field.name)

    save_features(conf.wandb_prefix, conf.wandb_model_ref, conf.split, conf.save_datasets_path, conf.features_cache_folder)

    

if __name__ == "__main__":
    main()