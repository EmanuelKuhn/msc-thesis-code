import dataclasses
import os
from pathlib import Path
import sys
import typing

from omegaconf import OmegaConf, MISSING

from enum import Enum

from pathlib import Path

from preprocessing.preprocessess_rplanpy import PreprocessingCategory, PreprocessingRplanpy

import datasets

from preprocessing.preprocess_floorplan import PreprocessingFloorplan

import pandas as pd

class PreprocessingStyle(Enum):
    CONSISTENT_WALL_THICKNESS_RGB = "ds_rplan_processed_geometry_rgb"

    # Stored as categorical value per pixel
    # CONSISTENT_WALL_THICKNESS_FOR_MASKING = "ds_rplan_processed_geometry_masking"
    
    CATEGORY_CHANNEL = "ds_rplan_category"
    RPLANPY = "ds_rplanpy_rgb"

@dataclasses.dataclass
class Config:
    
    splits: str =  "data/splits/"
    
    split: str =  "val"

    method: PreprocessingStyle=MISSING

    rplan_dataset_path: str="/home/emanuel/thesisdata/dataset/floorplan_dataset"

    save_datasets_path: str = "data/processed/"


def print_help(config_cls):
    print("Configuration options (option=default):")
    for field in config_cls.__dataclass_fields__.values():
        print(f"    {field.name}={field.default}")


def main():

    if "--help" in sys.argv:
        print_help(Config)
        sys.exit()

    conf = OmegaConf.structured(Config)
    conf.merge_with_cli()
    conf = typing.cast(Config, conf)

    # Check all fields are set
    for field in dataclasses.fields(Config):
        getattr(conf, field.name)

    split = datasets.NamedSplit(conf.split)

    ids = pd.read_csv(f"data/splits/{split._name}.csv")["id"].tolist()

    if conf.method == PreprocessingStyle.CATEGORY_CHANNEL:
        preprocessing = PreprocessingCategory(conf.rplan_dataset_path)

        data_dict = preprocessing.make_category_dict(ids)

    elif conf.method == PreprocessingStyle.RPLANPY:
        preprocessing = PreprocessingRplanpy(conf.rplan_dataset_path)

        data_dict = preprocessing.make_rplanpy_dict(ids)

    elif conf.method == PreprocessingStyle.CONSISTENT_WALL_THICKNESS_RGB:
        preprocessing = PreprocessingFloorplan(conf.rplan_dataset_path)

        data_dict = preprocessing.make_floorplan_dict_rgb(ids)
        
    else:
        raise ValueError(f"Unknown preprocessing method: {conf.method}")

    ds = datasets.arrow_dataset.Dataset.from_dict(data_dict, split=split, info=datasets.DatasetInfo(description=conf.method.value))

    os.makedirs(conf.save_datasets_path, exist_ok=True)

    ds.save_to_disk(Path(conf.save_datasets_path) / conf.method.value / split._name)



if __name__ == "__main__":
    main()