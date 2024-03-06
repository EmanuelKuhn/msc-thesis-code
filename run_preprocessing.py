import dataclasses
import os
from pathlib import Path
import typing

from omegaconf import OmegaConf, MISSING

import omegaconf

from enum import Enum, auto

import rplanpy

from pathlib import Path

from tqdm.contrib.concurrent import process_map

from preprocessing.preprocessess_rplanpy import PreprocessingCategory, PreprocessingRplanpy

import datasets

from preprocessing.preprocess_floorplan import PreprocessingFloorplan

class PreprocessingStyle(Enum):
    CONSISTENT_WALL_THICKNESS = "ds_rplan_processed_geometry_rgb"
    CATEGORY_CHANNEL = "ds_rplan_category"
    RPLANPY = "ds_rplanpy_rgb"

@dataclasses.dataclass
class Config:
    
    # splits: str =  "data/splits/"
    
    # split: str =  "val"

    method: PreprocessingStyle=MISSING

    rplan_dataset_path: str="/home/emanuel/thesisdata/dataset/floorplan_dataset"

    save_datasets_path: str = "data/processed/"


def main():

    conf = OmegaConf.structured(Config)
    conf.merge_with_cli()
    conf = typing.cast(Config, conf)

    for field in dataclasses.fields(Config):
        print(field)
        getattr(conf, field.name)

    ids = list(range(10000))

    split = datasets.NamedSplit("debugging")

    if conf.method == PreprocessingStyle.CATEGORY_CHANNEL:
        preprocessing = PreprocessingCategory(conf.rplan_dataset_path)

        category_dict = preprocessing.make_category_dict(ids)

        ds = datasets.arrow_dataset.Dataset.from_dict(category_dict, split=split)
    elif conf.method == PreprocessingStyle.RPLANPY:
        preprocessing = PreprocessingRplanpy(conf.rplan_dataset_path)

        rplanpy_dict = preprocessing.make_rplanpy_dict(ids)

        ds = datasets.arrow_dataset.Dataset.from_dict(rplanpy_dict, split=split)
    elif conf.method == PreprocessingStyle.CONSISTENT_WALL_THICKNESS:
        preprocessing = PreprocessingFloorplan(conf.rplan_dataset_path)

        fp_dict = preprocessing.make_floorplan_dict(ids)

        ds = datasets.arrow_dataset.Dataset.from_dict(fp_dict, split=split)

        
    else:
        raise ValueError(f"Unknown preprocessing method: {conf.method}")

    os.makedirs(conf.save_datasets_path, exist_ok=True)

    ds.save_to_disk(Path(conf.save_datasets_path) / conf.method.value)



if __name__ == "__main__":
    main()