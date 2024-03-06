

import numpy as np
import rplanpy

import datasets

from tqdm import tqdm

from pathlib import Path

from functools import cache

from tqdm.contrib.concurrent import process_map

class RplanWrapper:

    def __init__(self, rplan_dataset_path):
        self.rplan_dataset_path = rplan_dataset_path

    def load_rplanpy_data(self, id):
        return rplanpy.data.RplanData(Path(self.rplan_dataset_path) / f"{id}.png")
    
    @cache
    def compute_num_rooms(self, rplan_id):
        fp_rplan = self.load_rplanpy_data(rplan_id)

        num_rooms = len(fp_rplan.get_graph())

        return num_rooms

class PreprocessingCategory(RplanWrapper):

    def render_to_category(self, rplan_id):
        arr = self.load_rplanpy_data(rplan_id).category

        return datasets.Image(decode=False).encode_example(arr)
    
    def make_category_dict(self, ids):

        imgs = process_map(self.render_to_category, ids, desc="Process images", chunksize=10)

        num_rooms = process_map(self.compute_num_rooms, ids, desc="Process num_rooms", chunksize=10)

        return {
            "id": ids,
            "img": imgs,
            "num_rooms": num_rooms,
        }


class PreprocessingRplanpy(RplanWrapper):

    def render_with_rplanpy(self, rplan_id):
        arr = np.array(rplanpy.plot.floorplan_to_color(self.load_rplanpy_data(rplan_id))).astype(np.uint8)

        return datasets.Image(decode=False).encode_example(arr)
    
    def make_rplanpy_dict(self, ids):

        imgs = process_map(self.render_with_rplanpy, ids, desc="Process images", chunksize=10)

        num_rooms = process_map(self.compute_num_rooms, ids, desc="Process num_rooms", chunksize=10)

        return {
            "id": ids,
            "img": imgs,
            "num_rooms": num_rooms,
        }
