
from typing import List

import pandas as pd
from floorplan.render_palette_image import render_room_number_palette_image
from preprocessing.preprocessess_rplanpy import RplanWrapper

from tqdm.contrib.concurrent import process_map

from floorplan.floorplan import FloorPlan

import datasets

class PreprocessingFloorplan(RplanWrapper):

    def load_floorplan(self, rplan_id):
        try:
            return FloorPlan.from_rplan_data(self.load_rplanpy_data(rplan_id))
        except:
            raise ValueError(f"Failed to process {rplan_id=}; This method does not support the full split.")
    
    @staticmethod
    def render_rgb_image(fp: FloorPlan):
        if fp is None:
            return None
        
        try:
            img = render_room_number_palette_image(fp, 4, True)

            img = img.convert("RGB")
        
            return datasets.Image(decode=False).encode_example(img)
        except:
            return None

    @staticmethod
    def render_room_number_image(fp: FloorPlan):
        if fp is None:
            return None
        
        try:
            img = render_room_number_palette_image(fp, 4, True)
        
            return datasets.Image(decode=False).encode_example(img)
        except:
            return None


    def make_floorplan_dict_rgb(self, ids):

        floorplans: List[FloorPlan] = process_map(self.load_floorplan, ids, desc="Parsing floor plans", chunksize=10)

        imgs = process_map(self.render_rgb_image, floorplans, desc="drawing images", chunksize=10)
        
        num_rooms = [len(fp.room_categories) if fp is not None else None for fp in floorplans]

        df = pd.DataFrame({
            "id": ids,
            "img": imgs,
            "num_rooms": num_rooms
        })

        df = df.dropna()

        result_dict = df.to_dict(orient="list")

        return result_dict



    def make_floorplan_dict_room_number(self, ids):

        floorplans: List[FloorPlan] = process_map(self.load_floorplan, ids, desc="Parsing floor plans", chunksize=10)

        imgs = process_map(self.render_room_number_image, floorplans, desc="drawing images", chunksize=10)
        
        num_rooms = [len(fp.room_categories) if fp is not None else None for fp in floorplans]

        df = pd.DataFrame({
            "id": ids,
            "fp": floorplans,
            "img": imgs,
            "num_rooms": num_rooms
        })

        df = df.dropna()

        result_dict = df.to_dict(orient="list")

        result_dict["fp_bytes"] = [fp.to_pickle_str() for fp in result_dict["fp"]]

        del result_dict["fp"]

        return result_dict


