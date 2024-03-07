import datasets
import torch

import torch.utils.data

from torchvision.transforms.functional import to_tensor

from torchvision.transforms import Compose

from typing import List, Protocol

class RplanGetForId(Protocol):

    ids: List[int]

    # Returns a dictionary with at least the keys "id" and "img", where img is a tensor of shape (C, H, W)
    def get_for_id(self, rplan_id: int) -> dict:
        ...


class OriginalRow(RplanGetForId):
    def __init__(self, 
                 ds: datasets.arrow_dataset.Dataset, 
                 img_column: str="img", 
                 transform: torch.nn.Module | Compose | None=None, 
                 keep_columns=["id", "num_rooms"],
                 pillow_to_tensor_transform=to_tensor):
        
        if transform is None:
            transform = torch.nn.Identity()
        
        self.transform = transform

        self.pillow_to_tensor_transform = pillow_to_tensor_transform

        self.img_column = img_column
        self.keep_columns = keep_columns

        self._ds = ds

        self.ds_img = ds.select_columns([*keep_columns, img_column])
        self.ds_img = self.ds_img.cast_column(img_column, datasets.Image(decode=True))
        
        self.ds_img.set_transform(self._transform_row)

        # Create index data structure
        ids_series = ds.select_columns(["id"]).to_pandas()["id"]

        assert len(ids_series) == len(ids_series.unique()), "When using OriginalRow, the dataset must contain one image per id."

        self.id_to_index = {id: i for i, id in enumerate(ids_series)}

        # Available ids in the dataset
        self.ids = sorted(self.id_to_index.keys())
    
    def _transform_row(self, batch):
        return {
            self.img_column: [self.transform(self.pillow_to_tensor_transform(img)) for img in batch[self.img_column]],
            **{column: torch.tensor(batch[column]) for column in self.keep_columns}
        }

    def get_for_id(self, rplan_id):
        """Get index for a given id."""
        index = self.id_to_index[rplan_id]

        sample = self.ds_img[index]

        return {
            **{key: sample[key] for key in self.keep_columns},
            "img": sample[self.img_column],
        }

class SingleDataset(torch.utils.data.Dataset):
    def __init__(self, data: RplanGetForId) -> None:
        """Dataset for sampling a random augmentation (=row) for a given id. The ds should have columns: id, [img_column], num_rooms, augmentation_index.
        """

        self.data = data
    
    def __len__(self):
        """Number of unique ids in the dataset."""
        return len(self.data.ids)

    def __getitem__(self, i):
        assert isinstance(i, int), "Batch indexing is not supported"

        rplan_id = self.data.ids[i]

        return self.data.get_for_id(rplan_id)
