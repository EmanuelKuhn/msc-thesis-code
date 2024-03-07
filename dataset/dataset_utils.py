# Function that converts a list of dicts to a dict of lists. If a dict column is a tensor, that column is concatenated.
import typing

import torch


def collate_dicts(batch: typing.List[typing.Dict[str, torch.Tensor]]):
    res = {}
    for key in batch[0].keys():
        if isinstance(batch[0][key], torch.Tensor):
            res[key] = torch.cat([x[key] for x in batch], dim=0)
        else:
            res[key] = [x[key] for x in batch]
    return res
