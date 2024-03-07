from typing import Protocol
import typing

import torch

from models.simsiam_model import SimSiamResnet

from . import wandb_artifacts

from pathlib import Path


class ModelContext(Protocol):

    @property
    def model(self) -> torch.nn.Module:
        ...
    
    ref: str
    unique_ref: str

    run_url: str

    run_id: str
    
    name: str
    description: str

    @property
    def short_description(self) -> str:
        return self.description.split("\n")[1].replace("da", "GeomPerturb") if len(self.description.split("\n")) > 1 else self.description

    summary: dict[str, typing.Any]

    model_artifact_metadata: dict[str, typing.Any]

    

class WandbModelContext(ModelContext):
    def __init__(self, wandb_prefix, wandb_model_ref):
        self.wandb_prefix = wandb_prefix
        self.ref = wandb_model_ref

        self.unique_ref = f"{wandb_prefix}/{wandb_model_ref}"

        self.model_artifact = wandb_artifacts.get_artifact(f"{wandb_prefix}/{wandb_model_ref}", type="model")

        run = self.model_artifact.logged_by()

        assert run is not None, f"{wandb_prefix}/{wandb_model_ref} failed to resolve to a run"

        self.name = run.name
        self.description = run.description

        self.run_url = run.url
        self.run_id = run.id

        self.summary = run.summary._json_dict

        self._model = None

        self.run_config = self.run.config

        self.huggingface_dataset = self.run_config["huggingface_url"]
    
    @property
    def run(self):
        return self.model_artifact.logged_by()

    @property
    def model_artifact_metadata(self):
        return self.model_artifact.metadata

    @property
    def model(self):
        if self._model is None:
            self._model = SimSiamResnet.load_from_checkpoint(Path(self.model_artifact.file()))
        
        return self._model
