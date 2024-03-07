# Based on the train-cluster and model.py files from Casper's house_siam repo

"""
We opt for simplicity, and build up the contrastive framework from it's principles.
It is similar to the original SimSiam implementation, with minor modifications:
- compatibility to other encoder types (not necessarily CNNs)
- latest tricks on effective model coding (including some hack and best batchnorm and activation settings)
- head is separately initialized (instead of 'added' to the encoder)
"""

from typing import List, Tuple

import pytorch_lightning as pl
import torch

import torch.nn as nn

from torch.nn.modules.distance import CosineSimilarity

from torchvision import models



class SimSiamResnet(pl.LightningModule):
    """SimSiam model with ResNet18 backbone by default.

    Args:
        backbone: backbone to use for encoder. Default is resnet18.
        
        out_dim, head_dim, pred_dim:  dimensions of the output of the encoder, the head and the predictor respectively. 
                                                        Scaled by dividing by 4 compared to resnet50 implementation.

        imagenet_pretrained: whether to use imagenet pretrained weights for the encoder.
        **kwargs: caputure any additional arguments.

    """
    def __init__(self, 
                    backbone="resnet18", 
                    out_dim=512,
                    head_dim=512, 
                    pred_dim=128,

                    input_is_black_and_white=False,

                    imagenet_pretrained=False, 
                    **kwargs
                ):
        super().__init__()

        self.save_hyperparameters()

        # encoder is defined as the feature extractor only (different to original repo, although for lin evaluation original repo also throws away projection head)
        self.encoder = models.__dict__[backbone](weights=models.ResNet18_Weights.IMAGENET1K_V1 if imagenet_pretrained else None)

        if input_is_black_and_white:

            assert not imagenet_pretrained, "Using imagenet pretrained weights with black and white input is not supported."

            self.encoder.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

        # remove linear layer from the encoder
        self.encoder.fc = nn.Identity()

        self.embedding_dim = out_dim

        # transformation head (3 layers of equal dim)
        # like the original repo: bias=False if followed by batchnorm (hack)
        self.head = nn.Sequential(
            nn.Linear(out_dim, head_dim, bias=False),
            nn.BatchNorm1d(head_dim),
            nn.ReLU(inplace=True),
            
            nn.Linear(head_dim, head_dim, bias=False),
            nn.BatchNorm1d(head_dim),
            nn.ReLU(inplace=True),
            
            nn.Linear(head_dim, head_dim, bias=False),
            nn.BatchNorm1d(head_dim, affine=False),
        )

        # predictor (2 layer bottleneck) - only batchnorm at hidden (see simsiam paper)
        self.predictor = nn.Sequential(
            nn.Linear(head_dim, pred_dim, bias=False),
            nn.BatchNorm1d(pred_dim),
            nn.ReLU(inplace=True),

            nn.Linear(pred_dim, head_dim)
        )

        self.criterion = CosineSimilarity(dim=1)

        self.validation_step_outputs: List[Tuple[torch.Tensor]] = []


    # for purposes of evaluating the representation: fine tuning, linear evaluation, manifold embedding
    def forward(self, batch):

        if isinstance(batch, dict):
            batch["pred"] = self.encoder(batch["img"]) # these are the "frozen" features (after global pooling) - standard protocol

            del batch["img"] # delete the image to save memory

            return batch
        else:
            return self.encoder(batch)
        

    def forward_train(self, x1, x2, return_features=False):
        # features
        feats1 = self.encoder(x1).squeeze()
        feats2 = self.encoder(x2).squeeze() 

        # projections 
        r1 = self.head(feats1)    
        r2 = self.head(feats2)

        # predictions
        p1 = self.predictor(r1) 
        p2 = self.predictor(r2)
        
        if return_features:
            return r1.detach(), r2.detach(), p1, p2, feats1.detach()
        else:
            return r1.detach(), r2.detach(), p1, p2 # detach the projections = stopgradient

    def configure_optimizers(self):

        # optimizer = SGD(self.parameters(), self.init_lr, momentum=self.momentum,
        #             weight_decay=self.weight_decay)
                    
        # return {
        #     "optimizer": optimizer,
        #     "lr_scheduler": CosineAnnealingLR(optimizer, self.hparams.max_epochs),
        # }


        # Configure through CLI instead

        pass


    
    def training_step(self, batch, batch_idx):

        if isinstance(batch, dict):
            if "img" in batch:
                x1, x2 = batch["img"], batch["aug_img"]
            else:
                x1, x2 = batch["view1"], batch["view2"]
        else:
            (x1, x2), _ = batch

        r1, r2, p1, p2 = self.forward_train(x1, x2)

        # loss
        loss = -(self.criterion(r1, p2).mean() + self.criterion(r2, p1).mean())*0.5

        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return loss
    
    
    def validation_step(self, batch, batch_idx):

        assert isinstance(batch, dict), "Batch must be a dict."
        assert "view1" in batch and "view2" in batch, "Batch must contain views 1 and 2."

        view1, view2 = batch["view1"], batch["view2"]

        r1, r2, p1, p2, feats1 = self.forward_train(view1, view2, return_features=True)

        loss = -(self.criterion(r1, p2).mean() + self.criterion(r2, p1).mean()) * 0.5
                                                                
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        output = feats1.detach().cpu(), batch["num_rooms"].detach().cpu(), batch["id"].detach().cpu()

        self.validation_step_outputs.append(output)
