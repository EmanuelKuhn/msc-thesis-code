import torch
from enum import Enum

from torchvision import transforms

from torchvision.transforms.functional import InterpolationMode

DEFAULT_OUTPUT_SIZE = (256, 256)


img_trans_identity = transforms.Resize(DEFAULT_OUTPUT_SIZE, interpolation=InterpolationMode.BILINEAR, antialias=True)

default_img_trans = torch.nn.Sequential(
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomApply([transforms.RandomRotation((90, 90))], p=0.5),
    transforms.RandomResizedCrop(DEFAULT_OUTPUT_SIZE, scale=(0.8, 1.0), ratio=(0.75, 1.3333333333333333), interpolation=InterpolationMode.BILINEAR, antialias=True),
)


# The augmentations used in SimSiam paper:
# From https://github.com/facebookresearch/simsiam/blob/a7bc1772896d0dad0806c51f0bb6f3b16d290468/main_simsiam.py
# MoCo v2's aug: similar to SimCLR https://arxiv.org/abs/2002.05709
moco_augmentations = transforms.Compose([
        transforms.ToPILImage(),

        transforms.RandomResizedCrop(224, scale=(0.2, 1.)),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomApply([transforms.GaussianBlur(7, [.1, 2.])], p=0.5),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        # normalize
    ])

class AugmentationMode(Enum):
    DefaultImgTrans = "default_img_trans"

    NoAugmentation = "no_augmentation"

    MoCoV2Augmentations = "moco_v2"

    def get_aug_func(self):
        if self == AugmentationMode.DefaultImgTrans:
            return default_img_trans
        elif self == AugmentationMode.NoAugmentation:
            return img_trans_identity
        elif self == AugmentationMode.MoCoV2Augmentations:
            return moco_augmentations
        else:
            raise ValueError(f"Unknown augmentation mode: {self}")
