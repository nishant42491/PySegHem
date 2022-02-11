import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader,Dataset
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
from PIL import Image
import numpy as np


class Train_Data(Dataset):
    """
    classname: Train_Data
    Parent class: torch.utils.data.Dataset
    Methods:
        __len__():
            Arguments : None
            Output: length of dataset (int)
        
        __getitem__():
            Arguments:
                - idx (int): index
            Output : dataset item (image, mask) at index 'idx'
    """
    def __init__(self, path_to_data: str, trans=False):
        super().__init__()
        self.image_name_list = os.listdir(os.path.join(path_to_data, "segmented_images"))
        self.trans = trans
        self.BASE_IMG_DIR = f'{path_to_data}/segmented_images'
        self.BASE_MASK_DIR = f'{path_to_data}/segmented_masks'

    def __len__(self):
        return len(self.image_name_list)

    def __getitem__(self, idx):
        img_name = self.image_name_list[idx]

        img_id_path = os.path.join(self.BASE_IMG_DIR, img_name)
        mask_id_path = os.path.join(self.BASE_MASK_DIR, img_name)

        img = np.asarray(Image.open(img_id_path))
        mask = np.asarray(Image.open(mask_id_path))

        if self.trans:
            transformed = self.trans(image=img, mask=mask)
            img = transformed['image']
            mask = transformed['mask']

            mask = mask / 255

        return img, mask


def create_train_loader(path_to_train_data: str, batch_size: int = 8, apply_data_transform: bool = False, transform_prob: float = 0.3,
                        mean_data: float = 0.2289, std_data: float = 0.312):

    if apply_data_transform:
        transforms = A.Compose([
            A.Resize(height=512,width=512),
            A.Normalize(mean=(mean_data,), std=(std_data, )),
            A.HorizontalFlip(p=transform_prob),
            A.RandomRotate90(p=transform_prob),
            A.ShiftScaleRotate(p=transform_prob),
            A.VerticalFlip(p=transform_prob),
            ToTensorV2(), ])

    else:
        transforms = A.Compose([
            A.Resize(height=512,width=512),
            A.Normalize(mean=(mean_data,), std=(std_data, )),
            ToTensorV2(), ])

    train_data = Train_Data(path_to_train_data,transforms)

    return DataLoader(train_data, batch_size=batch_size, shuffle=True)


def create_validation_loader(path_to_validation_data: str, batch_size: int = 8,
                             mean_data: float = 0.2289, std_data: float = 0.312):

    transforms = A.Compose([
        A.Resize(height=512,width=512),
        A.Normalize(mean=(mean_data,), std=(std_data, )),
        ToTensorV2(), ])

    val_data = Train_Data(path_to_validation_data, transforms)

    return DataLoader(val_data, batch_size=batch_size)







