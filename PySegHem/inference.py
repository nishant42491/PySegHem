import torch
from train import New_Model
import torchvision.transforms as T
import os
from albumentations.pytorch.transforms import ToTensorV2
import albumentations as A
import pickle
import argparse
from PIL import Image
import numpy as np


transforms = A.Compose([
        A.Resize(height=512,width=512),
        A.Normalize(mean=(0.289,), std=(0.312,)),
        ToTensorV2(), ])


def inference(path_to_model_pickle,image_path,path_to_save_image_folder):

    f = open(path_to_model_pickle,'rb')
    mod = pickle.load(f)
    img = np.asarray(Image.open(image_path))
    img = transforms(image=img)['image'].unsqueeze(0)
    outs = mod(img)
    outs = outs.squeeze(0)
    img = T.ToPILImage()(outs)
    img.save(os.path.join(path_to_save_image_folder, "saved_masl.jpg"))




