from visualizations import visualise_masks
from utils import check_accuracy,get_loaders,train_val_split
import os
import torchvision
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2


BASE_DIR=r"C:\Users\taha.DESKTOP-BQA3SEM\Desktop\Stage\AB_UNET\DATA"
TRAIN_DIR=r"TRAIN"
VAL_DIR=r"VAL"
train_val_split(BASE_DIR,TRAIN_DIR,VAL_DIR,split_ratio=0.8,shuffle=False)



