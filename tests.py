from acquisition_functions import score_entropy
from visualizations import visualise_masks
from utils import check_accuracy,get_loaders,train_val_split,stochastic_prediction,output_masks
from AB_UNET_base_model import AB_UNET
import os
import torchvision
import torch
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch import nn



BATCH_SIZE = 1
NUM_EPOCHS = 2
NUM_WORKERS = 0
IMAGE_HEIGHT = 160  # 1280 originally
IMAGE_WIDTH = 240  # 1918 originally
PIN_MEMORY = False
LOAD_MODEL = False
TRAIN_IMG_DIR = r"C:\Users\taha.DESKTOP-BQA3SEM\Desktop\Stage\AB_UNET\DATA\TRAIN\train_images"
TRAIN_MASK_DIR = r"C:\Users\taha.DESKTOP-BQA3SEM\Desktop\Stage\AB_UNET\DATA\TRAIN\train_masks"
VAL_IMG_DIR = r"C:\Users\taha.DESKTOP-BQA3SEM\Desktop\Stage\AB_UNET\DATA\VAL\val_images"
VAL_MASK_DIR = r"C:\Users\taha.DESKTOP-BQA3SEM\Desktop\Stage\AB_UNET\DATA\VAL\val_masks"

    #torch.cuda.empty_cache()
train_transform = A.Compose(
        [   #A.Rotate(limit=35, p=1.0),
            #A.HorizontalFlip(p=0.3),
            #A.VerticalFlip(p=0.1),
            #A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            ToTensorV2()
        ],
    )

x=torch.randn((2,3,300,300)).to('cuda')
model = torch.load('model.pth')
model=model.float()
loader,_= get_loaders(
        TRAIN_IMG_DIR,
        TRAIN_MASK_DIR,
        VAL_IMG_DIR,
        VAL_MASK_DIR,
        BATCH_SIZE,
        train_transform,
        train_transform,
        NUM_WORKERS,
    )

dataiter = iter(loader)
data = dataiter.next()
features,_,image_name= data
print(image_name)
model.eval()
y=stochastic_prediction(model,features.float().to('cuda'),dropout_iteration=6,device='cuda')
print(torch.cuda.max_memory_allocated(device='cuda'))
with torch.no_grad():

    print(score_entropy(y.cpu().numpy()))





