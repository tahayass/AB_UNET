from visualizations import visualise_masks
from utils import check_accuracy,get_loaders
import os
import torchvision
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2




loaded_model = torch.load(r"C:\Users\taha.DESKTOP-BQA3SEM\Desktop\Stage\AB_UNET\model.pth")
loaded_model.eval()
test_dir=r"C:\Users\taha.DESKTOP-BQA3SEM\Desktop\Stage\AB_UNET\test_images"

visualise_masks(test_dir,loaded_model)



