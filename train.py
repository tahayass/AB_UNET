import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from AB_UNET_base_model import AB_UNET
import DataLoader

"""
# Hyperparameters etc.
LEARNING_RATE = 
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 
NUM_EPOCHS = 
NUM_WORKERS = 
IMAGE_HEIGHT = 
IMAGE_WIDTH =
PIN_MEMORY = 
LOAD_MODEL = 
TRAIN_IMG_DIR = 
TRAIN_MASK_DIR = 
VAL_IMG_DIR = 
VAL_MASK_DIR =
"""
def train_fn(loader, model, optimizer, loss_fn, scaler):
    loop = tqdm(loader)

    for batch_idx, (data, targets) in enumerate(loop):
        data = data.to(device=torch.device)
        #targets = targets.float().unsqueeze(1).to(device=DEVICE)

        # forward
        with torch.cuda.amp.autocast():
            predictions = model(data)
            loss = loss_fn(predictions, targets)

        # backward
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # update tqdm loop
        loop.set_postfix(loss=loss.item())