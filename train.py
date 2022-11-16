from sched import scheduler
import torch
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from AB_UNET_base_model import AB_UNET
import torch.optim.lr_scheduler
import DataLoader
import matplotlib.pyplot as plt
from utils import get_loaders,check_accuracy,train_val_split,check_accuracy_batch,reset_DATA



############## TENSORBOARD ########################
import sys
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
# default `log_dir` is "runs" - we'll be more specific here
writer = SummaryWriter('runs/AB-UNET')
###################################################


# Hyperparameters.
LEARNING_RATE = 1e-3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
#DEVICE="cpu"
BATCH_SIZE = 2
NUM_EPOCHS = 30
NUM_WORKERS = 2
IMAGE_HEIGHT = 160  # 1280 originally
IMAGE_WIDTH = 240  # 1918 originally
PIN_MEMORY = False
LOAD_MODEL = False
TRAIN_IMG_DIR = r".\DATA\TRAIN\train_images"
TRAIN_MASK_DIR = r".\DATA\TRAIN\train_masks"
VAL_IMG_DIR = r".\DATA\VAL\val_images"
VAL_MASK_DIR = r".\DATA\VAL\val_masks"

def train_fn(loader,model, optimizer, loss_fn, scaler,scheduler):
    loop = tqdm(loader)
    acc=[]
    di=[]
    for batch_idx, (data, targets, _) in enumerate(loop):
        #print(data.shape)
        data = data.float().to(device=DEVICE)
        targets = targets.float().to(device=DEVICE)
        # forward
        with torch.cuda.amp.autocast():
            predictions = model(data)
            loss = loss_fn(predictions, targets)
        # backward
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        batch_acc,batch_dice=check_accuracy_batch(data,targets,model,data.shape[0],device="cuda")
        acc.append(batch_acc.item())
        di.append(batch_dice.item())
        # update tqdm loop
        loop.set_postfix({'loss':loss.item(),'accuracy':np.average(acc),'dice':np.average(di)})
    if scheduler:
        scheduler.step(loss)


def main():

    BASE_DIR=r".\DATA"
    TRAIN_DIR=r"TRAIN"
    VAL_DIR=r"VAL"
    reset_DATA(r".\DATA")
    train_val_split(BASE_DIR,TRAIN_DIR,VAL_DIR,split_ratio=0.75,shuffle=True)


    #torch.cuda.empty_cache()
    train_transform = A.Compose(
        [   #A.Rotate(limit=35, p=1.0),
            #A.HorizontalFlip(p=0.3),
            #A.VerticalFlip(p=0.1),
            #A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            A.augmentations.transforms.Equalize(mode="cv",p=1),
            ToTensorV2()
        ],
    )


    model = AB_UNET(in_channels=3, out_channels=4,max_dropout=0,dropout=0).to(DEVICE)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler= torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode='min',patience=3,verbose=True)
    train_loader,val_loader= get_loaders(
        TRAIN_IMG_DIR,
        TRAIN_MASK_DIR,
        VAL_IMG_DIR,
        VAL_MASK_DIR,
        BATCH_SIZE,
        train_transform,
        train_transform,
        NUM_WORKERS,
    )
    scaler = torch.cuda.amp.GradScaler()
    dice_array=[]
    accuracy_array=[]
    val_dice=[]
    val_accuracy=[]


    for epoch in tqdm(range(NUM_EPOCHS)):
        train_fn(train_loader, model, optimizer, loss_fn, scaler, scheduler)
        print("\n")
        # check accuracy
        acc,dice=check_accuracy(train_loader, model,BATCH_SIZE, device=DEVICE)
        dice_array.append(dice)
        accuracy_array.append(acc)
        writer.add_scalar('training dice', dice, epoch+1)
        print("validation : ")
        vacc,vdice=check_accuracy(val_loader, model,BATCH_SIZE, device=DEVICE)
        val_dice.append(vdice)
        val_accuracy.append(vacc)
        writer.add_scalar('validation dice', vdice, epoch+1)

    dice_stats=torch.tensor(dice_array).detach().cpu().numpy()
    accuracy_stats=torch.tensor(accuracy_array).detach().cpu().numpy()
    vdice_stats=torch.tensor(val_dice).detach().cpu().numpy()
    vacc_stats=torch.tensor(val_accuracy).detach().cpu().numpy()

    plt.plot(np.arange(1,31,1),dice_stats,label='training dice',color='r')
    plt.plot(np.arange(1,31,1),vdice_stats,label='validation dice',color='g')
    plt.ylim(ymin=0)
    plt.title("training and validation dice")
    plt.xlabel("epochs")
    plt.ylabel("Dice metric")
    plt.legend()
    plt.show()

    plt.plot(np.arange(1,31,1),accuracy_stats,label='training accuracy',color='r')
    plt.plot(np.arange(1,31,1),vacc_stats,label='validation accuracy',color='g')
    plt.ylim(ymin=0)
    plt.title("training and validation accuracy")
    plt.xlabel("epochs")
    plt.ylabel("Accuracy metric")
    plt.legend()
    plt.show()


        
    #save_predictions_as_imgs(train_loader, model, folder="saved_images/", device="cuda")
    FILE = "model_30epochs_AO.pth"
    torch.save(model, FILE)
    writer.close()
    



if __name__ == "__main__":
    main()
