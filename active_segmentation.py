from itertools import dropwhile
import torch
import torch.nn as nn
from utils import train_val_split,get_loaders,check_accuracy
from train import train_fn
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from AB_UNET_base_model import AB_UNET
import os
import shutil
import numpy as np
import matplotlib.pyplot as plt








def move_images(BASE_DIR,TRAIN_DIR,VAL_DIR,num):
        print(len(os.listdir(os.path.join(BASE_DIR,VAL_DIR,'val_images'))))

        images=os.listdir(os.path.join(BASE_DIR,VAL_DIR,'val_images'))
        for im in images[:num] : 
            im_path=os.path.join(BASE_DIR,'images',im)
            target_path=os.path.join(BASE_DIR,TRAIN_DIR,'train_images')
            shutil.copy(im_path, target_path)
            os.remove(os.path.join(BASE_DIR,VAL_DIR,'val_images',im))

            for mask in os.listdir(os.path.join(BASE_DIR,'masks')):
                mask_path=os.path.join(os.path.join(BASE_DIR,'masks',mask),im.replace(".BMP"," "+mask.replace("GT_","")+"_Mask.bmp"))
                target_mask_path=os.path.join(BASE_DIR,TRAIN_DIR,'train_masks',mask)
                shutil.copy(mask_path, target_mask_path)
                os.remove(os.path.join(os.path.join(BASE_DIR,VAL_DIR,'val_masks',mask),im.replace(".BMP"," "+mask.replace("GT_","")+"_Mask.bmp")))

        print(len(os.listdir(os.path.join(BASE_DIR,VAL_DIR,'val_images'))))




def save_model_dict(model,step):
    FILE = f"models/model_step_{step}.pth"
    torch.save(model, FILE)



def random_sampling(num=10):

    
    LEARNING_RATE = 1e-3
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    #DEVICE="cpu"
    BATCH_SIZE = 2
    NUM_EPOCHS = 2
    NUM_WORKERS = 1
    PIN_MEMORY = False
    LOAD_MODEL = False
    TRAIN_IMG_DIR = r"C:\Users\taha.DESKTOP-BQA3SEM\Desktop\Stage\AB_UNET\DATA\Labeled_pool\train_images"
    TRAIN_MASK_DIR = r"C:\Users\taha.DESKTOP-BQA3SEM\Desktop\Stage\AB_UNET\DATA\Labeled_pool\train_masks"
    VAL_IMG_DIR = r"C:\Users\taha.DESKTOP-BQA3SEM\Desktop\Stage\AB_UNET\DATA\Unlabeled_pool\val_images"
    VAL_MASK_DIR = r"C:\Users\taha.DESKTOP-BQA3SEM\Desktop\Stage\AB_UNET\DATA\Unlabeled_pool\val_masks"

#intial split

    BASE_DIR=r"C:\Users\taha.DESKTOP-BQA3SEM\Desktop\Stage\AB_UNET\DATA"
    TRAIN_DIR=r"Labeled_pool"
    VAL_DIR=r"Unlabeled_pool"
    train_val_split(BASE_DIR,TRAIN_DIR,VAL_DIR,split_ratio=0.4,shuffle=False)
    num_images=len(os.listdir(r'C:\Users\taha.DESKTOP-BQA3SEM\Desktop\Stage\AB_UNET\DATA\Unlabeled_pool\val_images'))


#model init    
    model = AB_UNET(in_channels=3, out_channels=4,dropout=0.2,max_dropout=0.4).to(DEVICE)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scaler = torch.cuda.amp.GradScaler()



        #torch.cuda.empty_cache()
    train_transform = A.Compose(
        [   #A.Rotate(limit=35, p=1.0),
            #A.HorizontalFlip(p=0.3),
            #A.VerticalFlip(p=0.1),
            #A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            ToTensorV2()
        ],
    )


    for step in range(int(num_images/num)):
        print(f'step number {step} : ')

        train_loader,_= get_loaders(
            TRAIN_IMG_DIR,
            TRAIN_MASK_DIR,
            VAL_IMG_DIR,
            VAL_MASK_DIR,
            BATCH_SIZE,
            train_transform,
            train_transform,
            NUM_WORKERS,
        )
        for epoch in tqdm(range(NUM_EPOCHS)):
            dice_array=[]
            train_fn(train_loader, model, optimizer, loss_fn, scaler)
            acc,dice=check_accuracy(train_loader,model,BATCH_SIZE, device=DEVICE)
            #save_model_dict(model,step)

        dice_array.append(dice)
        #move 10 images at random to labeled pool
        move_images(BASE_DIR,TRAIN_DIR,VAL_DIR,num)


    dice_stats=np.array(dice_array.to(DEVICE)).to(DEVICE)
    plt.plot(dice_stats,np.arange(1,int(num_images/num),1))
    plt.show()
        




    






if __name__ == "__main__":
    random_sampling()