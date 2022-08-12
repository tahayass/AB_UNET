from itertools import dropwhile
from turtle import color
import torch
import torch.nn as nn
from utils import train_val_split,get_loaders_active,check_accuracy,create_score_dict,labeled_unlabeled_test_split
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
        print(len(os.listdir(os.path.join(BASE_DIR,VAL_DIR,'unlabeled_images'))))

        images=os.listdir(os.path.join(BASE_DIR,VAL_DIR,'unlabeled_images'))
        for im in images[:num] : 
            im_path=os.path.join(BASE_DIR,'images',im)
            target_path=os.path.join(BASE_DIR,TRAIN_DIR,'labeled_images')
            shutil.copy(im_path, target_path)
            os.remove(os.path.join(BASE_DIR,VAL_DIR,'unlabeled_images',im))

            for mask in os.listdir(os.path.join(BASE_DIR,'masks')):
                mask_path=os.path.join(os.path.join(BASE_DIR,'masks',mask),im.replace(".BMP"," "+mask.replace("GT_","")+"_Mask.bmp"))
                target_mask_path=os.path.join(BASE_DIR,TRAIN_DIR,'labeled_masks',mask)
                shutil.copy(mask_path, target_mask_path)
                os.remove(os.path.join(os.path.join(BASE_DIR,VAL_DIR,'unlabeled_masks',mask),im.replace(".BMP"," "+mask.replace("GT_","")+"_Mask.bmp")))

        print(len(os.listdir(os.path.join(BASE_DIR,VAL_DIR,'unlabeled_images'))))

def move_images_with_dict(BASE_DIR,TRAIN_DIR,VAL_DIR,score_dict,num):

        print(len(os.listdir(os.path.join(BASE_DIR,VAL_DIR,'unlabeled_images'))))

        dict_iterator=iter(score_dict)
        for i in range(num) : 
            im=next(dict_iterator)
            im_path=os.path.join(BASE_DIR,'images',im)
            target_path=os.path.join(BASE_DIR,TRAIN_DIR,'labeled_images')
            shutil.copy(im_path, target_path)
            os.remove(os.path.join(BASE_DIR,VAL_DIR,'unlabeled_images',im))

            for mask in os.listdir(os.path.join(BASE_DIR,'masks')):
                mask_path=os.path.join(os.path.join(BASE_DIR,'masks',mask),im.replace(".BMP"," "+mask.replace("GT_","")+"_Mask.bmp"))
                target_mask_path=os.path.join(BASE_DIR,TRAIN_DIR,'labeled_masks',mask)
                shutil.copy(mask_path, target_mask_path)
                os.remove(os.path.join(os.path.join(BASE_DIR,VAL_DIR,'unlabeled_masks',mask),im.replace(".BMP"," "+mask.replace("GT_","")+"_Mask.bmp")))

        print(len(os.listdir(os.path.join(BASE_DIR,VAL_DIR,'unlabeled_images'))))




def save_model_dict(model,step):
    FILE = f"models/model_step_{step}.pth"
    torch.save(model, FILE)



def random_sampling(sample_size=10):

    
    LEARNING_RATE = 1e-3
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    #DEVICE="cpu"
    BATCH_SIZE = 4
    NUM_EPOCHS = 8
    NUM_WORKERS = 1
    PIN_MEMORY = False
    LOAD_MODEL = False
    LABELED_IMG_DIR = r"C:\Users\taha.DESKTOP-BQA3SEM\Desktop\Stage\AB_UNET\DATA\Labeled_pool\labeled_images"
    LABELED_MASK_DIR = r"C:\Users\taha.DESKTOP-BQA3SEM\Desktop\Stage\AB_UNET\DATA\Labeled_pool\labeled_masks"
    UNLABELED_IMG_DIR = r"C:\Users\taha.DESKTOP-BQA3SEM\Desktop\Stage\AB_UNET\DATA\Unlabeled_pool\unlabeled_images"
    UNLABELED_MASK_DIR = r"C:\Users\taha.DESKTOP-BQA3SEM\Desktop\Stage\AB_UNET\DATA\Unlabeled_pool\unlabeled_masks"
    TEST_IMG_DIR=r"C:\Users\taha.DESKTOP-BQA3SEM\Desktop\Stage\AB_UNET\DATA\Test\test_images"
    TEST_MASK_DIR=r"C:\Users\taha.DESKTOP-BQA3SEM\Desktop\Stage\AB_UNET\DATA\Test\test_masks"


#intial split

    BASE_DIR=r"C:\Users\taha.DESKTOP-BQA3SEM\Desktop\Stage\AB_UNET\DATA"
    LABELED_DIR=r"Labeled_pool"
    UNLABELED_DIR=r"Unlabeled_pool"
    TEST_DIR=r"Test"
    labeled_unlabeled_test_split(BASE_DIR,LABELED_DIR,UNLABELED_DIR,TEST_DIR,label_split_ratio=0.05,test_split_ratio=0.3,shuffle=False)
    num_images=len(os.listdir(r'C:\Users\taha.DESKTOP-BQA3SEM\Desktop\Stage\AB_UNET\DATA\Unlabeled_pool\unlabeled_images'))

#model init    
    model = AB_UNET(in_channels=3, out_channels=4,dropout=0,max_dropout=0.22).to(DEVICE)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scaler = torch.cuda.amp.GradScaler()
    scheduler= torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode='min',patience=3,verbose=True)
    

    #torch.cuda.empty_cache()
    train_transform = A.Compose(
        [   #A.Rotate(limit=35, p=1.0),
            #A.HorizontalFlip(p=0.3),
            #A.VerticalFlip(p=0.1),
            #A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            ToTensorV2()
        ],
    )
    dice_array=[]
    for step in range(int(num_images/sample_size)):
        print(f'step number {step} : ')

        labeled_loader,_,test_loader= get_loaders_active(
            LABELED_IMG_DIR,
            LABELED_MASK_DIR,
            UNLABELED_IMG_DIR,
            UNLABELED_MASK_DIR,
            TEST_IMG_DIR,
            TEST_MASK_DIR,
            BATCH_SIZE,
            train_transform,
            train_transform,
            NUM_WORKERS,
        )
        
        for epoch in tqdm(range(NUM_EPOCHS)):
            scheduler= torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode='min',patience=3,verbose=True)
            train_fn(labeled_loader, model, optimizer, loss_fn, scaler,scheduler)
            if epoch==NUM_EPOCHS-1 : 
                acc,dice=check_accuracy(test_loader,model,BATCH_SIZE, device=DEVICE)
            #save_model_dict(model,step)

        dice_array.append(dice)
        #move 10 images at random to labeled pool
        move_images(BASE_DIR,LABELED_DIR,UNLABELED_DIR,sample_size)
    dice_stats=torch.tensor(dice_array).detach().cpu().numpy()
    np.save(r'C:\Users\taha.DESKTOP-BQA3SEM\Desktop\Stage\AB_UNET\stats\random_featuresinhalf.npy',dice_stats)
    plt.plot(np.arange(1,int(num_images/sample_size)+1,1),dice_stats)
    plt.show()


def Active_sampling(sample_size=10,acquistion_type=1):

    LEARNING_RATE = 1e-3
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    #DEVICE="cpu"
    BATCH_SIZE = 4
    NUM_EPOCHS = 8
    NUM_WORKERS = 1
    PIN_MEMORY = False
    LOAD_MODEL = False
    LABELED_IMG_DIR = r"C:\Users\taha.DESKTOP-BQA3SEM\Desktop\Stage\AB_UNET\DATA\Labeled_pool\labeled_images"
    LABELED_MASK_DIR = r"C:\Users\taha.DESKTOP-BQA3SEM\Desktop\Stage\AB_UNET\DATA\Labeled_pool\labeled_masks"
    UNLABELED_IMG_DIR = r"C:\Users\taha.DESKTOP-BQA3SEM\Desktop\Stage\AB_UNET\DATA\Unlabeled_pool\unlabeled_images"
    UNLABELED_MASK_DIR = r"C:\Users\taha.DESKTOP-BQA3SEM\Desktop\Stage\AB_UNET\DATA\Unlabeled_pool\unlabeled_masks"
    TEST_IMG_DIR=r"C:\Users\taha.DESKTOP-BQA3SEM\Desktop\Stage\AB_UNET\DATA\Test\test_images"
    TEST_MASK_DIR=r"C:\Users\taha.DESKTOP-BQA3SEM\Desktop\Stage\AB_UNET\DATA\Test\test_masks"


#intial split

    BASE_DIR=r"C:\Users\taha.DESKTOP-BQA3SEM\Desktop\Stage\AB_UNET\DATA"
    LABELED_DIR=r"Labeled_pool"
    UNLABELED_DIR=r"Unlabeled_pool"
    TEST_DIR=r"Test"
    labeled_unlabeled_test_split(BASE_DIR,LABELED_DIR,UNLABELED_DIR,TEST_DIR,label_split_ratio=0.05,test_split_ratio=0.3,shuffle=False)
    num_images=len(os.listdir(r'C:\Users\taha.DESKTOP-BQA3SEM\Desktop\Stage\AB_UNET\DATA\Unlabeled_pool\unlabeled_images'))

#model init    
    model = AB_UNET(in_channels=3, out_channels=4,dropout=0,max_dropout=0.22).to(DEVICE)
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
    dice_array=[]
    for step in range(int(num_images/sample_size)):
        print(f'step number {step} : ')
        scheduler= torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode='min',patience=3,verbose=True)
        labeled_loader,unlabeled_loader,test_loader= get_loaders_active(
            LABELED_IMG_DIR,
            LABELED_MASK_DIR,
            UNLABELED_IMG_DIR,
            UNLABELED_MASK_DIR,
            TEST_IMG_DIR,
            TEST_MASK_DIR,
            BATCH_SIZE,
            train_transform,
            train_transform,
            NUM_WORKERS,
        )
        
        for epoch in tqdm(range(NUM_EPOCHS)):
            train_fn(labeled_loader, model, optimizer, loss_fn, scaler,scheduler)
            if epoch==NUM_EPOCHS-1 : 
                acc,dice=check_accuracy(test_loader,model,BATCH_SIZE, device=DEVICE)
            #save_model_dict(model,step)
        dice_array.append(dice)
        score_dict=create_score_dict(model,unlabeled_loader,DEVICE,acquistion_type,4)
        move_images_with_dict(BASE_DIR,LABELED_DIR,UNLABELED_DIR,score_dict,sample_size)
        
    dice_stats=torch.tensor(dice_array).detach().cpu().numpy()
    np.save(r'C:\Users\taha.DESKTOP-BQA3SEM\Desktop\Stage\AB_UNET\stats\entropy_featuresinhalf.npy',np.array(dice_stats))
    plt.plot(np.arange(1,int(num_images/sample_size)+1,1),dice_stats)
    plt.show()




if __name__ == "__main__":
    dice_stats1=np.load(r'C:\Users\taha.DESKTOP-BQA3SEM\Desktop\Stage\AB_UNET\stats\random_featuresinhalf.npy')
    dice_stats2=np.load(r'C:\Users\taha.DESKTOP-BQA3SEM\Desktop\Stage\AB_UNET\stats\entropy_featuresinhalf.npy')
    fig,ax1 = plt.subplots()
    ax2=ax1.twinx()
    ax1.plot(np.arange(1,38,1),dice_stats1)
    ax2.plot(np.arange(1,38,1),dice_stats2,color='r')
    plt.show()
