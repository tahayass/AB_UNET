import torch
import torch.nn as nn
from PIL import Image
import matplotlib.pyplot as plt
import os
import numpy as np
from DataLoader import BlastocystDataset
from torch.utils.data import DataLoader
import torchvision
import random
import shutil


#Outputs the resulting masks after softmax operation
def output_masks(pred):
  batch=[]
  sm=nn.Softmax(dim=1)
  probs=sm(pred)
  cat=torch.argmax(probs,dim=1)
  for j in range(pred.shape[0]):
    stack=[]
    r=torch.reshape(cat[j],(pred.shape[2],pred.shape[3]))
    r=r.cpu()
    for i in range(pred.shape[1]):
      mask=(np.ones((pred.shape[2],pred.shape[3]))*i)==r.numpy()
      mask=mask*1
      stack.append(mask)
    batch.append(np.stack(stack,axis=0))
  return torch.from_numpy(np.stack(batch,axis=0))


#Return the loader for train and validation data
def get_loaders(
    train_dir,
    train_maskdir,
    val_dir,
    val_maskdir,
    batch_size,
    train_transform,
    val_transform,
    num_workers=4,
    pin_memory=True,
):
    train_ds = BlastocystDataset(
        image_dir=train_dir,
        mask_dir=train_maskdir,
        transform=train_transform,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=True,
    )
    val_ds = BlastocystDataset(
        image_dir=val_dir,
        mask_dir=val_maskdir,
        transform=val_transform,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=False,
    )
    return train_loader,val_loader


#checks accuarcy after each training iteration
def check_accuracy(loader, model,batch_size, device="cuda"):
    num_correct = 0
    num_pixels = 0
    dice_score = 0
    model=model.float()
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.float().to(device)
            y = y.float().to(device).unsqueeze(1)
            preds = output_masks(model(x)).float().to(device)
            
            num_correct += (preds == y).sum()
            num_pixels += torch.numel(preds)*batch_size
            dice_score += (2 * (preds * y).sum()) / ((preds + y).sum() + 1e-8)

    print(f"accuqrcy :{num_correct/num_pixels*100:.2f}")
    print(f"Dice score: {dice_score/len(loader)}")
    model.train()
    return num_correct/num_pixels*100,dice_score/len(loader)



#saves predictions as images (need to be worked on)
def save_predictions_as_imgs(loader, model, folder="saved_images/", device="cuda"):

    model.eval()
    model=model.float()
    for idx, (x, y) in enumerate(loader):
        x = x.float().to(device=device)
        with torch.no_grad():
            preds = output_masks(model(x)).float()
        for j in range(4):
            torchvision.utils.save_image(preds[0][j], f"{folder}/pred_{idx}_{j}.png")
        for i in range(4):
            torchvision.utils.save_image(y[0][i], f"{folder}{idx}_{i}.png")
        if idx > 2 :
            break

    model.train()


#self-explainatory
def train_val_split(BASE_DIR,TRAIN_DIR,VAL_DIR,split_ratio=0.8,shuffle=False):

    os.mkdir(os.path.join(BASE_DIR,TRAIN_DIR))
    os.mkdir(os.path.join(BASE_DIR,VAL_DIR))
    os.mkdir(os.path.join(BASE_DIR,TRAIN_DIR,'train_images'))
    os.mkdir(os.path.join(BASE_DIR,VAL_DIR,'val_images'))
    os.mkdir(os.path.join(BASE_DIR,TRAIN_DIR,'train_masks'))
    os.mkdir(os.path.join(BASE_DIR,VAL_DIR,'val_masks'))

    for mask in os.listdir(os.path.join(BASE_DIR,'masks')):
        os.mkdir(os.path.join(BASE_DIR,TRAIN_DIR,'train_masks',mask))
        os.mkdir(os.path.join(BASE_DIR,VAL_DIR,'val_masks',mask))

    images=os.listdir(os.path.join(BASE_DIR,'images'))

    if shuffle : 
        images=random.shuffle(images)

    train_split=images[:int(len(images)*split_ratio)]
    val_split=images[int(len(images)*split_ratio):]

    for im in train_split : 
        im_path=os.path.join(BASE_DIR,'images',im)
        target_path=os.path.join(BASE_DIR,TRAIN_DIR,'train_images',im)
        shutil.copy(im_path, target_path)

        for mask in os.listdir(os.path.join(BASE_DIR,'masks')):
            mask_path=os.path.join(os.path.join(BASE_DIR,'masks',mask),im.replace(".BMP"," "+mask.replace("GT_","")+"_Mask.bmp"))
            target_mask_path=os.path.join(BASE_DIR,TRAIN_DIR,'train_masks',mask)
            if os.path.exists(target_mask_path)==False:
                os.mkdir(target_mask_path)
            shutil.copy(mask_path, target_mask_path)
            
    for im in val_split : 
        im_path=os.path.join(BASE_DIR,'images',im)
        target_path=os.path.join(BASE_DIR,VAL_DIR,'val_images',im)
        shutil.copy(im_path, target_path)    

        for mask in os.listdir(os.path.join(BASE_DIR,'masks')):
            mask_path=os.path.join(os.path.join(BASE_DIR,'masks',mask),im.replace(".BMP"," "+mask.replace("GT_","")+"_Mask.bmp"))
            target_mask_path=os.path.join(BASE_DIR,VAL_DIR,'val_masks',mask)
            if os.path.exists(target_mask_path)==False:
                os.mkdir(target_mask_path)
            shutil.copy(mask_path, target_mask_path)


