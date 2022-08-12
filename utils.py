from typing import Dict
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
from tqdm import tqdm

from acquisition_functions import score_entropy, score_js_score, score_kl_divergence



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

def get_loaders_active(
    labeled_dir,
    labeled_maskdir,
    unlabeled_dir,
    unlabeled_maskdir,
    test_dir,
    test_maskdir,
    batch_size,
    train_transform,
    val_transform,
    num_workers=4,
    pin_memory=True,
):
    labeled_ds = BlastocystDataset(
        image_dir=labeled_dir,
        mask_dir=labeled_maskdir,
        transform=train_transform,
    )

    labeled_loader = DataLoader(
        labeled_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=True,
    )
    unlabeled_ds = BlastocystDataset(
        image_dir=unlabeled_dir,
        mask_dir=unlabeled_maskdir,
        transform=val_transform,
    )
    unlabeled_loader = DataLoader(
        unlabeled_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=False,
    )
    test_ds = BlastocystDataset(
        image_dir=test_dir,
        mask_dir=test_maskdir,
        transform=val_transform,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=False,
    )
    return labeled_loader,unlabeled_loader,test_loader


#checks accuarcy after each training iteration
def check_accuracy(loader, model,batch_size, device="cuda"):
    num_correct = 0
    num_pixels = 0
    dice_score = 0
    model=model.float()
    model.eval()

    with torch.no_grad():
        for x, y , _ in loader:
            x = x.float().to(device)
            y = y.float().to(device).unsqueeze(1)
            preds = output_masks(model(x)).float().to(device)
            num_correct += (preds == y).sum()
            num_pixels += torch.numel(preds)*batch_size
            dice_score += (2 * (preds * y).sum()) / ((preds + y).sum() + 1e-8)

    print(f"accuracy :{num_correct/num_pixels*100:.2f}")
    print(f"Dice score: {dice_score/len(loader)}")
    model.train()
    return num_correct/num_pixels*100,dice_score/len(loader)

def check_accuracy_batch(x,y,model,batch_size, device="cuda"):
    num_correct = 0
    num_pixels = 0
    dice_score = 0
    model=model.float()
    model.eval()

    with torch.no_grad():
        x = x.float().to(device)
        y = y.float().to(device).unsqueeze(1)
        preds = output_masks(model(x)).float().to(device)
        num_correct = (preds == y).sum()
        num_pixels = torch.numel(preds)*batch_size
        dice_score = (2 * (preds * y).sum()) / ((preds + y).sum() + 1e-8)
    model.train()
    return num_correct/num_pixels*100,dice_score



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
        random.shuffle(images)

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

def labeled_unlabeled_test_split(BASE_DIR,LABELED_DIR,UNLABELED_DIR,TEST_DIR,label_split_ratio=0.1,test_split_ratio=0.3,shuffle=False):

    os.mkdir(os.path.join(BASE_DIR,LABELED_DIR))
    os.mkdir(os.path.join(BASE_DIR,UNLABELED_DIR))
    os.mkdir(os.path.join(BASE_DIR,TEST_DIR))
    os.mkdir(os.path.join(BASE_DIR,LABELED_DIR,'labeled_images'))
    os.mkdir(os.path.join(BASE_DIR,UNLABELED_DIR,'unlabeled_images'))
    os.mkdir(os.path.join(BASE_DIR,LABELED_DIR,'labeled_masks'))
    os.mkdir(os.path.join(BASE_DIR,UNLABELED_DIR,'unlabeled_masks'))
    os.mkdir(os.path.join(BASE_DIR,TEST_DIR,'test_images'))
    os.mkdir(os.path.join(BASE_DIR,TEST_DIR,'test_masks'))

    for mask in os.listdir(os.path.join(BASE_DIR,'masks')):
        os.mkdir(os.path.join(BASE_DIR,LABELED_DIR,'labeled_masks',mask))
        os.mkdir(os.path.join(BASE_DIR,UNLABELED_DIR,'unlabeled_masks',mask))
        os.mkdir(os.path.join(BASE_DIR,TEST_DIR,'test_masks',mask))

    images=os.listdir(os.path.join(BASE_DIR,'images'))

    if shuffle : 
        random.shuffle(images)
    test_split=images[:int(len(images)*test_split_ratio)]
    labeled_split=images[int(len(images)*test_split_ratio):int(len(images)*(test_split_ratio+label_split_ratio))]
    unlabeled_split=images[int(len(images)*(test_split_ratio+label_split_ratio)):]

    for im in labeled_split : 
        im_path=os.path.join(BASE_DIR,'images',im)
        target_path=os.path.join(BASE_DIR,LABELED_DIR,'labeled_images',im)
        shutil.copy(im_path, target_path)

        for mask in os.listdir(os.path.join(BASE_DIR,'masks')):
            mask_path=os.path.join(os.path.join(BASE_DIR,'masks',mask),im.replace(".BMP"," "+mask.replace("GT_","")+"_Mask.bmp"))
            target_mask_path=os.path.join(BASE_DIR,LABELED_DIR,'labeled_masks',mask)
            if os.path.exists(target_mask_path)==False:
                os.mkdir(target_mask_path)
            shutil.copy(mask_path, target_mask_path)
            
    for im in unlabeled_split : 
        im_path=os.path.join(BASE_DIR,'images',im)
        target_path=os.path.join(BASE_DIR,UNLABELED_DIR,'unlabeled_images',im)
        shutil.copy(im_path, target_path)    

        for mask in os.listdir(os.path.join(BASE_DIR,'masks')):
            mask_path=os.path.join(os.path.join(BASE_DIR,'masks',mask),im.replace(".BMP"," "+mask.replace("GT_","")+"_Mask.bmp"))
            target_mask_path=os.path.join(BASE_DIR,UNLABELED_DIR,'unlabeled_masks',mask)
            if os.path.exists(target_mask_path)==False:
                os.mkdir(target_mask_path)
            shutil.copy(mask_path, target_mask_path)

    for im in test_split : 
        im_path=os.path.join(BASE_DIR,'images',im)
        target_path=os.path.join(BASE_DIR,TEST_DIR,'test_images',im)
        shutil.copy(im_path, target_path)    

        for mask in os.listdir(os.path.join(BASE_DIR,'masks')):
            mask_path=os.path.join(os.path.join(BASE_DIR,'masks',mask),im.replace(".BMP"," "+mask.replace("GT_","")+"_Mask.bmp"))
            target_mask_path=os.path.join(BASE_DIR,TEST_DIR,'test_masks',mask)
            if os.path.exists(target_mask_path)==False:
                os.mkdir(target_mask_path)
            shutil.copy(mask_path, target_mask_path)


def stochastic_prediction(model,batch,dropout_iteration,device):

    model.eval().float().to(device)
    for m in model.modules():
        if m.__class__.__name__.startswith('Dropout'):
            m.train().to(device)
    #batch.detach()
    x=torch.zeros((batch.shape[0],4,batch.shape[2],batch.shape[3])).to(device)
    softmax=nn.Softmax(dim=1)
    with torch.no_grad():
        for i in range(dropout_iteration):
            x += softmax(model(batch))
        x /= dropout_iteration
    del batch
    torch.cuda.empty_cache()


    return x


def standard_prediction(model,batch,device):

    model.float().eval().to(device)
    batch.detach()
    softmax=nn.Softmax(dim=1)
    return softmax(model(batch))


def create_score_dict(model,loader,device,acquisition_type,dropout_iteration):
    
    score_dict=dict()
    #model= model.float().to(device)
    if acquisition_type==1 :
        print("Calculating entropy scores ..")
        for batch_idx, (data, _, image_name) in tqdm(enumerate(loader)):
            #model = torch.load('model.pth')
            data = data.float().to(device)
            st_pred=stochastic_prediction(model,data,dropout_iteration,device)
            with torch.no_grad():
                scores=score_entropy(st_pred.cpu().numpy())
                for name,score in zip(image_name,torch.from_numpy(scores).to(device)):
                    score_dict[name]=score.item()
            data.detach()
            del data
            torch.cuda.empty_cache()
        score_dict=dict(sorted(score_dict.items(), key=lambda item: item[1],reverse=True))
        return score_dict

    elif acquisition_type==2 :
        print("Calculating BALD scores .. ")
        for batch_idx, (data, _, image_name) in tqdm(enumerate(loader)):
            #model = torch.load('model.pth')
            data = data.float().to(device)
            stn_prediction=standard_prediction(model,data,device)
            with torch.no_grad():
                stn_entropy_score=score_entropy(stn_prediction.cpu().numpy())
            st_entropy_score=np.zeros(data.shape[0])
            with torch.no_grad():
                for i in range(dropout_iteration):
                    st_entropy_score+=score_entropy(stochastic_prediction(model,data,1,device).cpu().numpy())
                st_entropy_score /= dropout_iteration
                scores=stn_entropy_score-st_entropy_score
                for name,score in zip(image_name,torch.from_numpy(scores).to(device)):
                    score_dict[name]=score.item()
            data.detach()
            del data
            torch.cuda.empty_cache()
        score_dict=dict(sorted(score_dict.items(), key=lambda item: item[1],reverse=True))
        return score_dict

    elif acquisition_type==3 :
        print("Calculating KL-divergence scores .. ")
        for batch_idx, (data, _, image_name) in tqdm(enumerate(loader)):
            #model = torch.load('model.pth')
            data = data.float().to(device)
            st_pred=stochastic_prediction(model,data,dropout_iteration,device)
            stn_prediction=standard_prediction(model,data,device)
            with torch.no_grad():
                scores=score_kl_divergence(stn_prediction.cpu().numpy(),st_pred.cpu().numpy())
                for name,score in zip(image_name,torch.from_numpy(scores).to(device)):
                    score_dict[name]=score.item()
            data.detach()
            del data
            torch.cuda.empty_cache()
        score_dict=dict(sorted(score_dict.items(), key=lambda item: item[1],reverse=True))
        return score_dict

    elif acquisition_type==4 :
        print("Calculating JS-divergence scores .. ")
        for batch_idx, (data, _, image_name) in tqdm(enumerate(loader)):
            #model = torch.load('model.pth')
            data = data.float().to(device)
            st_pred=stochastic_prediction(model,data,dropout_iteration,device)
            stn_prediction=standard_prediction(model,data,device)
            with torch.no_grad():
                scores=score_js_score(stn_prediction.cpu().numpy(),st_pred.cpu().numpy())
                for name,score in zip(image_name,torch.from_numpy(scores).to(device)):
                    score_dict[name]=score.item()
            data.detach()
            del data
            torch.cuda.empty_cache()
        score_dict=dict(sorted(score_dict.items(), key=lambda item: item[1],reverse=True))
        return score_dict

    return score_dict



            







    


    
    

    


