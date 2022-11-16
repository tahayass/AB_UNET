import os
from PIL import Image
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torchvision
import numpy as np
import torch

def stack_mask(mask_dir,images,idx):
    cls_list=[]
    for cls in os.listdir(mask_dir):
        cls_list.append(cls.replace("GT_",""))
    stack=[]
    for c in cls_list:
        mask_path = os.path.join(os.path.join(mask_dir,"GT_"+c), images[idx].replace(".BMP", " "+c+"_Mask.bmp"))
        label=Image.open(mask_path).convert("L").resize((200,200),resample=Image.NEAREST)
        label=np.asarray(label)/255
        stack.append(label)
    mask=np.zeros((200,200))
    total=np.zeros((200,200))
    for arr in stack:
        total+=arr
    background=(total==mask)*1

    stack.append(background)
    labels=np.stack(stack,axis=0)
    return labels


class BlastocystDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None): #change to accept list of images instead of folder names (way faster than creating new folders and actually moving files)
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = os.listdir(image_dir)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_path = os.path.join(self.image_dir, self.images[index])
        image = np.array(Image.open(img_path).convert("RGB").resize((200,200),resample=Image.NEAREST))
        #image= np.moveaxis(image,-1,0)
        try:
            mask = stack_mask(self.mask_dir,self.images,index).astype(dtype=np.float32)
        except:
            mask=np.zeros((200,200,3))

        if self.transform is not None:
            augmentations = self.transform(image=image, mask=mask)
            image = augmentations["image"]
            mask = augmentations["mask"]

        return image, mask, self.images[index] #add image name to keep track of scores

