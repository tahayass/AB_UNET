import numpy as np
from PIL import Image
import os
import torch
from utils import output_masks
from DataLoader import stack_mask
import cv2
import matplotlib.pyplot as plt
import albumentations as A
from tqdm import tqdm
import PIL





def visualise_masks(test_dir,output_folder,model,mode='pred'):

    if mode=='gt':
        blue=[0,0,255]
        red=[255,0,0]
        green=[0,255,0]
        colors=np.array([red,green,blue])
        for im in tqdm(os.listdir(test_dir)):
            mask_dir=os.path.join('.','DATA_AO_preprocessed','Labeled_pool','masks')
            cls_list=[]
            for cls in os.listdir(mask_dir):
                cls_list.append(cls.replace("GT_",""))
            stack=[]
            for c in cls_list:
                mask_path = os.path.join(os.path.join(mask_dir,"GT_"+c), im.replace(".jpg", " "+c+"_Mask.bmp"))
                label=Image.open(mask_path).convert("L").resize((200,200),resample=Image.NEAREST)
                label=np.asarray(label)/255
                stack.append(label)
            mask_bg=np.zeros((200,200))
            total=np.zeros((200,200))
            for arr in stack:
                total+=arr
            background=(total==mask_bg)*1
            stack.append(background)
            mask=np.stack(stack,axis=0)

            img_path=os.path.join(test_dir,im)
            image = np.array(Image.open(img_path).convert("RGB").resize((200,200),resample=Image.NEAREST))
            original_img=np.array(Image.open(img_path).convert("RGB"))
            modified_img=original_img.copy()
            height,width,_=original_img.shape  

            for i in range(3):
                mask_image=Image.fromarray(mask[i]*255)                
                mask_image=mask_image.resize((width,height),resample=Image.NEAREST)
                mask_array=np.array(mask_image)/255
                modified_img= np.where(mask_array[...,None], colors[i], modified_img)

            masked_img=cv2.addWeighted(np.asarray(original_img,np.uint8), 0.6, np.asarray(modified_img,np.uint8), 0.4,0)
            name=im.replace(".BMP","")
            cv2.imwrite(f"{output_folder}\{name}_masked.png",masked_img)


    if mode=='pred':
        kernel=np.ones((5,5),np.uint8)
        images=os.listdir(test_dir)
        colors=np.array([[0,255,0],[0,0,255],[255,0,0]])
        for im in tqdm(images):
            img_path=os.path.join(test_dir,im)
            image = np.array(Image.open(img_path).convert("RGB").resize((200,200),resample=Image.NEAREST))
            #transform=A.augmentations.transforms.Equalize(mode="cv",p=1)
            #augmentations =transform(image=image, mask=None)
            #image = augmentations["image"]
            original_img=np.array(Image.open(img_path).convert("RGB"))
            modified_img=original_img.copy()
            height,width,_=original_img.shape
            model_input=torch.from_numpy(image).unsqueeze(0)
            preds = output_masks(model(torch.moveaxis(model_input,-1,1).float().to('cuda')))

            for i in range(3):
                mask=preds[0][i].numpy()
                #mask = cv2.morphologyEx((mask*255).astype('uint8'), cv2.MORPH_OPEN, kernel)
                #mask = cv2.morphologyEx((mask*255).astype('uint8'), cv2.MORPH_CLOSE, kernel)
                mask_image=Image.fromarray(mask*255)                                
                mask_image=mask_image.resize((width,height),resample=Image.NEAREST)
                mask=np.array(mask_image)/255
                modified_img= np.where(mask[...,None], colors[i], modified_img)

            masked_img=cv2.addWeighted(np.asarray(original_img,np.uint8), 0.6, np.asarray(modified_img,np.uint8), 0.4,0)
            name=im.replace(".jpg","")
            cv2.imwrite(f"{output_folder}\{name}_masked.png",masked_img)

def main():
    test_dir=r"C:\Users\taha.DESKTOP-BQA3SEM\Desktop\Stage\AB_UNET\test_images"
    model=torch.load(r".\KL_div_model.pth")
    out=r"C:\Users\taha.DESKTOP-BQA3SEM\Desktop\Stage\AB_UNET\test_img_output"
    visualise_masks(test_dir=test_dir,output_folder=out,model=model,mode='gt')


if __name__ == "__main__":
    main()
