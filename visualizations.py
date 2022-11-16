import numpy as np
from PIL import Image
import os
import torch
from utils import output_masks
import cv2
import matplotlib.pyplot as plt
import albumentations as A





def visualise_masks(test_dir,output_folder,model):

    images=os.listdir(test_dir)
    colors=np.array([[255,0,0],[0,255,0],[0,0,255]])
    for im in images:
        img_path=os.path.join(test_dir,im)
        image = np.array(Image.open(img_path).convert("RGB").resize((200,200),resample=Image.NEAREST))
        #transform=A.augmentations.transforms.Equalize(mode="cv",p=1)
        #augmentations =transform(image=image, mask=None)
        #image = augmentations["image"]
        original_img=np.array(Image.open(img_path).convert("RGB"))
        modified_img=original_img.copy()
        height,width,_=original_img.shape
        model_input=torch.from_numpy(image).unsqueeze(0)
        print(model_input.shape)
        preds = output_masks(model(torch.moveaxis(model_input,-1,1).float().to('cuda')))

        for i in range(3):
            mask=preds[0][i].numpy()
            mask_image=Image.fromarray(mask*255)
            mask_image=mask_image.resize((width,height),resample=Image.NEAREST)
            mask=np.array(mask_image)/255
            modified_img= np.where(mask[...,None], colors[i], modified_img)

        masked_img=cv2.addWeighted(np.asarray(original_img,np.uint8), 0.6, np.asarray(modified_img,np.uint8), 0.4,0)
        name=im.replace(".jpg","")
        cv2.imwrite(f"{output_folder}\{name}_masked.png",masked_img)

def main():
    test_dir=r"C:\Users\taha.DESKTOP-BQA3SEM\Desktop\Stage\AB_UNET\test_images"
    model=torch.load(r"C:\Users\taha.DESKTOP-BQA3SEM\Desktop\Stage\AB_UNET\model_30epochs_AO.pth")
    out=r"C:\Users\taha.DESKTOP-BQA3SEM\Desktop\Stage\AB_UNET\test_img_output"
    visualise_masks(test_dir,out,model)


if __name__ == "__main__":
    main()









