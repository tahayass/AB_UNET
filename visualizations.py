import numpy as np
from PIL import Image
import os
import torch
from utils import output_masks
import cv2





def visualise_masks(test_dir,output_folder,model):
    images=os.listdir(test_dir)
    
    colors=np.array([[255,0,0],[0,255,0],[0,0,255]])
    for im in images:
        img_path=os.path.join(test_dir,im)
        image = np.array(Image.open(img_path).convert("RGB").resize((300,300),resample=Image.NEAREST))
        original_img=image.copy()
        model_input=torch.from_numpy(image).unsqueeze(0)
        preds = output_masks(model(torch.moveaxis(model_input,-1,1).float().to('cuda')))
        print()

        for i in range(3):
            mask=preds[0][i].numpy()
            image= np.where(mask[...,None], colors[i], image)

        masked_img=cv2.addWeighted(np.asarray(original_img,np.uint8), 0.6, np.asarray(image,np.uint8), 0.4,0)
        name=im.replace(".BMP","")
        cv2.imwrite(f"{output_folder}\{name}_masked.png",masked_img)

def main():
    test_dir=r"C:\Users\taha.DESKTOP-BQA3SEM\Desktop\Stage\AB_UNET\test_images"
    model=torch.load(r"C:\Users\taha.DESKTOP-BQA3SEM\Desktop\Stage\AB_UNET\model.pth")
    out=r"C:\Users\taha.DESKTOP-BQA3SEM\Desktop\Stage\AB_UNET\test_img_output"
    visualise_masks(test_dir,out,model)


if __name__ == "__main__":
    main()









