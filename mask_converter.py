import numpy as np
from PIL import Image
import os
import matplotlib.pyplot as plt


rgb_dict=dict()
rgb_dict['ICM']=np.array([51,221,255])
rgb_dict['TE']=np.array([250,50,83])
rgb_dict['ZP']=np.array([61,245,61])
rgb_dict['background']=np.array([0,0,0])


images=os.listdir(r".\DATA_AO_raw\Step_2\SegmentationClass")
for image_name in images:
    ICM_mask=np.zeros((400,400))
    TE_mask=np.zeros((400,400))
    ZP_mask=np.zeros((400,400))
    img_path=os.path.join(r".\DATA_AO_raw\Step_2\SegmentationClass",image_name)
    image = np.array(Image.open(img_path).convert("RGB"))
    #ICM
    for i in range(400):
        for j in range(400):
            if (image[i,j,:]==rgb_dict['ICM']).all():
                ICM_mask[i,j]=255
            else:
                ICM_mask[i,j]=0
    im = Image.fromarray(ICM_mask).convert('L')
    mask_path=os.path.join(r".\DATA_AO_preprocessed\labeled_pool\labeled_masks\GT_ICM",f"{image_name} ICM_Mask.bmp".replace(".png",""))
    im.save(mask_path)
    #TE
    for i in range(400):
        for j in range(400):
            if (image[i,j,:]==rgb_dict['TE']).all():
                TE_mask[i,j]=255
            else:
                TE_mask[i,j]=0
    im = Image.fromarray(TE_mask).convert('L')
    mask_path=os.path.join(r".\DATA_AO_preprocessed\labeled_pool\labeled_masks\GT_TE",f"{image_name} TE_Mask.bmp".replace(".png",""))
    im.save(mask_path)
    #ZP
    for i in range(400):
        for j in range(400):
            if (image[i,j,:]==rgb_dict['ZP']).all():
                ZP_mask[i,j]=255
            else:
                ZP_mask[i,j]=0
    im = Image.fromarray(ZP_mask).convert('L')
    mask_path=os.path.join(r".\DATA_AO_preprocessed\labeled_pool\labeled_masks\GT_ZP",f"{image_name} ZP_Mask.bmp".replace(".png",""))
    im.save(mask_path)






