import torch
import torch.nn as nn
from PIL import Image
import matplotlib.pyplot as plt
import os
import numpy as np
from DataLoader import BlastocystDataset
from torch.utils.data import DataLoader
import torchvision



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
    return train_loader
"""
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
"""

def check_accuracy(loader, model, device="cuda"):
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
            num_pixels += torch.numel(preds)*4
            dice_score += (2 * (preds * y).sum()) / (
                (preds + y).sum() + 1e-8
            )

    print(
        f"Got {num_correct}/{num_pixels} with acc {num_correct/num_pixels*100:.2f}"
    )
    print(f"Dice score: {dice_score/len(loader)}")
    model.train()

def save_predictions_as_imgs(
    loader, model, folder="saved_images/", device="cuda"
):
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