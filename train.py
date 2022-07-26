import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from AB_UNET_base_model import AB_UNET
import DataLoader
from utils import get_loaders,check_accuracy,save_predictions_as_imgs


# Hyperparameters.
LEARNING_RATE = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
#DEVICE="cpu"
BATCH_SIZE = 4
NUM_EPOCHS = 20
NUM_WORKERS = 1
IMAGE_HEIGHT = 160  # 1280 originally
IMAGE_WIDTH = 240  # 1918 originally
PIN_MEMORY = False
LOAD_MODEL = False
TRAIN_IMG_DIR = r"C:\Users\taha.DESKTOP-BQA3SEM\Desktop\Stage\AB_UNET\DATA\images"
TRAIN_MASK_DIR = r"C:\Users\taha.DESKTOP-BQA3SEM\Desktop\Stage\AB_UNET\DATA\masks"
VAL_IMG_DIR = ""
VAL_MASK_DIR = ""

def train_fn(loader, model, optimizer, loss_fn, scaler):
    loop = tqdm(loader)

    for batch_idx, (data, targets) in enumerate(loop):
        data = data.float().to(device=DEVICE)
        targets = targets.float().to(device=DEVICE)

        # forward
        with torch.cuda.amp.autocast():
            predictions = model(data)
            loss = loss_fn(predictions, targets)

        # backward
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # update tqdm loop
        loop.set_postfix(loss=loss.item())

def main():
    #torch.cuda.empty_cache()
    train_transform = A.Compose(
        [   #A.Rotate(limit=35, p=1.0),
            #A.HorizontalFlip(p=0.3),
            #A.VerticalFlip(p=0.1),
            #A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            ToTensorV2()
        ],
    )
    model = AB_UNET(in_channels=3, out_channels=4).to(DEVICE)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    train_loader= get_loaders(
        TRAIN_IMG_DIR,
        TRAIN_MASK_DIR,
        VAL_IMG_DIR,
        VAL_MASK_DIR,
        BATCH_SIZE,
        train_transform,
        None,
        NUM_WORKERS,
    )
    scaler = torch.cuda.amp.GradScaler()
    for epoch in tqdm(range(NUM_EPOCHS)):
        train_fn(train_loader, model, optimizer, loss_fn, scaler)


        # check accuracy
        check_accuracy(train_loader, model, device=DEVICE)

    save_predictions_as_imgs(train_loader, model, folder="saved_images/", device="cuda")
    FILE = "model.pth"
    torch.save(model, FILE)



if __name__ == "__main__":
    main()
