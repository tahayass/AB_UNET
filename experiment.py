from random import random
from turtle import color
import torch
import torch.nn as nn
from active_segmentation import random_sampling,Active_sampling
from visualizations import visualise_masks
from utils import reset_DATA
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
import json


def experiment(Config,exp_path=r""):

    print('BEGINNING RANDOM SAMPLING :')
    reset_DATA(os.path.join('.','DATA'))
    random_sampling(
    sample_size=Config['active_step']['sample size'],
    dropout=Config['active_step']['dropout'],
    max_dropout=Config['active_step']['max dropout'],
    label_split_ratio=Config['data']['label split ratio'],
    test_split_ratio=Config['data']['test split ratio'],
    num_epochs=Config['model']['number of epochs'],
    batch_size=Config['model']['batch size'],
    exp_path=exp_path)

    reset_DATA(os.path.join('.','DATA'))
    acq_fn=['entropy','BALD','KL-Divergence','JS-divergence']
    for acq in [1,2,3,4]:
        print(f"BEGINNING {acq_fn[acq-1]} SAMPLING :")
        Active_sampling(sample_size=Config['active_step']['sample size'],
        acquistion_type=acq,
        dropout_iteration=Config['active_step']['dropout iterations'],
        dropout=Config['active_step']['dropout'],
        max_dropout=Config['active_step']['max dropout'],
        label_split_ratio=Config['data']['label split ratio'],
        test_split_ratio=Config['data']['test split ratio'],
        num_epochs=Config['model']['number of epochs'],
        batch_size=Config['model']['batch size'],
        exp_path=exp_path)
        reset_DATA(os.path.join('.','DATA'))


if __name__ == "__main__":
    f = open("Config.json")
    Config = json.load(f)

    experiment_name=Config['experiment name']
    os.mkdir(os.path.join('.','experiments',experiment_name))
    shutil.copy(os.path.join('.','Config.json'),os.path.join('.','experiments',experiment_name))

    experiment(Config=Config,exp_path=os.path.join('.','experiments',experiment_name))



