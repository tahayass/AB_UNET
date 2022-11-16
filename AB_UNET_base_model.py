import torch
import sys
from PIL import Image
import numpy as np
import torch.nn as nn
import torchvision.transforms.functional as TF
from torch.utils.tensorboard import SummaryWriter
# default `log_dir` is "runs" 
writer = SummaryWriter('runs/AB-UNET')

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels,dropout=0.05):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            nn.Dropout2d(p=dropout),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.Dropout2d(p=dropout),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)

class AB_UNET(nn.Module):
    def __init__(
            self, in_channels=3, out_channels=1, features=[128, 256, 512, 1024],max_dropout=0.05,dropout=0.05
    ):
        super(AB_UNET, self).__init__()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.max_dropout=nn.Dropout2d(p=max_dropout)

        # Down part of UNET
        for feature in features:
            self.downs.append(DoubleConv(in_channels,feature,dropout))
            in_channels = feature

        # Up part of UNET
        for feature in reversed(features):
            self.ups.append(
                nn.ConvTranspose2d(
                    feature*2, feature, kernel_size=2, stride=2,
                )
            )
            self.ups.append(DoubleConv(feature*2, feature,dropout))

        self.bottleneck = DoubleConv(features[-1], features[-1]*2,dropout)
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)
        #self.softmax = nn.Softmax2d()
        #softmax will be applied implicitly in the nn.CrossEntropyLoss() module

    def forward(self, x):
        skip_connections = []

        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)
            x = self.max_dropout(x)


        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx//2]

            if x.shape != skip_connection.shape:
                x = TF.resize(x, size=skip_connection.shape[2:])

            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx+1](concat_skip)

        return self.final_conv(x)

def test():
    x = torch.randn((3, 3, 16, 16))
    model = AB_UNET(in_channels=3, out_channels=4)
    model=model.float()
    y=model(x)
    print(y.shape)
    #writer.add_graph(model, x)
    #writer.close()
    #sys.exit()
    #assert preds.shape == x.shape
    #print(preds[0][0]+preds[0][1]+preds[0][2])

if __name__ == "__main__":
    test()