# encoding: utf-8

import torch 
import torch.nn as nn



class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, downsample):
        super().__init__()
        if downsample:
            self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1)
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=2),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
            self.shortcut = nn.Sequential()

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, input):
        shortcut = self.shortcut(input)
        input = nn.ReLU()(self.bn1(self.conv1(input)))
        input = nn.ReLU()(self.bn2(self.conv2(input)))
        input = input + shortcut
        return nn.ReLU()(input)



class ResNet18(nn.Module):
    def __init__(self, in_channels, out_channels, output_function):
        super().__init__()
        self.layer0 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )

        self.layer1 = nn.Sequential(
            ResBlock(in_channels=64, out_channels=64, downsample=False),
            ResBlock(64, 64, downsample=False)
        )

        self.layer2 = nn.Sequential(
            ResBlock(64, 128, downsample=True),
            ResBlock(128, 128, downsample=False)
        )

        self.layer3 = nn.Sequential(
            ResBlock(128, 256, downsample=True),
            ResBlock(256, 256, downsample=False)
        )

        self.layer4 = nn.Sequential(
            ResBlock(256, 512, downsample=True),
            ResBlock(512, 512, downsample=False)
        )

        self.upsample_layer = nn.Sequential(
            nn.ConvTranspose2d(in_channels=512, out_channels=128, kernel_size=4, stride=2, padding=1), # x2 
            nn.ConvTranspose2d(in_channels=128, out_channels=32, kernel_size=4, stride=4, padding=0), # x4
            nn.ConvTranspose2d(in_channels=32, out_channels=8, kernel_size=4, stride=4, padding=0), # x4
            nn.BatchNorm2d(8),
            nn.Conv2d(in_channels=8, out_channels=out_channels, kernel_size=5, stride=1, padding=2)
        )

        self.output_layer = output_function


    def forward(self, input):
        input = self.layer0(input) # 3X256x256 -> 64x64x64
        input = self.layer1(input) # 64X256x256 -> 64x64x64
        input = self.layer2(input) # 64X256x256 -> 128x32x32
        input = self.layer3(input) # 128x128x128 -> 256x16x16
        input = self.layer4(input) # 256x64x64 -> 512x8x8

        input = self.upsample_layer(input) # 512x32x32 -> 1x256x256

        return self.output_layer(input)