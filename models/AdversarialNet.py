# encoding: utf-8

from turtle import forward
import torch
import torch.nn as nn



class Adversary(nn.Module):
    def __init__(self, input_nc, output_nc):
        super(Adversary, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels=input_nc, out_channels=input_nc, kernel_size=3, stride=1, padding=1)
        self.active = nn.LeakyReLU(0.2, inplace=True)
        self.conv2 = nn.Conv2d(in_channels=input_nc, out_channels=output_nc, kernel_size=3, stride=1, padding=1)

    def forward(self, input):
        output = self.conv2(self.active(self.conv1(input)))

        return output