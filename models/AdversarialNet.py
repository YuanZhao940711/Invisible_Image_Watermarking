# encoding: utf-8

from turtle import forward
import torch
import torch.nn as nn



class AdversaryConv(nn.Module):
    def __init__(self, input_nc, output_nc):
        super(AdversaryConv, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels=input_nc, out_channels=input_nc, kernel_size=3, stride=1, padding=1) # 3*256*256 -> 3*64*64
        self.active = nn.LeakyReLU(0.2, inplace=True)
        self.conv2 = nn.Conv2d(in_channels=input_nc, out_channels=output_nc, kernel_size=5, stride=1, padding=2) # 3*64*64 -> 3*256*256

    def forward(self, input):
        output = self.conv2(self.active(self.conv1(input)))

        return output



class AdversaryTransConv(nn.Module):
    def __init__(self, input_nc, output_nc):
        super(AdversaryTransConv, self).__init__()
        
        self.conv = nn.Conv2d(in_channels=input_nc, out_channels=input_nc, kernel_size=4, stride=4, padding=0) # 3*256*256 -> 3*64*64
        self.active = nn.LeakyReLU(0.2, inplace=True)
        self.deconv = nn.ConvTranspose2d(in_channels=input_nc, out_channels=output_nc, kernel_size=4, stride=4, padding=0) # 3*64*64 -> 3*256*256

    def forward(self, input):
        output = self.deconv(self.active(self.conv(input)))

        return output