# encoding: utf-8

import torch.nn as nn



def conv3x3(c_in, c_out, norm):
    return nn.Sequential(
        nn.Conv2d(in_channels=c_in, out_channels=c_out, kernel_size=3, stride=1, padding=1),
        norm(c_out),
        nn.ReLU(inplace=True)
    )


class RevealNet(nn.Module):
    def __init__(self, input_nc, output_nc, norm_layer, output_function):
        super(RevealNet, self).__init__()
        self.conv1 = conv3x3(c_in=input_nc, c_out=64, norm=norm_layer)
        self.conv2 = conv3x3(c_in=64, c_out=128, norm=norm_layer)
        self.conv3 = conv3x3(c_in=128, c_out=256, norm=norm_layer)
        self.conv4 = conv3x3(c_in=256, c_out=128, norm=norm_layer)
        self.conv5 = conv3x3(c_in=128, c_out=64, norm=norm_layer)

        self.output_layer = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=output_nc, kernel_size=3, stride=1, padding=1),
            output_function
        )

    def forward(self, input):
        x = self.conv1(input)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)

        x = self.output_layer(x)
        return x


