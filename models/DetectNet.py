# encoding: utf-8

import torch 


def conv3x3(c_in, c_out, norm):
    return torch.nn.Sequential(
        torch.nn.Conv2d(in_channels=c_in, out_channels=c_out, kernel_size=3, stride=1, padding=1),
        norm(c_out),
        torch.nn.ReLU(inplace=True)
    )


class DetectNet(torch.nn.Module):
    def __init__(self, input_nc, output_nc, norm_layer, output_function):
        super(DetectNet, self).__init__()
        self.conv1 = conv3x3(c_in=input_nc, c_out=16, norm=norm_layer) 
        self.conv2 = conv3x3(c_in=16, c_out=32, norm=norm_layer) 
        self.conv3 = conv3x3(c_in=32, c_out=64, norm=norm_layer) 
        self.conv4 = conv3x3(c_in=64, c_out=32, norm=norm_layer) 
        self.conv5 = conv3x3(c_in=32, c_out=16, norm=norm_layer) 

        self.output_layer = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=16, out_channels=output_nc, kernel_size=3, stride=1, padding=1), 
            output_function
        ) 

    def forward(self, input):
        feat = self.conv1(input)
        feat = self.conv2(feat)
        feat = self.conv3(feat)
        feat = self.conv4(feat)
        feat = self.conv5(feat)
        
        return self.output_layer(feat)