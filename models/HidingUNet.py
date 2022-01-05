# encoding: utf-8

import torch
import torch.nn as nn



def conv4x4(c_in, c_out, norm):
    return nn.Sequential(
        nn.Conv2d(in_channels=c_in, out_channels=c_out, kernel_size=4, stride=2, padding=1),
        norm(c_out),
        nn.LeakyReLU(0.2, inplace=True)
    )


class deconv4x4(nn.Module):
    def __init__(self, c_in, c_out, norm):
        super(deconv4x4, self).__init__()
        self.deconv = nn.ConvTranspose2d(in_channels=c_in, out_channels=c_out, kernel_size=4, stride=2, padding=1)
        self.norm = norm(c_out)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, input, feat):
        x = self.deconv(input)
        x = self.norm(x)
        x = self.relu(x)
        return torch.cat((x, feat), dim=1)


class UnetGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, norm_layer, output_function):
        super(UnetGenerator, self).__init__()
        self.conv1 = conv4x4(c_in=input_nc, c_out=32, norm=norm_layer)
        self.conv2 = conv4x4(c_in=32, c_out=64, norm=norm_layer)
        self.conv3 = conv4x4(c_in=64, c_out=128, norm=norm_layer)
        self.conv4 = conv4x4(c_in=128, c_out=256, norm=norm_layer)
        self.conv5 = conv4x4(c_in=256, c_out=512, norm=norm_layer)
        self.conv6 = conv4x4(c_in=512, c_out=1024, norm=norm_layer)
        self.conv7 = conv4x4(c_in=1024, c_out=1024, norm=norm_layer)

        self.deconv1 = deconv4x4(c_in=1024, c_out=1024, norm=norm_layer)
        self.deconv2 = deconv4x4(c_in=2048, c_out=512, norm=norm_layer)
        self.deconv3 = deconv4x4(c_in=1024, c_out=256, norm=norm_layer)
        self.deconv4 = deconv4x4(c_in=512, c_out=128, norm=norm_layer)
        self.deconv5 = deconv4x4(c_in=256, c_out=64, norm=norm_layer)
        self.deconv6 = deconv4x4(c_in=128, c_out=32, norm=norm_layer)

        self.output_layer = nn.Sequential(
            nn.ConvTranspose2d(in_channels=64, out_channels=output_nc, kernel_size=4, stride=2, padding=1),
            output_function
        )
        self.factor = 1 #10/255 # !

    def forward(self, Xt):
        down_feat1 = self.conv1(Xt) # Xt: 3x256x256 -> down_feat1: 32x128x128
        down_feat2 = self.conv2(down_feat1) # down_feat2: 64x64x64
        down_feat3 = self.conv3(down_feat2) # down_feat3: 128x32x32
        down_feat4 = self.conv4(down_feat3) # down_feat4: 256x16x16
        down_feat5 = self.conv5(down_feat4) # down_feat5: 512x8x8
        down_feat6 = self.conv6(down_feat5) # down_feat6: 1024x4x4

        up_feat1 = self.conv7(down_feat6) # up_feat1: 1024x2x2

        up_feat2 = self.deconv1(input=up_feat1, feat=down_feat6) # input(1024x2x2->1024x4x4) + feat(1024x4x4) = up_feat2(2048x4x4)
        up_feat3 = self.deconv2(input=up_feat2, feat=down_feat5) # input(2048x4x4->512x8x8) + feat(512x8x8) = up_feat3(1024x8x8)
        up_feat4 = self.deconv3(input=up_feat3, feat=down_feat4) # input(1024x8x8->256x16x16) + feat(256x16x16) = up_feat4(512x16x16)
        up_feat5 = self.deconv4(input=up_feat4, feat=down_feat3) # input(512x16x16->128x32x32) + feat(128x32x32) = up_feat5(256x32x32)
        up_feat6 = self.deconv5(input=up_feat5, feat=down_feat2) # input(256x32x32->64x64x64) + feat(64x64x64) = up_feat6(128x64x64)
        up_feat7 = self.deconv6(input=up_feat6, feat=down_feat1) # input(128x64x64->32x128x128) + feat(32x128x128) = up_feat7(64x128x128)

        x = self.output_layer(up_feat7) # up_feat7(64x128x128) -> x(3x256x256)
        return self.factor * x