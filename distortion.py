import os
import cv2
import math

import numpy as np

import torch
from torchvision import transforms


def draw_polygon(sides, radius=1, rotation=0, location=None):
    one_segment = math.pi * 2 / sides

    if type(radius) == list:
        points = [(math.sin(one_segment * i + rotation) * radius[i], math.cos(one_segment * i + rotation) * radius[i]) for i in range(sides)]
    else:
        points = [(math.sin(one_segment * i + rotation) * radius, math.cos(one_segment * i + rotation) * radius) for i in range(sides)]

    if location is not None:
        points = np.array([[sum(pair) for pair in zip(point, location)] for point in points], dtype=np.int32)

    return points    



def random_mask(opt, image):
    _, img_channel, img_hight, img_width = image.size()
    image_dis = torch.zeros_like(image)
    mask = torch.zeros_like(image)

    for idx, img_ori in enumerate(image):
        sides = np.random.randint(low=3, high=15, size=1)[0]
        radius = list(np.random.randint(low=5, high=50, size=sides))
        location = np.random.randint(low=max(radius), high=min(img_hight, img_width)-max(radius), size=2)

        contour_pts = draw_polygon(sides=sides, radius=radius, rotation=0, location=location) # (x,y)

        # The original pixel is 0, the masked pixel is 1
        mask_ori = np.zeros((img_hight, img_width, img_channel), dtype=np.uint8)
        cv2.drawContours(image=mask_ori, contours=[contour_pts], contourIdx=-1, color=(255,255,255), thickness=-1)

        mask_ori = transforms.Compose([transforms.ToTensor()])(mask_ori).to(opt.device)
        mask_inv = torch.ones_like(mask_ori) - mask_ori
        mask_dis = mask_ori * torch.rand_like(mask_ori)

        img_dis = img_ori * mask_dis + img_ori * mask_inv

        image_dis[idx] = img_dis
        mask[idx] = mask_ori 

    return image_dis, mask



def block_mask(opt, image):
    batch_size, img_channel, img_hight, img_width = image.size()
    # The original pixel is 0, the masked pixel is 1
    #masks = (torch.rand((batch_size, 1, img_hight, img_width)) > 0.5).int().to(opt.device)

    block_number = img_hight // opt.block_size # bigger block size correspond to smaller masked block

    block_mask = (torch.rand((batch_size,1,opt.block_size,opt.block_size)) > opt.block_ratio).float()

    masks = torch.repeat_interleave(input=block_mask, repeats=block_number, dim=2)

    masks = torch.repeat_interleave(input=masks, repeats=block_number, dim=3).to(opt.device)

    masked = masked * torch.rand_like(masked)
    image_masked = image * masked + image * (torch.ones_like(masks) - masks)
    #image_masked = image * (torch.ones_like(masks) - masks)

    image_masked = image_masked.to(opt.device)

    return image_masked, masks


def draw_cicle(image_size, diamiter):
    assert len(image_size) == 2
    TF = np.zeros(image_size,dtype=bool)
    center = np.array(TF.shape)/2.0

    for iy in range(image_size[0]):
        for ix in range(image_size[1]):
            TF[iy,ix] = (iy- center[0])**2 + (ix - center[1])**2 < diamiter **2
    return(TF)
def filter_circle(TFcircleLow, fft_img_channel):
    temp = np.zeros(fft_img_channel.shape[:2], dtype=complex)
    temp[TFcircleLow] = fft_img_channel[TFcircleLow]
    return(temp)
def inv_FFT_all_channel(fft_img):
    img_rec = []
    for channel_idx in range(fft_img.shape[0]):
        img_rec.append(np.fft.ifft2(np.fft.ifftshift(fft_img[channel_idx, :, :])))
    img_rec = np.array(img_rec) # (3, 256, 256)
    return(img_rec)        
def lowpass_filter(image_batch):
    image_lowfreq = np.zeros((image_batch.shape)) # (bs, 3, 256, 256)

    for idx, image in enumerate(image_batch):
        image = image.cpu().detach().numpy().transpose([1,2,0]) # (256, 256, 3)
            
        image_size = image.shape[:2]
        diamiter = 50

        TFcircleLow = draw_cicle(image_size, diamiter)

        fft_img = np.zeros_like(image, dtype=complex)
        for channel_idx in range(fft_img.shape[2]):
            fft_img[:,:,channel_idx] = np.fft.fftshift(np.fft.fft2(image[:,:,channel_idx]))
            
        fft_img_filtered_low = []
        for channel_idx in range(fft_img.shape[2]):
            fft_img_channel  = fft_img[:,:,channel_idx]

            ### circle low ###
            temp = filter_circle(TFcircleLow, fft_img_channel)
            fft_img_filtered_low.append(temp)
            
        fft_img_filtered_low = np.array(fft_img_filtered_low) # (3, 256, 256)

        img_rec_low  = inv_FFT_all_channel(fft_img_filtered_low) # (3, 256, 256)
        image_lowfreq[idx,:,:,:] = img_rec_low
        
    image_lowfreq = torch.from_numpy(image_lowfreq).type(torch.float)

    return image_lowfreq


def jpeg_compression(opt, image_batch):
    image_ori = image_batch.detach().cpu().numpy()

    image = np.transpose(image_ori, (0,2,3,1))

    N, _, _, _ = image.shape
    image = (np.clip(image, 0.0, 1.0)*255).astype(np.uint8)

    for i in range(N):
        img = cv2.cvtColor(image[i], cv2.COLOR_RGB2BGR)
        img_path = os.path.join(opt.jpegtemp, 'jpeg_' + str(opt.jpeg_quality) + '{:03d}'.format(i) + '.jpg')
        cv2.imwrite(img_path, img, [int(cv2.IMWRITE_JPEG_QUALITY), opt.jpeg_quality])

    image_reload = np.copy(image)
    for i in range(N):
        img_path = os.path.join(opt.jpegtemp, 'jpeg_' + str(opt.jpeg_quality) + '{:03d}'.format(i) + '.jpg')
        img = cv2.imread(img_path)

        image_reload[i] = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
    image_reload = np.transpose(image_reload, (0, 3, 1, 2)).astype(np.float32) / 255

    image_gap = image_reload - image_ori
    image_gap = torch.from_numpy(image_gap).type(torch.float).to(opt.device)

    img_pro = image_batch + image_gap
    return img_pro


def gaussian_blur(opt, image_batch):
    blurrer = transforms.GaussianBlur(kernel_size=opt.gaussian_kernelSize, sigma=opt.gaussian_sigma)

    img_pro = blurrer(image_batch)

    return img_pro


def attack_layer(opt, image_batch):
    if np.random.rand() < 0.25:
        image_batch = lowpass_filter(image_batch).to(opt.device)
        
    if np.random.rand() < 0.25:
        image_batch = jpeg_compression(opt, image_batch)

    if np.random.rand() < 0.25:
        image_batch = gaussian_blur(opt, image_batch)

    image_pro = image_batch 

    return image_pro
