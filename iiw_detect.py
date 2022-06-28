# encoding: utf-8

import os
import cv2
import math
import time
import shutil
import argparse
import numpy as np
from PIL import Image, ImageOps
from tqdm import tqdm

import matplotlib.pyplot as plt

import torch
import torch.utils.data
import torch.nn as nn
import torch.optim as optim
import torch.nn.init as init
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import ReduceLROnPlateau

from utils.dataset import ImageDataset
from models.DetectNet import ResNet18
from models.RevealNet import FullConvSkip, FullConv, TransConv
from models.HidingNet import UNetDeep, UNetShallow



parser = argparse.ArgumentParser()
parser.add_argument('--mode', type=str, default='', help='train | generate | reveal | detect')
parser.add_argument('--workers', type=int, default=2, help='number of data loading workers')
parser.add_argument('--imageSize', type=int, default=256, help='the number of frames')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate, default=0.001')
parser.add_argument('--beta_adam', type=float, default=0.5, help='beta_adam for adam. default=0.5')
parser.add_argument('--Hnet', default='', help="path to Hidingnet (to continue training)")
parser.add_argument('--Rnet', default='', help="path to Revealnet (to continue training)")
parser.add_argument('--Hnet_mode', type=str, default='UNetDeep', help='UNetDeep | UNetShallow')
parser.add_argument('--Rnet_mode', type=str, default='FullConvSkip', help='FullConvSkip | FullConv | TransConv')
parser.add_argument('--Rnet_beta', type=float, default=0.75, help='hyper parameter of Hnet factor')
parser.add_argument('--Hnet_factor', type=float, default=1, help='hyper parameter of Hnet factor')
parser.add_argument('--checkpoint', default='', help='checkpoint address')

parser.add_argument('--logFrequency', type=int, default=10, help='the frequency of print the log on the console')
parser.add_argument('--resultPicFrequency', type=int, default=100, help='the frequency of save the resultPic')
parser.add_argument('--norm', default='instance', help='batch or instance')
parser.add_argument('--loss', default='l2', help='l1 or l2')
parser.add_argument('--Rloss_mode', default='secret0', help='secret0 | secret1 | mask')
parser.add_argument('--Hnet_inchannel', type=int, default=3, help='1: gray; 3: color')
parser.add_argument('--Hnet_outchannel', type=int, default=3, help='1: gray; 3: color')
parser.add_argument('--Rnet_inchannel', type=int, default=3, help='1: gray; 3: color')
parser.add_argument('--Rnet_outchannel', type=int, default=3, help='1: gray; 3: color')
parser.add_argument('--max_val_iters', type=int, default=200)
parser.add_argument('--max_train_iters', type=int, default=2000)

parser.add_argument('--bs_train', type=int, default=16, help='training batch size')
parser.add_argument('--bs_generate', type=int, default=16, help='generation batch size')
parser.add_argument('--bs_extract', type=int, default=16, help='extraction batch size')

parser.add_argument('--output_dir', default='', help='directory of outputing results')
parser.add_argument('--val_dir', type=str, default='', help='directory of validation images in training process')
parser.add_argument('--train_dir', type=str, default='', help='directory of training images in training process')
parser.add_argument('--cover_dir', type=str, default='', help='directory of cover images')
parser.add_argument('--container_dir', type=str, default='', help='directory of container images')
parser.add_argument('--secret_dir', type=str, default='', help='directory of secret images')
parser.add_argument('--rev_secret_dir', type=str, default='', help='directory of revealed secret images')

parser.add_argument('--max_epoch', type=int, default=50, help='number of epochs to train for')
parser.add_argument('--dis_num', type=int, default=5, help='number of example image for visualization')
parser.add_argument('--threshold', type=float, default=0.9, help='value to decide whether a pixel is tampered')

parser.add_argument('--gen_mode', type=str, default='white', help='white | random | same')

parser.add_argument('--mask_mode', type=str, default='random', help='random | block | none')
parser.add_argument('--block_size', type=int, default=32, help='bigger block size correspond to smaller masked block')
parser.add_argument('--block_ratio', type=float, default=0.5, help='')



# Custom weights initialization called on netG and netD
def weights_init(m):
    if isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_out')
    elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1.0) 
        m.bias.data.fill_(0)



# Print the structure and parameters number of the net
def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print_log('Total number of parameters: %d' % num_params, logPath)



def main():
    ############### Define global parameters ###############
    global opt, optimizer, writer, logPath, scheduler, val_loader, smallestLoss

    opt = parser.parse_args()
    opt.Hnet_factor = 1
    opt.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print("[*]Running on device: {}".format(opt.device))

    ##################  Create the dirs to save the result ##################
    cur_time = time.strftime('%Y%m%dH%H%M%S', time.localtime())
    assert opt.Hnet_inchannel == opt.Rnet_outchannel, 'Please make sure the channel of input secret image equal to the extracted secret image!'
    if opt.mode == 'train':
        Hnet_comment = 'Hnet{}IC{}OC{}'.format(opt.Hnet_mode, opt.Hnet_inchannel, opt.Hnet_outchannel)
        Rnet_comment = 'Rnet{}IC{}OC{}'.format(opt.Rnet_mode, opt.Rnet_inchannel, opt.Rnet_outchannel)
        opt.experiment_dir = os.path.join(opt.output_dir, cur_time+"_"+str(opt.imageSize)+"_"+opt.norm+"_"+opt.loss+"_"+opt.Rloss_mode+"_"+Hnet_comment+"_"+Rnet_comment+"_"+str(opt.Hnet_factor)+"_"+opt.mask_mode)
        print("[*]Saving the experiment results at {}".format(opt.experiment_dir))

        opt.outckpts = os.path.join(opt.experiment_dir, "CheckPoints")
        os.makedirs(opt.outckpts, exist_ok=True)

        opt.trainpics = os.path.join(opt.experiment_dir, "TrainPics")
        os.makedirs(opt.trainpics, exist_ok=True)

        opt.validationpics = os.path.join(opt.experiment_dir, "ValidationPics")
        os.makedirs(opt.validationpics, exist_ok=True)

        opt.outlogs = os.path.join(opt.experiment_dir, "TrainingLogs")
        os.makedirs(opt.outlogs, exist_ok=True)

        opt.tensorboardlogs = os.path.join(opt.experiment_dir, "TensorBoardLogs")
        os.makedirs(opt.tensorboardlogs, exist_ok=True)
        writer = SummaryWriter(log_dir=opt.tensorboardlogs)

        logPath = os.path.join(opt.outlogs, '{}_log.txt'.format(opt.mode))
        print_log(str(opt), logPath)
            
    elif opt.mode == 'generate':
        opt.experiment_dir = opt.output_dir
        print("[*]Saving the generation results at {}".format(opt.experiment_dir))

        opt.watermark_dir = os.path.join(opt.experiment_dir, "watermark")
        print("[*]Generate the processed secret images at: {}".format(opt.watermark_dir))
        os.makedirs(opt.watermark_dir, exist_ok=True)
            
        opt.loaded_cover_dir = os.path.join(opt.experiment_dir, "cover")
        print("[*]Export the loaded cover images at: {}".format(opt.loaded_cover_dir))
        os.makedirs(opt.loaded_cover_dir, exist_ok=True)
            
        opt.container_dir = os.path.join(opt.experiment_dir, 'container')
        print("[*]Generate the container images at: {}".format(opt.container_dir))
        os.makedirs(opt.container_dir, exist_ok=True)
        
    elif opt.mode == 'reveal':
        opt.experiment_dir = opt.output_dir
        print("[*]Saving the revealed results at {}".format(opt.experiment_dir))

        opt.rev_secret_dir = os.path.join(opt.experiment_dir, "rev_secret")
        print("[*]Generate the retrieved secret images at: {}".format(opt.rev_secret_dir))
        os.makedirs(opt.rev_secret_dir, exist_ok=True)

    elif opt.mode == 'detect':
        opt.experiment_dir = opt.output_dir
        print("[*]Saving the detection results at {}".format(opt.experiment_dir))
        
        opt.gap_dir = os.path.join(opt.experiment_dir, "gap")
        print("[*]Export the gap between the revealed secret and original secret at: {}".format(opt.gap_dir))
        os.makedirs(opt.gap_dir, exist_ok=True)

        opt.mask_pd_dir = os.path.join(opt.experiment_dir, "mask_pd")
        print("[*]Export the predict masks at: {}".format(opt.mask_pd_dir))
        os.makedirs(opt.mask_pd_dir, exist_ok=True)

    ################## Datasets and Networks ##################
    if opt.norm == 'instance':
        norm_layer = nn.InstanceNorm2d
    elif opt.norm == 'batch':
        norm_layer = nn.BatchNorm2d
    elif opt.norm == 'none':
        norm_layer = None
    else:
        raise ValueError("Invalid norm option. Must be one of [instance, batch, none]") 

    if opt.mode == 'train':
        if opt.Hnet_inchannel == 1:  
            transforms_secret = transforms.Compose([
                transforms.Grayscale(num_output_channels=1),
                transforms.Resize([opt.imageSize, opt.imageSize]), 
                transforms.ToTensor()
            ])
        else:
            transforms_secret = transforms.Compose([
                transforms.Resize([opt.imageSize, opt.imageSize]), 
                transforms.ToTensor()
            ])
            
        if opt.Rnet_inchannel == 1:  
            transforms_cover = transforms.Compose([
                transforms.Grayscale(num_output_channels=1),
                transforms.Resize([opt.imageSize, opt.imageSize]), 
                transforms.ToTensor()
            ])
        else:
            transforms_cover = transforms.Compose([
                transforms.Resize([opt.imageSize, opt.imageSize]), 
                transforms.ToTensor()
            ])    

        secret_dataset = ImageDataset(
            root = opt.secret_dir,
            transforms = transforms_secret)
        train_dataset_cover = ImageDataset(
            root = opt.train_dir,
            transforms = transforms_cover)
        val_dataset_cover = ImageDataset(
            root = opt.val_dir,
            transforms = transforms_cover)

        Dnet = ResNet18(in_channels=3, out_channels=1, output_function=nn.Sigmoid()).to(opt.device)

        if opt.Hnet_mode == 'UNetDeep':
            Hnet = UNetDeep(input_nc=opt.Hnet_inchannel, output_nc=opt.Hnet_outchannel, norm_layer=norm_layer, output_function=nn.Sigmoid()).to(opt.device)
        elif opt.Hnet_mode == 'UNetShallow':
            Hnet = UNetShallow(input_nc=opt.Hnet_inchannel, output_nc=opt.Hnet_outchannel, norm_layer=norm_layer, output_function=nn.Sigmoid()).to(opt.device)
        else:
            raise ValueError("Invalid Hiding Net Mode. Must be one of [UNetDeep, UNetShallow]")

        if opt.Rnet_mode == 'FullConvSkip':
            Rnet = FullConvSkip(input_nc=opt.Rnet_inchannel, output_nc=opt.Rnet_outchannel, norm_layer=norm_layer, output_function=nn.Sigmoid()).to(opt.device)
        elif opt.Rnet_mode == 'FullConv':
            Rnet = FullConv(input_nc=opt.Rnet_inchannel, output_nc=opt.Rnet_outchannel, norm_layer=norm_layer, output_function=nn.Sigmoid()).to(opt.device)
        elif opt.Rnet_mode == 'TransConv':
            Rnet = TransConv(input_nc=opt.Rnet_inchannel, output_nc=opt.Rnet_outchannel, norm_layer=norm_layer, output_function=nn.Sigmoid()).to(opt.device)
        else:
            raise ValueError("Invalid Reveal Net Mode. Must be one of [FullConvSkip, FullConv, TransConv]")

        # Load Pre-trained mode
        if opt.checkpoint != "":
            print('[*]Load pre-trained model from: {}'.format(opt.checkpoint))
            checkpoint = torch.load(opt.checkpoint)
            Dnet.load_state_dict(checkpoint['D_state_dict'], strict=True)
            Hnet.load_state_dict(checkpoint['H_state_dict'], strict=True)
            Rnet.load_state_dict(checkpoint['R_state_dict'], strict=True)
        else:
            # Using Kaiming Normalization to initialize network's parameters
            print('[*]Training from scratch')
            Dnet.apply(weights_init)
            Hnet.apply(weights_init)
            Rnet.apply(weights_init)

        # Loss and Metric
        if opt.loss == 'l1':
            criterion = nn.L1Loss().to(opt.device)
        elif opt.loss == 'l2':
            criterion = nn.MSELoss().to(opt.device)
        else:
            raise ValueError("Invalid Loss Function. Must be one of [l1, l2]")
        
        d_criterion = nn.BCELoss().to(opt.device)
        
        params = list(Dnet.parameters())+list(Hnet.parameters())+list(Rnet.parameters())
        optimizer = optim.Adam(params, lr=opt.lr, betas=(opt.beta_adam, 0.999))
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=8, verbose=True)

        train_loader_cover = DataLoader(
            train_dataset_cover,
            batch_size=opt.bs_train,
            shuffle=True,
            num_workers=int(opt.workers)
        )
        secret_loader = DataLoader(
            secret_dataset,
            batch_size=opt.bs_train,
            shuffle=True,
            num_workers=int(opt.workers)
        )
        val_loader_cover = DataLoader(
            val_dataset_cover,
            batch_size=opt.bs_train,
            shuffle=False,
            num_workers=int(opt.workers)
        )

        smallestLoss = 10000
        print_log("Training is beginning .......................................................", logPath)
        for epoch in range(opt.max_epoch):
            adjust_learning_rate(optimizer, epoch)

            train_loader = zip(secret_loader, train_loader_cover)
            val_loader = zip(secret_loader, val_loader_cover)

            ################## train ##################
            train(train_loader, epoch, Dnet=Dnet, Hnet=Hnet, Rnet=Rnet, criterion=criterion, d_criterion=d_criterion)

            ################## validation  ##################
            with torch.no_grad():
                val_dloss, val_hloss, val_rloss, val_ddiff, val_hdiff, val_rdiff = validation(val_loader, epoch, Dnet=Dnet, Hnet=Hnet, Rnet=Rnet, criterion=criterion, d_criterion=d_criterion)

            ################## adjust learning rate ##################
            scheduler.step(val_rloss) # 注意！这里只用 R 网络的 loss 进行 learning rate 的更新

            # Save the best model parameters
            sum_diff = val_ddiff + val_hdiff + val_rdiff
            is_best = sum_diff < globals()["smallestLoss"]
            globals()["smallestLoss"] = sum_diff

            stat_dict = {
                'epoch': epoch + 1,
                'D_state_dict': Dnet.state_dict(),
                'H_state_dict': Hnet.state_dict(),
                'R_state_dict': Rnet.state_dict(),
                'optimizer' : optimizer.state_dict(),
            }

            save_checkpoint(stat_dict, is_best)
        
        writer.close()

    elif opt.mode == 'generate':
        if opt.Hnet_inchannel == 1:
            transforms_secret = transforms.Compose([
                transforms.Grayscale(num_output_channels=1),
                transforms.Resize([opt.imageSize, opt.imageSize]), 
                transforms.ToTensor()
            ])
        else:
            transforms_secret = transforms.Compose([
                transforms.Resize([opt.imageSize, opt.imageSize]), 
                transforms.ToTensor()
            ])

        if opt.Rnet_inchannel == 1:  
            transforms_cover = transforms.Compose([
                transforms.Grayscale(num_output_channels=1),
                transforms.Resize([opt.imageSize, opt.imageSize]), 
                transforms.ToTensor()
            ])
        else:
            transforms_cover = transforms.Compose([
                transforms.Resize([opt.imageSize, opt.imageSize]), 
                transforms.ToTensor()
            ])    

        cover_dataset = ImageDataset(root=opt.cover_dir, transforms=transforms_cover)

        print('[*]Load the secret image from: {}'.format(opt.secret_dir))
        secret_image = Image.open(opt.secret_dir).convert('RGB')
        secret_image = secret_image.resize((opt.imageSize, opt.imageSize))
        secret_image.save(os.path.join(opt.experiment_dir, 'secret_image.png'))
        secret_image = transforms_secret(secret_image).to(opt.device)

        if opt.Hnet_mode == 'UNetDeep':
            Hnet = UNetDeep(input_nc=opt.Hnet_inchannel, output_nc=opt.Hnet_outchannel, norm_layer=norm_layer, output_function=nn.Sigmoid()).to(opt.device)
        elif opt.Hnet_mode == 'UNetShallow':
            Hnet = UNetShallow(input_nc=opt.Hnet_inchannel, output_nc=opt.Hnet_outchannel, norm_layer=norm_layer, output_function=nn.Sigmoid()).to(opt.device)
        else:
            raise ValueError("Invalid Hiding Net Mode. Must be one of [UNetDeep, UNetShallow]")

        # Load Pre-trained mode
        assert opt.checkpoint != None, 'Please assign correct directory of pre-trained mode'
        print('[*]Load the pre-trained model from: {}'.format(opt.checkpoint))
        checkpoint = torch.load(opt.checkpoint, map_location=opt.device)
        Hnet.load_state_dict(checkpoint['H_state_dict'], strict=True)

        cover_loader = DataLoader(
            cover_dataset, 
            batch_size=opt.bs_generate,
            shuffle=False, 
            num_workers=int(opt.workers)
        )
        with torch.no_grad():
            generate(dataset=cover_dataset, cov_loader=cover_loader, secret_img=secret_image, Hnet=Hnet)        

    elif opt.mode == 'reveal':
        if opt.Rnet_inchannel == 1:
            transforms_container = transforms.Compose([
                transforms.Grayscale(num_output_channels=1),
                transforms.Resize([opt.imageSize, opt.imageSize]), 
                transforms.ToTensor()
            ])
        else:
            transforms_container = transforms.Compose([
                transforms.Resize([opt.imageSize, opt.imageSize]), 
                transforms.ToTensor()
            ])

        container_dataset = ImageDataset(root=opt.container_dir, transforms=transforms_container)

        if opt.Rnet_mode == 'FullConvSkip':
            Rnet = FullConvSkip(input_nc=opt.Rnet_inchannel, output_nc=opt.Rnet_outchannel, norm_layer=norm_layer, output_function=nn.Sigmoid()).to(opt.device)
        elif opt.Rnet_mode == 'FullConv':
            Rnet = FullConv(input_nc=opt.Rnet_inchannel, output_nc=opt.Rnet_outchannel, norm_layer=norm_layer, output_function=nn.Sigmoid()).to(opt.device)
        elif opt.Rnet_mode == 'TransConv':
            Rnet = TransConv(input_nc=opt.Rnet_inchannel, output_nc=opt.Rnet_outchannel, norm_layer=norm_layer, output_function=nn.Sigmoid()).to(opt.device)
        else:
            raise ValueError("Invalid Reveal Net Mode. Must be one of [FullConvSkip, FullConv, TransConv]")

        # Load Pre-trained mode
        assert opt.checkpoint != None, 'Please assign correct directory of pre-trained mode'
        print('[*]Load the pre-trained model from: {}'.format(opt.checkpoint))
        checkpoint = torch.load(opt.checkpoint)
        Rnet.load_state_dict(checkpoint['R_state_dict'], strict=True)

        container_loader = DataLoader(
            container_dataset, 
            batch_size=opt.bs_extract,
            shuffle=False, 
            num_workers=int(opt.workers)
        )
        with torch.no_grad():
            reveal(dataset=container_dataset, con_loader=container_loader, Rnet=Rnet)   

    elif opt.mode == 'detect':
        print('[*]Load the secret image from: {}'.format(opt.secret_dir))
        secret_image = Image.open(opt.secret_dir).convert('RGB')
        #secret_image = secret_image.resize((opt.imageSize, opt.imageSize))
        #secret_image = cv2.cvtColor(cv2.imread(opt.secret_dir), cv2.COLOR_BGR2RGB)

        if opt.Rnet_inchannel == 1:
            image_transforms = transforms.Compose([
                transforms.Grayscale(num_output_channels=1),
                transforms.Resize([opt.imageSize, opt.imageSize]), 
                transforms.ToTensor()
            ])
        else:
            image_transforms = transforms.Compose([
                transforms.Resize([opt.imageSize, opt.imageSize]), 
                transforms.ToTensor()
            ])

        secret_image = image_transforms(secret_image).to(opt.device)
        #secret_image = torch.unsqueeze(secret_image, dim=0)

        Dnet = ResNet18(in_channels=3, out_channels=1, output_function=nn.Sigmoid()).to(opt.device)

        assert opt.checkpoint != None, 'Please assign correct directory of pre-trained mode'
        print('[*]Load the pre-trained model from: {}'.format(opt.checkpoint))
        checkpoint = torch.load(opt.checkpoint)
        Dnet.load_state_dict(checkpoint['D_state_dict'], strict=True)

        if opt.container_dir == '' and opt.rev_secret_dir != '':
            print('[*]The directory asssigned is pointing to revealed secret images, so executing detection without Rnet!')
            image_dataset = ImageDataset(root=opt.rev_secret_dir, transforms=image_transforms)

            image_loader = DataLoader(
                        image_dataset, 
                        batch_size=opt.bs_extract,
                        shuffle=False, 
                        num_workers=int(opt.workers)
                    )
            with torch.no_grad():
                detect(dataset=image_dataset, data_loader=image_loader, secret_img=secret_image, Dnet=Dnet, Rnet=None)

        elif opt.container_dir != '' and opt.rev_secret_dir == '':
            print('[*]The directory asssigned is pointing to container images, so executing detection with Rnet!')
            image_dataset = ImageDataset(root=opt.container_dir, transforms=image_transforms)

            if opt.Rnet_mode == 'FullConvSkip':
                Rnet = FullConvSkip(input_nc=opt.Rnet_inchannel, output_nc=opt.Rnet_outchannel, norm_layer=norm_layer, output_function=nn.Sigmoid()).to(opt.device)
            elif opt.Rnet_mode == 'FullConv':
                Rnet = FullConv(input_nc=opt.Rnet_inchannel, output_nc=opt.Rnet_outchannel, norm_layer=norm_layer, output_function=nn.Sigmoid()).to(opt.device)
            elif opt.Rnet_mode == 'TransConv':
                Rnet = TransConv(input_nc=opt.Rnet_inchannel, output_nc=opt.Rnet_outchannel, norm_layer=norm_layer, output_function=nn.Sigmoid()).to(opt.device)
            else:
                raise ValueError("Invalid Reveal Net Mode. Must be one of [FullConvSkip, FullConv, TransConv]")

            # Load Pre-trained mode
            Rnet.load_state_dict(checkpoint['R_state_dict'], strict=True)
            
            image_loader = DataLoader(
                        image_dataset, 
                        batch_size=opt.bs_extract,
                        shuffle=False, 
                        num_workers=int(opt.workers)
                    )
            with torch.no_grad():
                detect(dataset=image_dataset, data_loader=image_loader, secret_img=secret_image, Dnet=Dnet, Rnet=Rnet)
        else:
            raise ValueError("The target images to detect must be one of revealed secret images or container images")

    else:
        raise ValueError("Please select correct running mode. Must be one of [train | generate | reveal | detect]")



def save_checkpoint(state, is_best):
    filename = 'checkpoint.pth.tar'

    checkpoint_path = os.path.join(opt.outckpts, filename)
    torch.save(state, checkpoint_path)

    if is_best:
        shutil.copyfile(checkpoint_path, os.path.join(opt.outckpts, 'best_checkpoint.pth.tar'))



def draw_polygon(sides, radius=1, rotation=0, location=None):
    one_segment = math.pi * 2 / sides

    if type(radius) == list:
        points = [(math.sin(one_segment * i + rotation) * radius[i], math.cos(one_segment * i + rotation) * radius[i]) for i in range(sides)]
    else:
        points = [(math.sin(one_segment * i + rotation) * radius, math.cos(one_segment * i + rotation) * radius) for i in range(sides)]

    if location is not None:
        points = np.array([[sum(pair) for pair in zip(point, location)] for point in points], dtype=np.int32)

    return points    



def random_mask(image):
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



def block_mask(image):
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



def forward_pass(secret_img, cover_img, Dnet, Hnet, Rnet, criterion, d_criterion):
    secret = secret_img.to(opt.device)
    cover = cover_img.to(opt.device)

    watermark = Hnet(secret) * opt.Hnet_factor
    
    container = watermark + cover

    errH = criterion(container, cover)  # Hiding net

    if opt.mask_mode == 'random':
        container_tampered, mask = random_mask(container)
    elif opt.mask_mode == 'block':
        container_tampered, mask = block_mask(container)
    elif opt.mask_mode == 'none':
        container_tampered = container
        mask = torch.zeros_like(container)
    else:
        raise ValueError("Invalid Mask Mode. Must be one of [random, block, none]")
    
    if opt.Rloss_mode == 'secret0':
        tampered = secret * (torch.ones_like(mask) - mask) 
    elif opt.Rloss_mode == 'secret1':
        tampered = secret * (torch.ones_like(mask) - mask) + mask
    elif opt.Rloss_mode == 'mask':
        tampered = mask
    
    retrieved = Rnet(container_tampered) 
    errR = criterion(retrieved, tampered)  # Reveal net

    #gap = torch.abs(retrieved - secret)
    gap = retrieved - secret

    """relu = nn.ReLU()
    gap = relu(gap*10 + 0.5)"""

    rev_mask = Dnet(gap)
    rev_mask_tri = rev_mask.repeat(1,3,1,1)
    errD = d_criterion(rev_mask_tri, mask)

    diffD = (rev_mask_tri-mask).abs().mean()*255
    diffH = (container-cover).abs().mean()*255
    diffR = (retrieved-tampered).abs().mean()*255

    image_dict = {
        'secret': secret,
        'cover': cover,
        'watermark': watermark,
        'container': container,
        'container_tampered': container_tampered,
        'secret_tampered': tampered,
        'secret_retrieved': retrieved,
        'mask_gt': mask,
        'mask_pd': rev_mask_tri
    }
    data_dict = {
        'errD': errD,
        'errH': errH,
        'errR': errR, 
        'diffD': diffD, 
        'diffH': diffH, 
        'diffR': diffR
    }
    return image_dict, data_dict


def train(train_loader, epoch, Dnet, Hnet, Rnet, criterion, d_criterion):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    Dlosses = AverageMeter()  
    Hlosses = AverageMeter()  
    Rlosses = AverageMeter() 
    SumLosses = AverageMeter()
    Ddiff = AverageMeter()
    Hdiff = AverageMeter()
    Rdiff = AverageMeter()

    # Switch to train mode
    Dnet.train()
    Hnet.train()
    Rnet.train()

    start_time = time.time()

    for i, (secret_img, cover_img) in enumerate(train_loader):

        data_time.update(time.time() - start_time)

        image_dict, data_dict = forward_pass(secret_img, cover_img, Dnet, Hnet, Rnet, criterion, d_criterion)

        Dlosses.update(data_dict['errD'].data, opt.bs_train)  # D loss
        Hlosses.update(data_dict['errH'].data, opt.bs_train)  # H loss
        Rlosses.update(data_dict['errR'].data, opt.bs_train)  # R loss
        SumLosses.update(data_dict['errD'].data + data_dict['errH'].data + data_dict['errR'].data, opt.bs_train) # H loss + R loss
        Ddiff.update(data_dict['diffD'].data, opt.bs_train)
        Hdiff.update(data_dict['diffH'].data, opt.bs_train)
        Rdiff.update(data_dict['diffR'].data, opt.bs_train)

        # Loss, backprop, and optimization step
        err_sum = data_dict['errD']*10.0 +  data_dict['errH'] + data_dict['errR']*opt.Rnet_beta
        optimizer.zero_grad()
        err_sum.backward()
        optimizer.step()

        # Time spent on one batch
        batch_time.update(time.time() - start_time)
        start_time = time.time()

        log = '[{:d}/{:d}][{:d}/{:d}] Loss_D: {:.6f} Loss_H: {:.6f} Loss_R: {:.6f} Loss_Sum: {:.6f} L1_D: {:.4f} L1_H: {:.4f} L1_R: {:.4f} datatime: {:.4f} batchtime: {:.4f}'.format(
            epoch, opt.max_epoch, i, opt.max_train_iters,
            Dlosses.val, Hlosses.val, Rlosses.val, SumLosses.val, Ddiff.val, Hdiff.val, Rdiff.val, 
            data_time.val, batch_time.val
        )

        if i % opt.logFrequency == 0:
            print(log)

        if epoch <= 0 and i % opt.resultPicFrequency == 0:
            save_result_pic(opt.dis_num, image_dict, epoch, i, opt.trainpics)

        if i == opt.max_train_iters-1:
            break

    # To save the last batch only
    save_result_pic(opt.dis_num, image_dict, epoch, i, opt.trainpics)

    epoch_log = "Training[{:d}] Dloss={:.6f} Hloss={:.6f} Rloss={:.6f} SumLoss={:.6f} Ddiff={:.4f} Hdiff={:.4f} Rdiff={:.4f} lr={:.6f} Epoch time={:.4f}".format(
        epoch, Dlosses.avg, Hlosses.avg, Rlosses.avg, SumLosses.avg, Ddiff.avg, Hdiff.avg, Rdiff.avg, optimizer.param_groups[0]['lr'], batch_time.sum
    )
    print_log(epoch_log, logPath)

    writer.add_scalar("lr/lr", optimizer.param_groups[0]['lr'], epoch)
    writer.add_scalar('train/D_loss', Dlosses.avg, epoch)
    writer.add_scalar('train/H_loss', Hlosses.avg, epoch)
    writer.add_scalar('train/R_loss', Rlosses.avg, epoch)
    writer.add_scalar('train/Sum_loss', SumLosses.avg, epoch)
    writer.add_scalar('train/D_diff', Ddiff.avg, epoch)
    writer.add_scalar('train/H_diff', Hdiff.avg, epoch)
    writer.add_scalar('train/R_diff', Rdiff.avg, epoch)



def validation(val_loader, epoch, Dnet, Hnet, Rnet, criterion, d_criterion):
    print("#################################################### validation begin ####################################################")    
    batch_time = AverageMeter()
    Dlosses = AverageMeter()
    Hlosses = AverageMeter()
    Rlosses = AverageMeter()  
    Ddiff = AverageMeter()
    Hdiff = AverageMeter()
    Rdiff = AverageMeter()

    start_time = time.time()

    Dnet.eval()
    Hnet.eval()
    Rnet.eval()

    for i, (secret_img, cover_img) in enumerate(val_loader):

        image_dict, data_dict = forward_pass(secret_img, cover_img, Dnet, Hnet, Rnet, criterion, d_criterion)

        Dlosses.update(data_dict['errD'].data, opt.bs_train)  # D loss
        Hlosses.update(data_dict['errH'].data, opt.bs_train)  # H loss
        Rlosses.update(data_dict['errR'].data, opt.bs_train)  # R loss
        Ddiff.update(data_dict['diffD'].data, opt.bs_train)
        Hdiff.update(data_dict['diffH'].data, opt.bs_train)
        Rdiff.update(data_dict['diffR'].data, opt.bs_train)
        
        if i == opt.max_val_iters-1:
            break

        batch_time.update(time.time() - start_time)
        start_time = time.time()

        val_log = "Validation[{:d}][{:d}/{:d}] val_Dloss = {:.6f} val_Hloss = {:.6f} val_Rloss = {:.6f} val_Ddiff = {:.4f} val_Hdiff = {:.4f} val_Rdiff={:.4f} batch time={:.4f}".format(
            epoch, i, opt.max_val_iters,
            Dlosses.val, Hlosses.val, Rlosses.val, Ddiff.val, Hdiff.val, Rdiff.val, 
            batch_time.val
        )
        if i % opt.logFrequency == 0:
            print(val_log)
    
    save_result_pic(opt.dis_num, image_dict, epoch, i, opt.validationpics)
    
    val_log = "Validation[{:d}] val_Dloss = {:.6f} val_Hloss = {:.6f} val_Rloss = {:.6f} val_Ddiff = {:.4f} val_Hdiff = {:.4f} val_Rdiff={:.4f} batch time={:.4f}".format(
        epoch, Dlosses.avg, Hlosses.avg, Rlosses.avg, Ddiff.avg, Hdiff.avg, Rdiff.avg, batch_time.sum)
    print_log(val_log, logPath)

    writer.add_scalar('validation/D_loss_avg', Dlosses.avg, epoch)
    writer.add_scalar('validation/H_loss_avg', Hlosses.avg, epoch)
    writer.add_scalar('validation/R_loss_avg', Rlosses.avg, epoch)
    writer.add_scalar('validation/D_diff_avg', Ddiff.avg, epoch)
    writer.add_scalar('validation/H_diff_avg', Hdiff.avg, epoch)
    writer.add_scalar('validation/R_diff_avg', Rdiff.avg, epoch)

    print("#################################################### validation end ####################################################")
    return Dlosses.avg, Hlosses.avg, Rlosses.avg, Ddiff.avg, Hdiff.avg, Rdiff.avg



def generate(dataset, cov_loader, secret_img, Hnet):
    Hnet.eval()

    idx = 0
    for cover_batch in tqdm(cov_loader):
        cover_batch = cover_batch.to(opt.device)
        batch_size_cover, _, _, _ = cover_batch.size()

        H_input = secret_img.repeat(batch_size_cover, 1, 1, 1)

        watermark_batch = Hnet(H_input) * opt.Hnet_factor

        container_batch = watermark_batch + cover_batch

        for i, container in enumerate(container_batch):
            watermark_img = tensor2img(watermark_batch[i]*10 +0.5)
            cover_img = tensor2img(cover_batch[i])
            container_img = tensor2img(container)

            img_name = os.path.basename(dataset.image_paths[idx]).split('.')[0]

            watermark_img.save(os.path.join(opt.watermark_dir, '{}.png'.format(img_name)))
            cover_img.save(os.path.join(opt.loaded_cover_dir, '{}.png'.format(img_name)))
            container_img.save(os.path.join(opt.container_dir, '{}.png'.format(img_name)))

            idx += 1



def reveal(dataset, con_loader, Rnet):
    Rnet.eval()

    idx = 0

    for container_batch in tqdm(con_loader):
        container_batch = container_batch.to(opt.device)

        rev_secret_bath = Rnet(container_batch)

        for _, rev_secret in enumerate(rev_secret_bath):
            img_name = os.path.basename(dataset.image_paths[idx])

            rev_secret_img = tensor2img(rev_secret)
            rev_secret_img.save(os.path.join(opt.rev_secret_dir, img_name))

            idx += 1



def detect(dataset, data_loader, secret_img, Dnet, Rnet):
    #secret_img = np.array(secret_img, dtype='int32')
    #secret_image = torch.unsqueeze(secret_image, dim=0)
    Dnet.eval()

    if Rnet != None:
        Rnet.eval()
        idx = 0
        for container_batch in tqdm(data_loader):
            container_batch = container_batch.to(opt.device)

            rev_secret_bath = Rnet(container_batch)

            for _, rev_secret in enumerate(rev_secret_bath):
                img_name = os.path.basename(dataset.image_paths[idx])

                #gap = torch.abs(rev_secret - secret_img)
                gap = rev_secret - secret_img
                #gap = gap*10 + 0.5#.clamp_(0.0, 1.0)
                gap = torch.unsqueeze(gap, dim=0)
                mask = Dnet(gap)

                #gap = (gap*10 + 0.5).clamp_(0.0, 1.0)
                gap_img = tensor2img(gap[0])
                gap_img.save(os.path.join(opt.gap_dir, img_name))

                mask_img = tensor2img(mask[0])
                mask_img.save(os.path.join(opt.mask_pd_dir, img_name))

                idx += 1
    else:
        idx = 0
        for rev_secret_batch in tqdm(data_loader):
            rev_secret_batch = rev_secret_batch.to(opt.device)

            for _, rev_secret in enumerate(rev_secret_batch):
                img_name = os.path.basename(dataset.image_paths[idx])

                #gap = torch.abs(rev_secret - secret_img)
                gap = rev_secret - secret_img
                #gap = (gap*10 + 0.5).clamp_(0.0, 1.0)
                #gap = gap*10 + 0.5
                gap = torch.unsqueeze(gap, dim=0)
                mask = Dnet(gap)

                gap_img = tensor2img(gap[0])
                gap_img.save(os.path.join(opt.gap_dir, img_name))

                mask_img = tensor2img(mask[0])
                mask_img.save(os.path.join(opt.mask_pd_dir, img_name))

                idx += 1        



def tensor2array(var):
    var = var.cpu().detach().numpy().transpose([1,2,0])
    var[var < 0] = 0
    var[var > 1] = 1
    var = var * 255
    if var.shape[2] == 1:
        var = np.squeeze(var, axis=2)
    return var.astype('int32')



def tensor2img(var):
    var = var.cpu().detach().numpy().transpose([1,2,0])
    var[var < 0] = 0
    var[var > 1] = 1
    var = var * 255
    if var.shape[2] == 1:
        var = np.squeeze(var, axis=2)
    return Image.fromarray(var.astype('uint8'))



def print_log(log_info, log_path, console=True):
    # print the info into the console
    if console:
        print(log_info)
    # write the log into log file
    if not os.path.exists(log_path):
        fp = open(log_path, "w")
        fp.writelines(log_info + "\n")
    else:
        with open(log_path, 'a+') as f:
            f.writelines(log_info + '\n')



def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = opt.lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr



def save_result_pic(dis_num, image_dict, epoch, i, save_path):
    resultImgName = os.path.join(save_path, 'ResultPics_epoch{:03d}_batch{:04d}.png'.format(epoch, i))

    cover_gap = image_dict['container'] - image_dict['cover']
    cover_gap = (cover_gap*10 + 0.5).clamp_(0.0, 1.0)
    
    secret_gap = image_dict['secret_retrieved'] - image_dict['secret_tampered']
    secret_gap = (secret_gap*10 + 0.5).clamp_(0.0, 1.0)

    mask_gap = image_dict['mask_pd'] - image_dict['mask_gt']
    mask_gap = (mask_gap*10 + 0.5).clamp_(0.0, 1.0)

    fig = plt.figure(figsize=(48, 4*dis_num))
    gs = fig.add_gridspec(nrows=dis_num, ncols=12)
    for img_idx in range(dis_num):
        fig.add_subplot(gs[img_idx, 0])
        sec_img = tensor2img(image_dict['secret'][img_idx])
        plt.imshow(sec_img)
        plt.title("Secret")

        fig.add_subplot(gs[img_idx, 1])
        cov_img = tensor2img(image_dict['cover'][img_idx])
        plt.imshow(cov_img)
        plt.title("Cover")

        fig.add_subplot(gs[img_idx, 2])
        wat_img = tensor2img(image_dict['watermark'][img_idx]*10 +0.5)
        plt.imshow(wat_img)
        plt.title("Watermark")

        fig.add_subplot(gs[img_idx, 3])
        con_img = tensor2img(image_dict['container'][img_idx])
        plt.imshow(con_img)
        plt.title("Container")

        fig.add_subplot(gs[img_idx, 4])
        covgap_img = tensor2img(cover_gap[img_idx])
        plt.imshow(covgap_img)
        plt.title("Cover Gap")

        fig.add_subplot(gs[img_idx, 5])
        tamcon_img = tensor2img(image_dict['container_tampered'][img_idx])
        plt.imshow(tamcon_img)
        plt.title("Tampered Container")

        fig.add_subplot(gs[img_idx, 6])
        tamsec_img = tensor2img(image_dict['secret_tampered'][img_idx])
        plt.imshow(tamsec_img)
        plt.title("Tampered Secret")

        fig.add_subplot(gs[img_idx, 7])
        retsec_img = tensor2img(image_dict['secret_retrieved'][img_idx])
        plt.imshow(retsec_img)
        plt.title("Retrieved Secret")

        fig.add_subplot(gs[img_idx, 8])
        secgap_img = tensor2img(secret_gap[img_idx])
        plt.imshow(secgap_img)
        plt.title("Secret Gap")

        fig.add_subplot(gs[img_idx, 9])
        maskgt_img = tensor2img(image_dict['mask_gt'][img_idx])
        plt.imshow(maskgt_img)
        plt.title("Mask GT")

        fig.add_subplot(gs[img_idx, 10])
        maskpd_img = tensor2img(image_dict['mask_pd'][img_idx])
        plt.imshow(maskpd_img)
        plt.title("Mask PD")

        fig.add_subplot(gs[img_idx, 11])
        maskgap_img = tensor2img(mask_gap[img_idx])
        plt.imshow(maskgap_img)
        plt.title("Mask Gap")
    
    plt.tight_layout()
    fig.savefig(resultImgName)
    plt.close(fig)



class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count



if __name__ == '__main__':
    main()