# encoding: utf-8

import os
import time
import numpy as np

from PIL import Image

import torch
import torch.utils.data
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import ReduceLROnPlateau

from options import Options
from utils.dataset import QRCodeDataset, ImageDataset
from criteria.lpips.lpips import LPIPS
from models.HidingNet import UNetDeep, UNetShallow
from models.RevealNet import FullConvSkip, FullConv, TransConv
from functions import training, validation, generation, revealing, detection
from utils.common import print_log, weights_init, save_checkpoint, save_best_result_pic



def IIW_Main(opt):
    opt.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print("[*]Running on device: {}".format(opt.device))

    ##################  Create the dirs to save the result ##################
    cur_time = time.strftime('%Y%m%dH%H%M%S', time.localtime())

    if opt.secret_mode == 'Gray':
        assert opt.Hnet_inchannel == 1, '[*]Secret mode is [Gray], please set the input channel of Hnet as 1!'
    elif opt.secret_mode == 'RGB':
        assert opt.Hnet_inchannel == 3, '[*]Secret mode is [RGB], please set the input channel of Hnet as 3!'
    else:
        assert opt.secret_mode == 'QRCode', '[*]Please set correct secrect mode, must be one of [RGB, Gray, QRCode]!'

    assert opt.Hnet_inchannel == opt.Rnet_outchannel, '[*]Please make sure the channel of input secret image equal to the extracted secret image!'

    if opt.mode == 'train':
        Hnet_comment = 'Hnet{}IC{}OC{}'.format(opt.Hnet_mode, opt.Hnet_inchannel, opt.Hnet_outchannel)
        Rnet_comment = 'Rnet{}IC{}OC{}'.format(opt.Rnet_mode, opt.Rnet_inchannel, opt.Rnet_outchannel)

        opt.experiment_dir = os.path.join(opt.output_dir, \
            cur_time+"_"+str(opt.imageSize)+"_"+opt.norm+"_"+opt.loss+"_"+opt.Rloss_mode+"_"+Hnet_comment+"_"+Rnet_comment+"_"+str(opt.Hnet_factor)+"_"+opt.attack+"_"+opt.mask_mode+"_"+opt.secret_mode)

        print("[*]Saving the experiment results at {}".format(opt.experiment_dir))

        opt.jpegtemp = os.path.join(opt.experiment_dir, 'JpegTemp')
        os.makedirs(opt.jpegtemp, exist_ok=True)

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

        opt.logPath = os.path.join(opt.outlogs, '{}_log.txt'.format(opt.mode))
        print_log(str(opt), opt.logPath)
            
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

        #opt.mask_pd_dir = os.path.join(opt.experiment_dir, "mask_pd")
        opt.mask_pd_dir = opt.experiment_dir
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
        raise ValueError("[*]Invalid norm option. Must be one of [instance, batch, none]") 
######################################################################################################################################################
    if opt.mode == 'train':
        ##### Initialize dataloader #####
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

        if opt.secret_mode == 'QRCode':
            secret_dataset = QRCodeDataset(
                opt = opt,
                transforms = transforms_secret
            )
        else:
            secret_dataset = ImageDataset(
                root = opt.secret_dir,
                transforms = transforms_secret
            )
        train_dataset_cover = ImageDataset(
            root = opt.train_dir,
            transforms = transforms_cover)
        val_dataset_cover = ImageDataset(
            root = opt.val_dir,
            transforms = transforms_cover)

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
        
        ##### Initialize Network architecture #####
        if opt.Hnet_mode == 'UNetDeep':
            Hnet = UNetDeep(input_nc=opt.Hnet_inchannel, output_nc=opt.Hnet_outchannel, norm_layer=norm_layer, output_function=nn.Sigmoid()).to(opt.device)
        elif opt.Hnet_mode == 'UNetShallow':
            Hnet = UNetShallow(input_nc=opt.Hnet_inchannel, output_nc=opt.Hnet_outchannel, norm_layer=norm_layer, output_function=nn.Sigmoid()).to(opt.device)
        else:
            raise ValueError("[*]Invalid Hiding Net Mode. Must be one of [UNetDeep, UNetShallow]")

        if opt.Rnet_mode == 'FullConvSkip':
            Rnet = FullConvSkip(input_nc=opt.Rnet_inchannel, output_nc=opt.Rnet_outchannel, norm_layer=norm_layer, output_function=nn.Sigmoid()).to(opt.device)
        elif opt.Rnet_mode == 'FullConv':
            Rnet = FullConv(input_nc=opt.Rnet_inchannel, output_nc=opt.Rnet_outchannel, norm_layer=norm_layer, output_function=nn.Sigmoid()).to(opt.device)
        elif opt.Rnet_mode == 'TransConv':
            Rnet = TransConv(input_nc=opt.Rnet_inchannel, output_nc=opt.Rnet_outchannel, norm_layer=norm_layer, output_function=nn.Sigmoid()).to(opt.device)
        else:
            raise ValueError("[*]Invalid Reveal Net Mode. Must be one of [FullConvSkip, FullConv, TransConv]")

        # Load Pre-trained mode
        if opt.checkpoint != "":
            print('[*]Load pre-trained model from: {}'.format(opt.checkpoint))
            checkpoint = torch.load(opt.checkpoint)
            Hnet.load_state_dict(checkpoint['H_state_dict'], strict=True)
            Rnet.load_state_dict(checkpoint['R_state_dict'], strict=True)
        else:
            # Using Kaiming Normalization to initialize network's parameters
            print('[*]Training from scratch')
            Hnet.apply(weights_init)
            Rnet.apply(weights_init)

        # Loss and Metric
        criterion_dict = {
            'Hnet_loss' : nn.MSELoss().to(opt.device),
            #'Hnet_loss' : LPIPS(net_type='alex').to(opt.device).eval(),
            'Rnet_imgloss' : nn.MSELoss().to(opt.device),
            'Rnet_watloss' : nn.MSELoss().to(opt.device),
        }

        opt_Hnet = optim.Adam(Hnet.parameters(), lr=opt.lr, betas=(opt.beta_adam, 0.999))
        opt_Rnet = optim.Adam(Rnet.parameters(), lr=opt.lr, betas=(opt.beta_adam, 0.999))
        optimizer_dict = {
            'opt_Hnet' : opt_Hnet,
            'opt_Rnet' : opt_Rnet
        }
        
        Hnet_scheduler = ReduceLROnPlateau(opt_Hnet, mode='min', factor=0.2, patience=8, verbose=True)
        Rnet_scheduler = ReduceLROnPlateau(opt_Rnet, mode='min', factor=0.2, patience=8, verbose=True)
        scheduler_dict = {
            'Hnet_scheduler' : Hnet_scheduler,
            'Rnet_scheduler' : Rnet_scheduler
        }

        smallestLoss = 10000
        print_log("Training is beginning .......................................................", opt.logPath)
        for epoch in range(opt.max_epoch):
            #adjust_learning_rate(opt, optimizer, epoch)

            train_loader = zip(secret_loader, train_loader_cover)
            val_loader = zip(secret_loader, val_loader_cover)

            #################### training ####################
            training(opt, train_loader, epoch, Hnet=Hnet, Rnet=Rnet, criterion_dict=criterion_dict, optimizer_dict=optimizer_dict, writer=writer, scheduler_dict=scheduler_dict)

            #################### validation  ####################
            with torch.no_grad():
                losses_dict = validation(opt, val_loader, epoch, Hnet=Hnet, Rnet=Rnet, criterion_dict=criterion_dict, writer=writer)

            # Save the best model parameters
            sum_diff = losses_dict['Cdiff'] + losses_dict['Sdiff']
            is_best = sum_diff < smallestLoss
            if is_best:
                save_best_result_pic(opt, val_loader, Hnet, Rnet)
            smallestLoss = sum_diff

            stat_dict = {
                'epoch': epoch + 1,
                'H_state_dict': Hnet.state_dict(),
                'R_state_dict': Rnet.state_dict()
            }

            save_checkpoint(opt, stat_dict, is_best)
        
        writer.close()
######################################################################################################################################################
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

        if opt.secret_mode == 'QRCode':
            np.random.seed(5)
            bit_size = 16
            bit_number = opt.imageSize // bit_size
            
            random_bits = np.random.randint(low=0, high=2, size=(bit_number, bit_number))
            random_bits = np.repeat(random_bits, bit_size, 0)
            random_bits = np.repeat(random_bits, bit_size, 1)

            random_bits = np.stack((random_bits, random_bits, random_bits), 0)
            
            random_bits = (random_bits * 255).astype('uint8').transpose([1,2,0])

            secret_image = Image.fromarray(random_bits)
        else:
            print('[*]Load the secret image from: {}'.format(opt.secret_dir))
            if opt.secret_mode == 'Gray':
                secret_image = Image.open(opt.secret_dir).convert('L')
            else:
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
            generation(opt, dataset=cover_dataset, cov_loader=cover_loader, secret_img=secret_image, Hnet=Hnet)        
######################################################################################################################################################
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
            revealing(opt=opt, dataset=container_dataset, con_loader=container_loader, Rnet=Rnet)   
######################################################################################################################################################
    elif opt.mode == 'detect':
        print('[*]Load the secret image from: {}'.format(opt.secret_dir))
        if opt.secret_mode == 'Gray':
            secret_image = Image.open(opt.secret_dir).convert('L')
        else:
            secret_image = Image.open(opt.secret_dir).convert('RGB')
        secret_image = secret_image.resize((opt.imageSize, opt.imageSize))

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
                detection(opt=opt, dataset=image_dataset, data_loader=image_loader, secret_img=secret_image, Rnet=None)

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
            assert opt.checkpoint != None, 'Please assign correct directory of pre-trained mode'
            print('[*]Load the pre-trained model from: {}'.format(opt.checkpoint))
            checkpoint = torch.load(opt.checkpoint)
            Rnet.load_state_dict(checkpoint['R_state_dict'], strict=True)
            
            image_loader = DataLoader(
                        image_dataset, 
                        batch_size=opt.bs_extract,
                        shuffle=False, 
                        num_workers=int(opt.workers)
                    )
            with torch.no_grad():
                detection(opt=opt, dataset=image_dataset, data_loader=image_loader, secret_img=secret_image, Rnet=Rnet)
        else:
            raise ValueError("The target images to detect must be one of revealed secret images or container images")

    else:
        raise ValueError("Please select correct running mode. Must be one of [train | generate | reveal | detect]")



if __name__ == '__main__':
    opts = Options().parse()
    
    IIW_Main(opt=opts)