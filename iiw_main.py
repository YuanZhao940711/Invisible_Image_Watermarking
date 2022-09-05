# encoding: utf-8

import os
import time

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
from utils.dataset import ImageDataset
from models.HidingNet import UNetDeep, UNetShallow
from models.RevealNet import FullConvSkip, FullConv, TransConv
from functions import training, validation, generation, revealing, detection
from utils.common import print_log, weights_init, adjust_learning_rate, save_checkpoint



def IIW_Main(opt):
    ################## Define global parameters ##################
    #global opt, optimizer, writer, logPath, scheduler, val_loader, smallestLoss

    opt.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print("[*]Running on device: {}".format(opt.device))

    ##################  Create the dirs to save the result ##################
    cur_time = time.strftime('%Y%m%dH%H%M%S', time.localtime())
    assert opt.Hnet_inchannel == opt.Rnet_outchannel, '[*]Please make sure the channel of input secret image equal to the extracted secret image!'

    if opt.mode == 'train':
        Hnet_comment = 'Hnet{}IC{}OC{}'.format(opt.Hnet_mode, opt.Hnet_inchannel, opt.Hnet_outchannel)
        Rnet_comment = 'Rnet{}IC{}OC{}'.format(opt.Rnet_mode, opt.Rnet_inchannel, opt.Rnet_outchannel)

        opt.experiment_dir = os.path.join(opt.output_dir, cur_time+"_"+str(opt.imageSize)+"_"+opt.norm+"_"+opt.loss+"_"+opt.Rloss_mode+"_"+Hnet_comment+"_"+Rnet_comment+"_"+str(opt.Hnet_factor)+"_"+opt.mask_mode+"_"+opt.attack)
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
        raise ValueError("[*]Invalid norm option. Must be one of [instance, batch, none]") 

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
        if opt.loss == 'l1':
            criterion = nn.L1Loss().to(opt.device)
        elif opt.loss == 'l2':
            criterion = nn.MSELoss().to(opt.device)
        else:
            raise ValueError("[*]Invalid Loss Function. Must be one of [l1, l2]")
        
        params = list(Hnet.parameters())+list(Rnet.parameters())
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
        print_log("Training is beginning .......................................................", opt.logPath)
        for epoch in range(opt.max_epoch):
            adjust_learning_rate(opt, optimizer, epoch)

            train_loader = zip(secret_loader, train_loader_cover)
            val_loader = zip(secret_loader, val_loader_cover)

            ################## training ##################
            training(opt, train_loader, epoch, Hnet=Hnet, Rnet=Rnet, criterion=criterion, optimizer=optimizer, writer=writer)

            ################## validation  ##################
            with torch.no_grad():
                val_hloss, val_rloss, val_mloss, val_hdiff, val_rdiff = validation(opt, val_loader, epoch, Hnet=Hnet, Rnet=Rnet, criterion=criterion, writer=writer)

            ################## adjust learning rate ##################
            scheduler.step(val_rloss) # 注意！这里只用 R 网络的 loss 进行 learning rate 的更新

            # Save the best model parameters
            sum_diff = val_hdiff + val_rdiff
            """
            is_best = sum_diff < globals()["smallestLoss"]
            globals()["smallestLoss"] = sum_diff
            """
            is_best = sum_diff < smallestLoss
            smallestLoss = sum_diff

            stat_dict = {
                'epoch': epoch + 1,
                'H_state_dict': Hnet.state_dict(),
                'R_state_dict': Rnet.state_dict(),
                'optimizer' : optimizer.state_dict(),
            }

            save_checkpoint(opt, stat_dict, is_best)
        
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
            generation(opt, dataset=cover_dataset, cov_loader=cover_loader, secret_img=secret_image, Hnet=Hnet)        

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

    elif opt.mode == 'detect':
        print('[*]Load the secret image from: {}'.format(opt.secret_dir))
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