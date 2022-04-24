# encoding: utf-8

import os
import cv2
import math
import time
import shutil
import argparse
from cv2 import threshold
from matplotlib import container
import numpy as np
from PIL import Image
from tqdm import tqdm

import matplotlib.pyplot as plt

import torch
import torch.utils.data
import torch.nn as nn
import torch.optim as optim
import torch.nn.init as init
import torch.backends.cudnn as cudnn
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import ReduceLROnPlateau

from utils.dataset import ImageDataset
from models.AdversarialNet import Adversary
from models.HidingUNet import UnetGenerator
from models.RevealNet import RevealNet




parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default="train", help='train | val | test')
parser.add_argument('--workers', type=int, default=2, help='number of data loading workers')
parser.add_argument('--imageSize', type=int, default=256, help='the number of frames')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate, default=0.001')
parser.add_argument('--decay_round', type=int, default=10, help='learning rate decay 0.5 each decay_round')
parser.add_argument('--beta_adam', type=float, default=0.5, help='beta_adam for adam. default=0.5')
parser.add_argument('--Anet', default='', help="path to Hidingnet (to continue training)")
parser.add_argument('--Hnet', default='', help="path to Hidingnet (to continue training)")
parser.add_argument('--Rnet', default='', help="path to Revealnet (to continue training)")
parser.add_argument('--Hnet_factor', type=float, default=1, help='hyper parameter of Hnet factor')
parser.add_argument('--beta', type=float, default=0.75, help='hyper parameter of beta')
parser.add_argument('--checkpoint', default='', help='checkpoint address')

parser.add_argument('--logFrequency', type=int, default=10, help='the frequency of print the log on the console')
parser.add_argument('--resultPicFrequency', type=int, default=100, help='the frequency of save the resultPic')
parser.add_argument('--norm', default='instance', help='batch or instance')
parser.add_argument('--loss', default='l2', help='l1 or l2')
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
parser.add_argument('--val_dir', type=str, default='', help='directory of validation images')
parser.add_argument('--train_dir', type=str, default='', help='directory of training images')
parser.add_argument('--secret_dir', type=str, default='', help='directory of secret images for training')

parser.add_argument('--max_epoch', type=int, default=50, help='number of epochs to train for')
parser.add_argument('--dis_num', type=int, default=5, help='number of example image for visualization')
parser.add_argument('--threshold', type=float, default=0.9, help='value to decide whether a pixel is tampered')

parser.add_argument('--mode', type=str, default='', help='train | generate | extract')
parser.add_argument('--origin_dir', type=str, default='', help='directory of original images')
parser.add_argument('--secret_path', type=str, default='', help='path of origin secret image')
parser.add_argument('--container_dir', type=str, default='', help='directory of container images')

parser.add_argument('--compression', type=str, default='No')
parser.add_argument('--com_quality', type=int, default=70)



# Custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_out')
    elif classname.find('BatchNorm') != -1:
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
    global opt, optimizer_A, optimizer_HR, writer, logPath, scheduler, schedulerR, val_loader, smallestLoss, secret_img

    opt = parser.parse_args()
    opt.Hnet_factor = 1
    opt.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print("[*]Running on device: {}".format(opt.device))

    ##################  Create the dirs to save the result ##################
    cur_time = time.strftime('%Y-%m-%d_H%H-%M-%S', time.localtime())
    assert opt.Hnet_inchannel == opt.Rnet_outchannel, 'Please make sure the channel of input secret image equal to the extracted secret image!'
    if opt.mode == 'train':
        Hnet_comment = 'Hnet_in{}out{}'.format(opt.Hnet_inchannel, opt.Hnet_outchannel)
        Rnet_comment = 'Rnet_in{}out{}'.format(opt.Rnet_inchannel, opt.Rnet_outchannel)
        opt.experiment_dir = os.path.join(opt.output_dir, cur_time+"_"+str(opt.imageSize)+"_"+opt.norm+"_"+opt.loss+"_"+str(opt.beta)+"_"+Hnet_comment+"_"+Rnet_comment+"_"+str(opt.Hnet_factor))
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

        logPath = opt.outlogs + '/{:}_log.txt'.format(opt.dataset)
        print_log(str(opt), logPath)
            
    elif opt.mode == 'generate':
        opt.experiment_dir = opt.output_dir
        print("[*]Saving the generation results at {}".format(opt.experiment_dir))

        opt.secret_dir = os.path.join(opt.experiment_dir, "pro_secret")
        print("[*]Generating processed secret images at: {}".format(opt.secret_dir))
        os.makedirs(opt.secret_dir, exist_ok=True)
            
        opt.cover_dir = os.path.join(opt.experiment_dir, "cover")
        print("[*]Generating cover images at: {}".format(opt.cover_dir))
        os.makedirs(opt.cover_dir, exist_ok=True)
            
        opt.container_dir = os.path.join(opt.experiment_dir, 'container')
        print("[*]Generating container images at: {}".format(opt.container_dir))
        os.makedirs(opt.container_dir, exist_ok=True)
        
    elif opt.mode == 'extract':
        opt.experiment_dir = opt.output_dir

        opt.rev_secret_dir = os.path.join(opt.experiment_dir, "rev_secret")
        print("[*]Generating retrieved secret images at: {}".format(opt.rev_secret_dir))
        os.makedirs(opt.rev_secret_dir, exist_ok=True)

    ################## Datasets and Networks ##################
    if opt.norm == 'instance':
        norm_layer = nn.InstanceNorm2d
    elif opt.norm == 'batch':
        norm_layer = nn.BatchNorm2d
    elif opt.norm == 'none':
        norm_layer = None
    else:
        raise ValueError("Invalid norm option. Must be one of [instance, batch, none]")

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

    if opt.mode == 'train':
        secret_dataset = ImageDataset(
            root = opt.secret_dir,
            transforms = transforms_secret)
        train_dataset_cover = ImageDataset(
            root = opt.train_dir,
            transforms = transforms_cover)
        val_dataset_cover = ImageDataset(
            root = opt.val_dir,
            transforms = transforms_cover)

        Anet = Adversary(input_nc=opt.Hnet_outchannel, output_nc=opt.Rnet_inchannel)
        Hnet = UnetGenerator(input_nc=opt.Hnet_inchannel, output_nc=opt.Hnet_outchannel, norm_layer=norm_layer, output_function=nn.Sigmoid())
        Rnet = RevealNet(input_nc=opt.Rnet_inchannel, output_nc=opt.Rnet_outchannel, norm_layer=norm_layer, output_function=nn.Sigmoid())

        # Using Kaiming Normalization to initialize network's parameters
        Anet.apply(weights_init).to(opt.device)
        Hnet.apply(weights_init).to(opt.device)
        Rnet.apply(weights_init).to(opt.device)

        # Load Pre-trained mode
        if opt.checkpoint != "":
            checkpoint = torch.load(opt.checkpoint)
            Anet.load_state_dict(checkpoint['A_state_dict'], strict=True)
            Hnet.load_state_dict(checkpoint['H_state_dict'], strict=True)
            Rnet.load_state_dict(checkpoint['R_state_dict'], strict=True)

        # Loss and Metric
        if opt.loss == 'l1':
            criterion = nn.L1Loss().cuda()
        elif opt.loss == 'l2':
            criterion = nn.MSELoss().cuda()
        else:
            raise ValueError("Invalid Loss Function. Must be one of [l1, l2]")

    elif opt.mode == 'generate':
        # Secret Image
        random_bits = np.ones((opt.imageSize, opt.imageSize), dtype=np.uint8)
        if opt.Hnet_inchannel == 1:
            random_bits = torch.from_numpy(random_bits).float().to(opt.device)
            random_bits = random_bits.unsqueeze(dim=0)
        else:
            random_bits = np.stack(arrays=(random_bits, random_bits, random_bits), axis=0)
            random_bits = torch.from_numpy(random_bits).float().to(opt.device)
            
        random_bits_img = tensor2img(random_bits.clone())
        random_bits_img.save(os.path.join(opt.experiment_dir, 'secret_img_ori.png'))

        secret_img = random_bits.unsqueeze(dim=0)

        cover_dataset = ImageDataset(root=opt.origin_dir, transforms=transforms_cover)

        Hnet = UnetGenerator(input_nc=opt.Hnet_inchannel, output_nc=opt.Hnet_outchannel, norm_layer=norm_layer, output_function=nn.Sigmoid())
        Hnet.to(opt.device)

        # Load Pre-trained mode
        if opt.checkpoint != "":
            checkpoint = torch.load(opt.checkpoint)
            Hnet.load_state_dict(checkpoint['H_state_dict'], strict=True)
                    
    elif opt.mode == 'extract':
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

        Anet = Adversary(input_nc=opt.Hnet_outchannel, output_nc=opt.Rnet_inchannel)
        Anet.to(opt.device)

        Rnet = RevealNet(input_nc=opt.Rnet_inchannel, output_nc=opt.Rnet_outchannel, norm_layer=norm_layer, output_function=nn.Sigmoid())
        Rnet.to(opt.device)

        # Load Pre-trained mode
        if opt.checkpoint != "":
            checkpoint = torch.load(opt.checkpoint)
            Anet.load_state_dict(checkpoint['A_state_dict'], strict=True)
            Rnet.load_state_dict(checkpoint['R_state_dict'], strict=True)

    # Print networks
    #print_network(Hnet)
    #print_network(Rnet)

    if opt.mode == 'train':
        params_A = Anet.parameters()
        optimizer_A = optim.Adam(params_A, lr=opt.lr, betas=(opt.beta_adam, 0.999))
        
        params_HR = list(list(Hnet.parameters()) + list(Rnet.parameters()))
        optimizer_HR = optim.Adam(params_HR, lr=opt.lr, betas=(opt.beta_adam, 0.999))
        scheduler = ReduceLROnPlateau(optimizer_HR, mode='min', factor=0.2, patience=8, verbose=True)

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
            adjust_learning_rate(optimizer_HR, epoch)

            train_loader = zip(secret_loader, train_loader_cover)
            val_loader = zip(secret_loader, val_loader_cover)

            ################## train ##################
            train(train_loader, epoch, Anet=Anet, Hnet=Hnet, Rnet=Rnet, criterion=criterion)

            ################## validation  ##################
            val_aloss, val_hloss, val_rloss, val_adiff, val_hdiff, val_rdiff = validation(val_loader, epoch, Anet=Anet, Hnet=Hnet, Rnet=Rnet, criterion=criterion)

            ################## adjust learning rate ##################
            scheduler.step(val_rloss) # 注意！这里只用 R 网络的 loss 进行 learning rate 的更新

            # Save the best model parameters
            """
            sum_diff = val_adiff + val_hdiff + val_rdiff
            is_best = sum_diff < globals()["smallestLoss"]
            globals()["smallestLoss"] = sum_diff
            """
            sum_loss = val_aloss + val_hloss + val_rloss             
            is_best = sum_loss < globals()["smallestLoss"]             
            globals()["smallestLoss"] = sum_loss

            stat_dict = {
                'epoch': epoch + 1,
                'A_state_dict': Anet.state_dict(),
                'H_state_dict': Hnet.state_dict(),
                'R_state_dict': Rnet.state_dict(),
                'optimizer_A': optimizer_A.state_dict(),
                'optimizer_HR': optimizer_HR.state_dict()
            }

            save_checkpoint(stat_dict, is_best)

        writer.close()
    elif opt.mode == 'generate':
        cover_loader = DataLoader(
            cover_dataset, 
            batch_size=opt.bs_generate,
            shuffle=False, 
            num_workers=int(opt.workers)
        )
        generate(dataset=cover_dataset, cov_loader=cover_loader, Hnet=Hnet)
    elif opt.mode == 'extract':
        container_loader = DataLoader(
            container_dataset, 
            batch_size=opt.bs_extract,
            shuffle=False, 
            num_workers=int(opt.workers)
        )
        extract(dataset=container_dataset, con_loader=container_loader, Rnet=Rnet)
        #extract(dataset=container_dataset, con_loader=container_loader, Anet=Anet, Rnet=Rnet)


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


def image_distorion(image):
    _, img_channel, img_hight, img_width = image.size()
    image_dis = torch.zeros_like(image)
    mask = torch.zeros_like(image)

    for idx, img_ori in enumerate(image):
        sides = np.random.randint(low=3, high=15, size=1)[0]
        radius = list(np.random.randint(low=5, high=50, size=sides))
        location = np.random.randint(low=max(radius), high=min(img_hight, img_width)-max(radius), size=2)

        contour_pts = draw_polygon(sides=sides, radius=radius, rotation=0, location=location) # (x,y)

        mask_ori = np.zeros((img_hight, img_width, img_channel), dtype=np.uint8)
        cv2.drawContours(image=mask_ori, contours=[contour_pts], contourIdx=-1, color=(255,255,255), thickness=-1)

        trans = transforms.Compose([transforms.ToTensor()])
        mask_ori = trans(mask_ori).to(opt.device)

        mask_dis = mask_ori * torch.rand_like(mask_ori)
        img_dis = img_ori * mask_dis + img_ori * (torch.ones_like(mask_ori) - mask_ori)

        image_dis[idx] = img_dis
        mask[idx] = mask_ori

    return image_dis, mask


def forward_pass(secret_img, cover_img, Anet, Hnet, Rnet, criterion, threshold):
    secret_img = secret_img.to(opt.device)
    cover_img = cover_img.to(opt.device)

    watermark = Hnet(secret_img) * opt.Hnet_factor
    
    container_img = watermark + cover_img

    errH = criterion(container_img, cover_img)  # Hiding net

    img_tampered, tampered_mask = image_distorion(container_img) 
    tampered_secret = secret_img * (torch.ones_like(tampered_mask) - tampered_mask) 

    if np.random.rand() > threshold:
        img_adversarial = Anet(img_tampered)
    else:
        img_adversarial = img_tampered

    rev_secret_img = Rnet(img_adversarial) 

    errR = criterion(rev_secret_img, tampered_secret)  # Reveal net

    errA = 15*criterion(img_adversarial, cover_img) - criterion(rev_secret_img, tampered_secret)

    diffA = (img_adversarial-cover_img).abs().mean()*255
    diffH = (container_img-cover_img).abs().mean()*255
    diffR = (rev_secret_img-tampered_secret).abs().mean()*255

    image_dict = {
        'cover': cover_img,
        'watermark': watermark,
        'container': container_img,
        'img_tampered': img_tampered,
        'tampered_secret': tampered_secret,
        'img_adversarial': img_adversarial,
        'rev_secret': rev_secret_img
    }
    data_dict = {
        'errA': errA, 
        'errH': errH, 
        'errR': errR, 
        'diffA': diffA, 
        'diffH': diffH, 
        'diffR': diffR
    }
    return image_dict, data_dict


def train(train_loader, epoch, Anet, Hnet, Rnet, criterion):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    Alosses = AverageMeter()  
    Hlosses = AverageMeter()  
    Rlosses = AverageMeter() 
    SumLosses = AverageMeter()
    Adiff = AverageMeter()
    Hdiff = AverageMeter()
    Rdiff = AverageMeter()

    start_time = time.time()

    for i, (secret_img, cover_img) in enumerate(train_loader):

        data_time.update(time.time() - start_time)
        
        ##### Train Adversarial Net #####
        Anet.train()
        Hnet.eval()
        Rnet.eval()
        image_dict, data_dict = forward_pass(secret_img, cover_img, Anet, Hnet, Rnet, criterion, threshold=0.0)
        
        Alosses.update(data_dict['errA'].data, opt.bs_train)  # A loss
        Adiff.update(data_dict['diffA'].data, opt.bs_train)

        err_A = data_dict['errA']
        optimizer_A.zero_grad()
        err_A.backward()
        optimizer_A.step()

        ##### Train Hiding Net and Reveal Net #####
        Anet.eval()
        Hnet.train()
        Rnet.train()
        image_dict, data_dict = forward_pass(secret_img, cover_img, Anet, Hnet, Rnet, criterion, threshold=0.5)

        betaerrR_secret = opt.beta * data_dict['errR']
        err_sum = data_dict['errH'] + betaerrR_secret
        optimizer_HR.zero_grad()
        err_sum.backward()
        optimizer_HR.step()

        Hlosses.update(data_dict['errH'].data, opt.bs_train)  # H loss
        Rlosses.update(data_dict['errR'].data, opt.bs_train)  # R loss
        Hdiff.update(data_dict['diffH'].data, opt.bs_train)
        Rdiff.update(data_dict['diffR'].data, opt.bs_train)

        SumLosses.update(data_dict['errH'].data + data_dict['errR'].data, opt.bs_train) # H loss + R loss

        ##### Time spent on one batch #####
        batch_time.update(time.time() - start_time)
        start_time = time.time()

        log = '[{:d}/{:d}][{:d}/{:d}] Loss_A: {:.6f} Loss_H: {:.6f} Loss_R: {:.6f} Loss_Sum: {:.6f} L1_A: {:.4f} L1_H: {:.4f} L1_R: {:.4f} datatime: {:.4f} batchtime: {:.4f}'.format(
            epoch, opt.max_epoch, i, opt.max_train_iters,
            Alosses.val, Hlosses.val, Rlosses.val, SumLosses.val, Adiff.val, Hdiff.val, Rdiff.val, 
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

    epoch_log = "Training[{:d}] Aloss={:.6f} Hloss={:.6f} Rloss={:.6f} SumLoss={:.6f} Adiff={:.4f} Hdiff={:.4f} Rdiff={:.4f} lr={:.6f} Epoch time={:.4f}".format(
        epoch, Alosses.avg, Hlosses.avg, Rlosses.avg, SumLosses.avg, Adiff.avg, Hdiff.avg, Rdiff.avg, optimizer_HR.param_groups[0]['lr'], batch_time.sum
    )
    print_log(epoch_log, logPath)

    writer.add_scalar("lr/lr", optimizer_HR.param_groups[0]['lr'], epoch)
    writer.add_scalar("lr/beta", opt.beta, epoch)
    writer.add_scalar('train/A_loss', Alosses.avg, epoch)
    writer.add_scalar('train/H_loss', Hlosses.avg, epoch)
    writer.add_scalar('train/R_loss', Rlosses.avg, epoch)
    writer.add_scalar('train/Sum_loss', SumLosses.avg, epoch)
    writer.add_scalar('train/A_diff', Adiff.avg, epoch)
    writer.add_scalar('train/H_diff', Hdiff.avg, epoch)
    writer.add_scalar('train/R_diff', Rdiff.avg, epoch)


def validation(val_loader, epoch, Anet, Hnet, Rnet, criterion):
    print("#################################################### validation begin ####################################################")    
    batch_time = AverageMeter()
    Alosses = AverageMeter()
    Hlosses = AverageMeter()
    Rlosses = AverageMeter()  
    Adiff = AverageMeter()
    Hdiff = AverageMeter()
    Rdiff = AverageMeter()

    start_time = time.time()

    Anet.eval()
    Hnet.eval()
    Rnet.eval()

    for i, (secret_img, cover_img) in enumerate(val_loader):

        image_dict, data_dict = forward_pass(secret_img, cover_img, Anet, Hnet, Rnet, criterion, threshold=0.0)

        Alosses.update(data_dict['errH'].data, opt.bs_train)  # A loss
        Hlosses.update(data_dict['errH'].data, opt.bs_train)  # H loss
        Rlosses.update(data_dict['errR'].data, 2*opt.bs_train)  # R loss
        Adiff.update(data_dict['diffA'].data, opt.bs_train)
        Hdiff.update(data_dict['diffH'].data, opt.bs_train)
        Rdiff.update(data_dict['diffR'].data, 2*opt.bs_train)
        
        if i == opt.max_val_iters-1:
            break

        batch_time.update(time.time() - start_time)
        start_time = time.time()

        val_log = "Validation[{:d}][{:d}/{:d}] val_Aloss = {:.6f}  val_Hloss = {:.6f} val_Rloss = {:.6f} val_Adiff = {:.4f} val_Hdiff = {:.4f} val_Rdiff={:.4f} batch time={:.4f}".format(
            epoch, i, opt.max_val_iters,
            Alosses.avg, Hlosses.avg, Rlosses.avg, Adiff.avg, Hdiff.avg, Rdiff.avg,
            batch_time.val
        )
        if i % opt.logFrequency == 0:
            print(val_log)
    
    save_result_pic(opt.dis_num, image_dict, epoch, i, opt.validationpics)
    
    val_log = "Validation[{:d}] val_Aloss = {:.6f} val_Hloss = {:.6f} val_Rloss = {:.6f} val_Adiff = {:.4f} val_Hdiff = {:.4f} val_Rdiff={:.4f} batch time={:.4f}".format(
        epoch, Alosses.avg, Hlosses.avg, Rlosses.avg, Adiff.avg, Hdiff.avg, Rdiff.avg, batch_time.sum)
    print_log(val_log, logPath)

    writer.add_scalar('validation/A_loss_avg', Alosses.avg, epoch)
    writer.add_scalar('validation/H_loss_avg', Hlosses.avg, epoch)
    writer.add_scalar('validation/R_loss_avg', Rlosses.avg, epoch)
    writer.add_scalar('validation/A_diff_avg', Adiff.avg, epoch)
    writer.add_scalar('validation/H_diff_avg', Hdiff.avg, epoch)
    writer.add_scalar('validation/R_diff_avg', Rdiff.avg, epoch)

    print("#################################################### validation end ####################################################")
    return Alosses.avg, Hlosses.avg, Rlosses.avg, Adiff.avg, Hdiff.avg, Rdiff.avg


def generate(dataset, cov_loader, Hnet):
    Hnet.eval()

    idx = 0
    for cover_batch in tqdm(cov_loader):
        cover_batch = cover_batch.to(opt.device)
        batch_size_cover, _, _, _ = cover_batch.size()

        H_input = secret_img.repeat(batch_size_cover, 1, 1, 1)

        pro_secret_batch = Hnet(H_input) * opt.Hnet_factor

        container_batch = pro_secret_batch + cover_batch

        for i, container in enumerate(container_batch):
            pro_secret_img = tensor2img(pro_secret_batch[i])
            cover_img = tensor2img(cover_batch[i])
            container_img = tensor2img(container)

            img_name = os.path.basename(dataset.image_paths[idx]).split('.')[0]

            pro_secret_img.save(os.path.join(opt.secret_dir, '{}.png'.format(img_name)))
            cover_img.save(os.path.join(opt.cover_dir, '{}.png'.format(img_name)))

            if opt.compression == 'Yes':
                container_img.save(os.path.join(opt.container_dir, '{}.jpg'.format(img_name)), quality=opt.com_quality, optimize=True, subsampling=0)
            else:
                container_img.save(os.path.join(opt.container_dir, '{}.png'.format(img_name)))

            idx += 1


def extract(dataset, con_loader, Rnet):
#def extract(dataset, con_loader, Anet, Rnet):
    Rnet.eval()

    idx = 0

    for container_batch in tqdm(con_loader):
        container_batch = container_batch.to(opt.device)

        rev_secret_bath = Rnet(container_batch)

        #adversarial_img = Anet(container_batch)
        #rev_secret_bath = Rnet(adversarial_img)

        for _, rev_secret in enumerate(rev_secret_bath):
            img_name = os.path.basename(dataset.image_paths[idx])

            detection_img = (rev_secret < opt.threshold).type(torch.int8)

            img_ori = tensor2array(detection_img)
            img_gray = cv2.cvtColor(img_ori, cv2.COLOR_RGB2GRAY)

            _, img_binary = cv2.threshold(src=img_gray, thresh=0, maxval=255, type=cv2.THRESH_BINARY) 

            contours, _ = cv2.findContours(img_binary, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

            result = np.zeros_like(img_gray)
            try:
                for contour in contours:
                    cv2.drawContours(image=result, contours=[contour], contourIdx=-1, color=(255,255,255), thickness=-1)
            except:
                None

            img_result = Image.fromarray(result.astype('uint8'))
            img_result.save(os.path.join(opt.rev_secret_dir, img_name))

            idx += 1


def tensor2array(var):
    var = var.cpu().detach().numpy().transpose([1,2,0])
    var[var < 0] = 0
    var[var > 1] = 1
    var = var * 255
    if var.shape[2] == 1:
        var = np.squeeze(var, axis=2)
    return var.astype('uint8')


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
    
    mask_secret_gap = image_dict['rev_secret'] - image_dict['tampered_secret']
    mask_secret_gap = (mask_secret_gap*10 + 0.5).clamp_(0.0, 1.0)

    adversarial_gap = image_dict['img_adversarial'] - image_dict['img_tampered']
    adversarial_gap = (adversarial_gap*10 + 0.5).clamp_(0.0, 1.0)

    fig = plt.figure(figsize=(40, 4*dis_num))
    gs = fig.add_gridspec(nrows=dis_num, ncols=10)
    for img_idx in range(dis_num):
        fig.add_subplot(gs[img_idx, 0])
        cov_img = tensor2img(image_dict['cover'][img_idx])
        plt.imshow(cov_img)
        plt.title("Cover")

        fig.add_subplot(gs[img_idx, 1])
        itmsec_img = tensor2img(image_dict['watermark'][img_idx])
        plt.imshow(itmsec_img)
        plt.title("Watermark")

        fig.add_subplot(gs[img_idx, 2])
        con_img = tensor2img(image_dict['container'][img_idx])
        plt.imshow(con_img)
        plt.title("Container")

        fig.add_subplot(gs[img_idx, 3])
        covgap_img = tensor2img(cover_gap[img_idx])
        plt.imshow(covgap_img)
        plt.title("Cover Gap")

        fig.add_subplot(gs[img_idx, 4])
        tam_img = tensor2img(image_dict['img_tampered'][img_idx])
        plt.imshow(tam_img)
        plt.title("Tampered Image")

        fig.add_subplot(gs[img_idx, 5])
        mask_img = tensor2img(image_dict['tampered_secret'][img_idx])
        plt.imshow(mask_img)
        plt.title("Tampered Mask")

        fig.add_subplot(gs[img_idx, 6])
        mask_img = tensor2img(image_dict['rev_secret'][img_idx])
        plt.imshow(mask_img)
        plt.title("Rev_Tampered Mask")

        fig.add_subplot(gs[img_idx, 7])
        masksec_gap = tensor2img(mask_secret_gap[img_idx])
        plt.imshow(masksec_gap)
        plt.title("Masked Secret Gap")

        fig.add_subplot(gs[img_idx, 8])
        tam_img = tensor2img(image_dict['img_adversarial'][img_idx])
        plt.imshow(tam_img)
        plt.title("Adversarial Image")

        fig.add_subplot(gs[img_idx, 9])
        mask_img = tensor2img(adversarial_gap[img_idx])
        plt.imshow(mask_img)
        plt.title("Adversarial Gap")
    
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