# encoding: utf-8

import argparse
import os
import shutil
import socket
import time

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.init as init
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from torch.utils.tensorboard import SummaryWriter
from torch.autograd import Variable
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from torchvision import transforms
from models.HidingUNet import UnetGenerator
from models.RevealNet import RevealNet
from torchvision.datasets import ImageFolder

import numpy as np
from skimage.metrics import structural_similarity as SSIM
from skimage.metrics import peak_signal_noise_ratio as PSNR

###
import matplotlib.pyplot as plt

from PIL import Image
from tqdm import tqdm
from utils.dataset import TrainDataset, ImageDataset
###


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default="train", help='train | val | test')
parser.add_argument('--workers', type=int, default=2, help='number of data loading workers')
parser.add_argument('--imageSize', type=int, default=256, help='the number of frames')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate, default=0.001')
parser.add_argument('--decay_round', type=int, default=10, help='learning rate decay 0.5 each decay_round')
parser.add_argument('--beta_adam', type=float, default=0.5, help='beta_adam for adam. default=0.5')
parser.add_argument('--Hnet', default='', help="path to Hidingnet (to continue training)")
parser.add_argument('--Rnet', default='', help="path to Revealnet (to continue training)")
parser.add_argument('--beta', type=float, default=0.75, help='hyper parameter of beta')
parser.add_argument('--test_diff', default='', help='another checkpoint folder')
parser.add_argument('--checkpoint', default='', help='checkpoint address')
parser.add_argument('--checkpoint_diff', default='', help='another checkpoint address')

parser.add_argument('--hostname', default=socket.gethostname(), help='the  host name of the running server')
parser.add_argument('--debug', type=bool, default=False, help='debug mode do not create folders')
parser.add_argument('--logFrequency', type=int, default=10, help='the frequency of print the log on the console')
parser.add_argument('--resultPicFrequency', type=int, default=100, help='the frequency of save the resultPic')
parser.add_argument('--norm', default='instance', help='batch or instance')
parser.add_argument('--loss', default='l2', help='l1 or l2')
parser.add_argument('--channel_cover', type=int, default=3, help='1: gray; 3: color')
parser.add_argument('--channel_secret', type=int, default=3, help='1: gray; 3: color')
parser.add_argument('--iters_per_epoch', type=int, default=2000, help='1: gray; 3: color')
parser.add_argument('--no_cover', type=bool, default=False, help='debug mode do not create folders')
parser.add_argument('--plain_cover', type=bool, default=False, help='use plain cover')
parser.add_argument('--noise_cover', type=bool, default=False, help='use noise cover')
parser.add_argument('--cover_dependent', type=bool, default=False, help='Whether the secret image is dependent on the cover image')
###
parser.add_argument('--bs_train', type=int, default=16, help='training batch size')
parser.add_argument('--bs_generate', type=int, default=16, help='generation batch size')
parser.add_argument('--bs_extract', type=int, default=16, help='extraction batch size')
parser.add_argument('--output_dir', default='', help='directory of outputing results')
parser.add_argument('--val_dir', type=str, default='', help='directory of validation images')
parser.add_argument('--train_dir', type=str, default='', help='directory of training images')
parser.add_argument('--secret_dir', type=str, default='', help='directory of secret images for training')

parser.add_argument('--max_epoch', type=int, default=50, help='number of epochs to train for')
parser.add_argument('--dis_num', type=int, default=5, help='number of example image for visualization')

parser.add_argument('--mode', type=str, default='', help='train | generate | extract')
parser.add_argument('--origin_dir', type=str, default='', help='directory of original images')
parser.add_argument('--secret_path', type=str, default='', help='path of origin secret image')
parser.add_argument('--container_dir', type=str, default='', help='directory of container images')
###


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
    global opt, optimizer, optimizerR, writer, logPath, scheduler, schedulerR, val_loader, smallestLoss, secret_img

    opt = parser.parse_args()
    opt.ngpu = torch.cuda.device_count()
    opt.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print("Running on device: {}".format(opt.device))

    cudnn.benchmark = True

    ##################  Create the dirs to save the result ##################
    if not opt.debug:
        cur_time = time.strftime('%Y-%m-%d_H%H-%M-%S', time.localtime())
        if opt.mode == 'train':
            secret_comment = 'color' if opt.channel_secret == 3 else 'gray'
            cover_comment = 'color' if opt.channel_cover == 3 else 'gray'
            comment = secret_comment+'_In_'+cover_comment
            opt.experiment_dir = os.path.join(opt.output_dir, cur_time+"_"+str(opt.imageSize)+"_"+opt.norm+"_"+opt.loss+"_"+str(opt.beta)+"_"+comment)
            print("Saving the experiment results at {}".format(opt.experiment_dir))

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

        elif opt.mode == 'generate':
            opt.experiment_dir = opt.output_dir
            print("Saving the generation results at {}".format(opt.experiment_dir))

            opt.outlogs = os.path.join(opt.experiment_dir, "GeneratingLogs")
            os.makedirs(opt.outlogs, exist_ok=True)

            opt.cover_dir = os.path.join(opt.experiment_dir, "cover")
            print("Generating cover images at: {}".format(opt.cover_dir))
            os.makedirs(opt.cover_dir, exist_ok=True)
            
            opt.container_dir = os.path.join(opt.experiment_dir, 'container')
            print("Generating container images at: {}".format(opt.container_dir))
            os.makedirs(opt.container_dir, exist_ok=True)
        
        elif opt.mode == 'extract':
            assert os.path.exists(opt.secret_path), "Cannot load secret image"

            opt.experiment_dir = opt.output_dir
            opt.outlogs = os.path.join(opt.experiment_dir, "ExtractingLogs")
            os.makedirs(opt.outlogs, exist_ok=True)

            opt.rev_secret_dir = os.path.join(opt.experiment_dir, "rev_secret")
            print("Generating retrieved secret images at: {}".format(opt.rev_secret_dir))
            os.makedirs(opt.rev_secret_dir, exist_ok=True)

    logPath = opt.outlogs + '/{:}_log.txt'.format(opt.dataset)

    if opt.debug:
        logPath = './debug/debug_logs/debug.txt'

    print_log(str(opt), logPath)

    ################## Datasets ##################
    transforms_gray = transforms.Compose([ 
                transforms.Grayscale(num_output_channels=1),
                transforms.Resize([opt.imageSize, opt.imageSize]), 
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
    transforms_color = transforms.Compose([ 
                transforms.Resize([opt.imageSize, opt.imageSize]), 
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
    if opt.channel_cover == 1:  
        transforms_cover = transforms_gray
    else:
         transforms_cover = transforms_color

    if opt.mode == 'train':
        train_dataset_cover = TrainDataset(
            opt.train_dir,
            transforms_cover)
        secret_dataset = TrainDataset(
            opt.secret_dir,
            transforms_cover)
        val_dataset_cover = TrainDataset(
            opt.val_dir,
            transforms_cover)
    elif opt.mode == 'generate':
        # Secret Image
        random_bits = np.ones((opt.imageSize, opt.imageSize))
        random_bits = np.stack(arrays=(random_bits, random_bits, random_bits), axis=0)
        random_bits = torch.from_numpy(random_bits).float().to(opt.device)
        
        random_bits_img = tensor2img(random_bits.clone())
        random_bits_img.save(os.path.join(opt.experiment_dir, 'secret_img_ori.png'))

        secret_img = random_bits.unsqueeze(dim=0)

        cover_dataset = ImageDataset(root=opt.origin_dir)
    elif opt.mode == 'extract':
        print("Load secret image at: {}".format(opt.secret_path))
        secret_img = Image.open(opt.secret_path).convert('RGB')
        secret_transform = transforms.Compose([
            transforms.ToTensor()
        ])
        secret_img = secret_transform(secret_img).to(opt.device)
        container_dataset = ImageDataset(root=opt.container_dir)
     
    ################## Hiding and Reveal Networks ##################
    assert opt.imageSize % 32 == 0
    num_downs = 5
    if opt.norm == 'instance':
        norm_layer = nn.InstanceNorm2d
    if opt.norm == 'batch':
        norm_layer = nn.BatchNorm2d
    if opt.norm == 'none':
        norm_layer = None
    if opt.cover_dependent:
        Hnet = UnetGenerator(input_nc=opt.channel_secret+opt.channel_cover, output_nc=opt.channel_cover, num_downs=num_downs, norm_layer=norm_layer, output_function=nn.Sigmoid)
    else:
        Hnet = UnetGenerator(input_nc=opt.channel_secret, output_nc=opt.channel_cover, num_downs=num_downs, norm_layer=norm_layer, output_function=nn.Tanh)
    #Rnet = RevealNet(input_nc=opt.channel_cover, output_nc=opt.channel_secret, nhf=64, norm_layer=norm_layer, output_function=nn.Sigmoid)
    Rnet = RevealNet(input_nc=opt.channel_cover, output_nc=opt.channel_secret, nhf=64, norm_layer=norm_layer, output_function=nn.Tanh)
    
    if opt.cover_dependent:
        assert opt.num_training == 1
        assert opt.no_cover == False

    ################## Kaiming Normalization ##################
    Hnet.apply(weights_init)
    Rnet.apply(weights_init)

    HnetD = UnetGenerator(input_nc=opt.channel_secret+opt.channel_cover, output_nc=opt.channel_cover, num_downs=num_downs, norm_layer=norm_layer, output_function=nn.Sigmoid)
    RnetD = RevealNet(input_nc=opt.channel_cover, output_nc=opt.channel_secret, nhf=64, norm_layer=norm_layer, output_function=nn.Sigmoid)
    HnetD.apply(weights_init)
    RnetD.apply(weights_init)

    ################## Always set to multiple GPU mode ##################
    Hnet = torch.nn.DataParallel(Hnet).to(opt.device)
    Rnet = torch.nn.DataParallel(Rnet).to(opt.device)

    HnetD = torch.nn.DataParallel(HnetD).to(opt.device)
    RnetD = torch.nn.DataParallel(RnetD).to(opt.device)

    ################## Load Pre-trained mode ##################
    if opt.checkpoint != "":
        if opt.checkpoint_diff != "":
            checkpoint = torch.load(opt.checkpoint)
            Hnet.load_state_dict(checkpoint['H_state_dict'], strict=True)
            Rnet.load_state_dict(checkpoint['R_state_dict'], strict=True)

            checkpointD = torch.load(opt.checkpoint_diff)
            HnetD.load_state_dict(checkpointD['H_state_dict'], strict=True)
            RnetD.load_state_dict(checkpointD['R_state_dict'], strict=True)
        else:
            checkpoint = torch.load(opt.checkpoint)
            Hnet.load_state_dict(checkpoint['H_state_dict'], strict=True)
            Rnet.load_state_dict(checkpoint['R_state_dict'], strict=True)

    # Print networks
    print_network(Hnet)
    print_network(Rnet)

    # Loss and Metric
    if opt.loss == 'l1':
        criterion = nn.L1Loss().cuda()
    if opt.loss == 'l2':
        criterion = nn.MSELoss().cuda()

    # Train the networks when opt.test is empty
    if opt.mode == 'train':
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
        print_log("Training is beginning .......................................................", logPath)
        for epoch in range(opt.max_epoch):
            adjust_learning_rate(optimizer, epoch)

            train_loader = zip(secret_loader, train_loader_cover)
            val_loader = zip(secret_loader, val_loader_cover)

            ################## train ##################
            train(train_loader, epoch, Hnet=Hnet, Rnet=Rnet, criterion=criterion)

            ################## validation  ##################
            val_hloss, val_rloss, val_hdiff, val_rdiff = validation(val_loader, epoch, Hnet=Hnet, Rnet=Rnet, criterion=criterion)

            ################## adjust learning rate ##################
            scheduler.step(val_rloss)

            # Save the best model parameters
            sum_diff = val_hdiff + val_rdiff
            is_best = sum_diff < globals()["smallestLoss"]
            globals()["smallestLoss"] = sum_diff

            stat_dict = {
                'epoch': epoch + 1,
                'H_state_dict': Hnet.state_dict(),
                'R_state_dict': Rnet.state_dict(),
                'optimizer' : optimizer.state_dict(),
            }

            save_checkpoint(stat_dict, is_best)

        if not opt.debug:
            writer.close()

     # For testing the trained network
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


def save_checkpoint(state, is_best):
    filename = 'checkpoint.pth.tar'

    checkpoint_path = os.path.join(opt.outckpts, filename)
    torch.save(state, checkpoint_path)

    if is_best:
        shutil.copyfile(checkpoint_path, os.path.join(opt.outckpts, 'best_checkpoint.pth.tar'))


def forward_pass(secret_img, cover_img, Hnet, Rnet, criterion):
    secret_img = secret_img.to(opt.device)
    cover_img = cover_img.to(opt.device)

    itm_secret_img = Hnet(secret_img)
    
    container_img = itm_secret_img + cover_img

    errH = criterion(container_img, cover_img)  # Hiding net

    rev_secret_img = Rnet(container_img) 
    errR = criterion(rev_secret_img, secret_img)  # Reveal net

    diffH = (container_img-cover_img).abs().mean()*255
    diffR = (rev_secret_img-secret_img).abs().mean()*255
    return cover_img, container_img, secret_img, rev_secret_img, errH, errR, diffH, diffR


def train(train_loader, epoch, Hnet, Rnet, criterion):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    Hlosses = AverageMeter()  
    Rlosses = AverageMeter() 
    SumLosses = AverageMeter()
    Hdiff = AverageMeter()
    Rdiff = AverageMeter()

    # Switch to train mode
    Hnet.train()
    Rnet.train()

    start_time = time.time()

    for i, (secret_img, cover_img) in enumerate(train_loader):

        data_time.update(time.time() - start_time)

        cover_imgv, container_img, secret_imgv_nh, rev_secret_img, errH, errR, diffH, diffR = forward_pass(secret_img, cover_img, Hnet, Rnet, criterion)

        Hlosses.update(errH.data, opt.bs_train)  # H loss
        Rlosses.update(errR.data, opt.bs_train)  # R loss
        Hdiff.update(diffH.data, opt.bs_train)
        Rdiff.update(diffR.data, opt.bs_train)

        # Loss, backprop, and optimization step
        betaerrR_secret = opt.beta * errR
        err_sum = errH + betaerrR_secret
        optimizer.zero_grad()
        err_sum.backward()
        optimizer.step()

        # Time spent on one batch
        batch_time.update(time.time() - start_time)
        start_time = time.time()

        log = '[{:d}/{:d}][{:d}/{:d}]\tLoss_H: {:.6f} Loss_R: {:.6f} L1_H: {:.4f} L1_R: {:.4f} \tdatatime: {:.4f} \tbatchtime: {:.4f}'.format(
            epoch, opt.max_epoch, i, opt.iters_per_epoch,
            Hlosses.val, Rlosses.val, Hdiff.val, Rdiff.val, data_time.val, batch_time.val
        )

        if i % opt.logFrequency == 0:
            print(log)

        if epoch <= 0 and i % opt.resultPicFrequency == 0:
            save_result_pic(opt.dis_num, cover_imgv, container_img.data, secret_imgv_nh, rev_secret_img.data, epoch, i, opt.trainpics)
            
        if i == opt.iters_per_epoch-1:
            break

    # To save the last batch only
    save_result_pic(opt.dis_num, cover_imgv, container_img.data, secret_imgv_nh, rev_secret_img.data, epoch, i, opt.trainpics)

    epoch_log = "Training[{:d}] Hloss={:.6f}\tRloss={:.6f}\tHdiff={:.4f}\tRdiff={:.4f}\tlr={:.6f}\t Epoch time={:.4f}".format(
        epoch, Hlosses.avg, Rlosses.avg, Hdiff.avg, Rdiff.avg, optimizer.param_groups[0]['lr'], batch_time.sum
    )
    print_log(epoch_log, logPath)

    if not opt.debug:
        writer.add_scalar("lr/lr", optimizer.param_groups[0]['lr'], epoch)
        writer.add_scalar("lr/beta", opt.beta, epoch)
        writer.add_scalar('train/H_loss', Hlosses.avg, epoch)
        writer.add_scalar('train/R_loss', Rlosses.avg, epoch)
        writer.add_scalar('train/sum_loss', SumLosses.avg, epoch)
        writer.add_scalar('train/H_diff', Hdiff.avg, epoch)
        writer.add_scalar('train/R_diff', Rdiff.avg, epoch)


def validation(val_loader, epoch, Hnet, Rnet, criterion):
    print("#################################################### validation begin ####################################################")
    
    start_time = time.time()

    Hnet.eval()
    Rnet.eval()
    
    batch_time = AverageMeter()
    Hlosses = AverageMeter()
    Rlosses = AverageMeter()  
    Hdiff = AverageMeter()
    Rdiff = AverageMeter()

    for i, (secret_img, cover_img) in enumerate(val_loader):

        cover_imgv, container_img, secret_imgv_nh, rev_secret_img, errH, errR, diffH, diffR = forward_pass(secret_img, cover_img, Hnet, Rnet, criterion)

        Hlosses.update(errH.data, opt.bs_train)  # H loss
        Rlosses.update(errR.data, opt.bs_train)  # R loss
        Hdiff.update(diffH.data, opt.bs_train)
        Rdiff.update(diffR.data, opt.bs_train)
        
        i_total = 200
        if i == i_total-1:
            break

        batch_time.update(time.time() - start_time)
        start_time = time.time()

        val_log = "Validation[{:d}] val_Hloss = {:.6f}\t val_Rloss = {:.6f}\t val_Hdiff = {:.6f}\t val_Rdiff={:.4f}\t batch time={:.2f}".format(
            epoch, Hlosses.val, Rlosses.val, Hdiff.val, Rdiff.val, batch_time.val
        )
        if i % opt.logFrequency == 0:
            print(val_log)
    
    save_result_pic(opt.dis_num, cover_imgv, container_img.data, secret_imgv_nh, rev_secret_img.data, epoch, i, opt.validationpics)
    
    val_log = "Validation[{:d}] val_Hloss = {:.6f}\t val_Rloss = {:.6f}\t val_Hdiff = {:.6f}\t val_Rdiff={:.4f}\t batch time={:.2f}".format(
        epoch, Hlosses.avg, Rlosses.avg, Hdiff.avg, Rdiff.avg, batch_time.sum)
    print_log(val_log, logPath)

    if not opt.debug:
        writer.add_scalar('validation/H_loss_avg', Hlosses.avg, epoch)
        writer.add_scalar('validation/R_loss_avg', Rlosses.avg, epoch)
        writer.add_scalar('validation/H_diff_avg', Hdiff.avg, epoch)
        writer.add_scalar('validation/R_diff_avg', Rdiff.avg, epoch)

    print("#################################################### validation end ####################################################")
    return Hlosses.avg, Rlosses.avg, Hdiff.avg, Rdiff.avg


def generate(dataset, cov_loader, Hnet):
    Hnet.eval()

    idx = 0
    for cover_batch in tqdm(cov_loader):
        cover_batch = cover_batch.to(opt.device)
        batch_size_cover, _, _, _ = cover_batch.size()

        H_input = secret_img.repeat(batch_size_cover, 1, 1, 1)

        itm_secret_img = Hnet(H_input)

        container_batch = itm_secret_img + cover_batch

        for i, container in enumerate(container_batch):
            cover_img = tensor2img(cover_batch[i])
            container_img = tensor2img(container)

            img_name = os.path.basename(dataset.image_paths[idx])

            cover_img.save(os.path.join(opt.cover_dir, img_name))
            container_img.save(os.path.join(opt.container_dir, img_name))
            
            idx += 1


def extract(dataset, con_loader, Rnet):
    Rnet.eval()

    idx = 0
    err_ratio = 0.0
    for container_batch in tqdm(con_loader):
        container_batch = container_batch.to(opt.device)
        batch_size_container, _, _, _ = container_batch.size()

        secret_img_batch = secret_img.repeat(batch_size_container, 1, 1, 1)

        rev_secret_batch = Rnet(container_batch) # rev_secret_img; bs x 3 x 128 x 128
        
        err = rev_secret_batch - secret_img_batch
        err = err.abs().sum(dim=(1,2,3)) / (3 * opt.imageSize * opt.imageSize)
        err_ratio += err
        
        for _, rev_secret in enumerate(rev_secret_batch):
            rev_secret_img = tensor2img(rev_secret)

            img_name = os.path.basename(dataset.image_paths[idx])

            rev_secret_img.save(os.path.join(opt.rev_secret_dir, img_name))

            idx += 1

    accuracy_percent = (1 - err_ratio/idx) * 100
    print("Total average correct bit reveal accuracy: {:.4f}".format(accuracy_percent.mean().item()))


def print_log(log_info, log_path, console=True):
    # print the info into the console
    if console:
        print(log_info)
    # debug mode don't write the log into files
    if not opt.debug:
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


def tensor2img(var):
    # var: 3 x 256 x 256 --> 256 x 256 x 3
    var = var.cpu().detach().numpy().transpose([1,2,0])
    var = ((var+1) / 2)
    var[var < 0] = 0
    var[var > 1] = 1
    var = var * 255
    return Image.fromarray(var.astype('uint8'))


# save result pic and the coverImg filePath and the secretImg filePath
def save_result_pic(dis_num, cover, container, secret, rev_secret, epoch, i, save_path):
    if opt.debug:
        save_path='./debug/debug_images'
    resultImgName = os.path.join(save_path, 'ResultPics_epoch{:03d}_batch{:04d}.png'.format(epoch, i))

    cover_gap = container - cover
    secret_gap = rev_secret - secret
    cover_gap = (cover_gap*10 + 0.5).clamp_(0.0, 1.0)
    secret_gap = (secret_gap*10 + 0.5).clamp_(0.0, 1.0)

    fig = plt.figure(figsize=(24, 4*dis_num))
    gs = fig.add_gridspec(nrows=dis_num, ncols=6)
    for img_idx in range(dis_num):
        fig.add_subplot(gs[img_idx, 0])
        cov_img = tensor2img(cover[img_idx])
        plt.imshow(cov_img)
        plt.title("Cover Image")

        fig.add_subplot(gs[img_idx, 1])
        con_img = tensor2img(container[img_idx])
        plt.imshow(con_img)
        plt.title("Container Image")

        fig.add_subplot(gs[img_idx, 2])
        covgap_img = tensor2img(cover_gap[img_idx])
        plt.imshow(covgap_img)
        plt.title("Cover_gap Image")

        fig.add_subplot(gs[img_idx, 3])
        sec_img = tensor2img(secret[img_idx])
        plt.imshow(sec_img)
        plt.title("Secret Image")

        fig.add_subplot(gs[img_idx, 4])
        revsec_img = tensor2img(rev_secret[img_idx])
        plt.imshow(revsec_img)
        plt.title("Rev_secret Image")

        fig.add_subplot(gs[img_idx, 5])
        secgap_img = tensor2img(secret_gap[img_idx])
        plt.imshow(secgap_img)
        plt.title("Secret_gap Image")
    
    plt.tight_layout()
    fig.savefig(resultImgName)
    plt.close(fig)


class AverageMeter(object):
    """
    Computes and stores the average and current value.
    """

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