import os
import shutil

from PIL import Image

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.init as init



# Custom weights initialization called on netG and netD
def weights_init(m):
    if isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_out')
    elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1.0) 
        m.bias.data.fill_(0)



# Print the structure and parameters number of the net
def print_network(opt, net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print_log('Total number of parameters: %d' % num_params, opt.logPath)



def save_checkpoint(opt, state, is_best):
    filename = 'checkpoint.pth.tar'

    checkpoint_path = os.path.join(opt.outckpts, filename)
    torch.save(state, checkpoint_path)

    if is_best:
        shutil.copyfile(checkpoint_path, os.path.join(opt.outckpts, 'best_checkpoint.pth.tar'))



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



def adjust_learning_rate(opt, optimizer, epoch):
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

    fig = plt.figure(figsize=(44, 4*dis_num))
    gs = fig.add_gridspec(nrows=dis_num, ncols=11)
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
        plt.title("Mask_GT")

        fig.add_subplot(gs[img_idx, 10])
        maskpd_img = tensor2img(image_dict['mask_pd'][img_idx])
        plt.imshow(maskpd_img)
        plt.title("Mask_PD")
    
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