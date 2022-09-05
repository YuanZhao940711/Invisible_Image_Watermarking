import os
import time

from PIL import Image
from tqdm import tqdm

import numpy as np

import torch

from utils.common import AverageMeter, save_result_pic, print_log, tensor2img, tensor2array
from distortion import random_mask, block_mask, attack_layer



def forward_pass(opt, secret_img, cover_img, Hnet, Rnet, criterion):
    secret = secret_img.to(opt.device)
    cover = cover_img.to(opt.device)

    watermark = Hnet(secret) * opt.Hnet_factor
    
    container = watermark + cover

    errH = criterion(container, cover)  # Hiding net

    if opt.mask_mode == 'random':
        container_tampered, mask = random_mask(opt, container)
    elif opt.mask_mode == 'block':
        container_tampered, mask = block_mask(opt, container)
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

    if opt.attack == 'Yes':
        #print("attacking...")
        container_tampered = attack_layer(opt, container_tampered)
    else:
        #print("no attacking...")
        container_tampered = container_tampered
    
    retrieved = Rnet(container_tampered) 
    errR = criterion(retrieved, tampered)  # Reveal net

    mask_pd = (retrieved-secret).abs() 
    errM = criterion(mask_pd, mask) # Predicted mask loss

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
        'mask_pd': mask_pd
    }
    data_dict = {
        'errH': errH,
        'errR': errR,
        'errM': errM,
        'diffH': diffH, 
        'diffR': diffR
    }
    return image_dict, data_dict



def training(opt, train_loader, epoch, Hnet, Rnet, criterion, optimizer, writer):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    Hlosses = AverageMeter()  
    Rlosses = AverageMeter() 
    Mlosses = AverageMeter() 
    SumLosses = AverageMeter()
    Hdiff = AverageMeter()
    Rdiff = AverageMeter()

    # Switch to train mode
    Hnet.train()
    Rnet.train()

    start_time = time.time()

    for i, (secret_img, cover_img) in enumerate(train_loader):

        data_time.update(time.time() - start_time)

        image_dict, data_dict = forward_pass(opt, secret_img, cover_img, Hnet, Rnet, criterion)

        Hlosses.update(data_dict['errH'].data, opt.bs_train)  # H loss
        Rlosses.update(data_dict['errR'].data, opt.bs_train)  # R loss
        Mlosses.update(data_dict['errM'].data, opt.bs_train)  # R loss
        SumLosses.update(data_dict['errH'].data + data_dict['errR'].data + data_dict['errM'].data, opt.bs_train) # H loss + R loss + M Loss
        Hdiff.update(data_dict['diffH'].data, opt.bs_train)
        Rdiff.update(data_dict['diffR'].data, opt.bs_train)

        # Loss, backprop, and optimization step
        err_sum = data_dict['errH'] + data_dict['errR'] * opt.Rnet_beta + data_dict['errM'] * opt.Rnet_beta
        optimizer.zero_grad()
        err_sum.backward()
        optimizer.step()

        # Time spent on one batch
        batch_time.update(time.time() - start_time)
        start_time = time.time()

        log = '[{:d}/{:d}][{:d}/{:d}] Loss_H: {:.6f} Loss_R: {:.6f} Loss_M: {:.6f} Loss_Sum: {:.6f} L1_H: {:.4f} L1_R: {:.4f} datatime: {:.4f} batchtime: {:.4f}'.format(
            epoch, opt.max_epoch, i, opt.max_train_iters,
            Hlosses.val, Rlosses.val, Mlosses.val, SumLosses.val, Hdiff.val, Rdiff.val, 
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

    epoch_log = "Training[{:d}] Hloss={:.6f} Rloss={:.6f} Mloss={:.6f} SumLoss={:.6f} Hdiff={:.4f} Rdiff={:.4f} lr={:.6f} Epoch time={:.4f}".format(
        epoch, Hlosses.avg, Rlosses.avg, Mlosses.avg, SumLosses.avg, Hdiff.avg, Rdiff.avg, optimizer.param_groups[0]['lr'], batch_time.sum
    )
    print_log(epoch_log, opt.logPath)

    writer.add_scalar("lr/lr", optimizer.param_groups[0]['lr'], epoch)
    writer.add_scalar('train/H_loss', Hlosses.avg, epoch)
    writer.add_scalar('train/R_loss', Rlosses.avg, epoch)
    writer.add_scalar('train/M_loss', Mlosses.avg, epoch)
    writer.add_scalar('train/Sum_loss', SumLosses.avg, epoch)
    writer.add_scalar('train/H_diff', Hdiff.avg, epoch)
    writer.add_scalar('train/R_diff', Rdiff.avg, epoch)



def validation(opt, val_loader, epoch, Hnet, Rnet, criterion, writer):
    print("#################################################### validation begin ####################################################")    
    batch_time = AverageMeter()
    Hlosses = AverageMeter()
    Rlosses = AverageMeter()  
    Mlosses = AverageMeter()  
    Hdiff = AverageMeter()
    Rdiff = AverageMeter()

    start_time = time.time()

    Hnet.eval()
    Rnet.eval()

    for i, (secret_img, cover_img) in enumerate(val_loader):

        image_dict, data_dict = forward_pass(secret_img, cover_img, Hnet, Rnet, criterion)

        Hlosses.update(data_dict['errH'].data, opt.bs_train)  # H loss
        Rlosses.update(data_dict['errR'].data, opt.bs_train)  # R loss
        Rlosses.update(data_dict['errM'].data, opt.bs_train)  # M loss
        Hdiff.update(data_dict['diffH'].data, opt.bs_train)
        Rdiff.update(data_dict['diffR'].data, opt.bs_train)
        
        if i == opt.max_val_iters-1:
            break

        batch_time.update(time.time() - start_time)
        start_time = time.time()

        val_log = "Validation[{:d}][{:d}/{:d}] val_Hloss = {:.6f} val_Rloss = {:.6f} val_Mloss = {:.6f} val_Hdiff = {:.4f} val_Rdiff={:.4f} batch time={:.4f}".format(
            epoch, i, opt.max_val_iters,
            Hlosses.val, Rlosses.val, Mlosses.val, Hdiff.val, Rdiff.val, 
            batch_time.val
        )
        if i % opt.logFrequency == 0:
            print(val_log)
    
    save_result_pic(opt.dis_num, image_dict, epoch, i, opt.validationpics)
    
    val_log = "Validation[{:d}] val_Hloss = {:.6f} val_Rloss = {:.6f} val_Mloss = {:.6f} val_Hdiff = {:.4f} val_Rdiff={:.4f} batch time={:.4f}".format(
        epoch, Hlosses.avg, Rlosses.avg, Mlosses.avg, Hdiff.avg, Rdiff.avg, batch_time.sum)
    print_log(val_log, opt.logPath)

    writer.add_scalar('validation/H_loss_avg', Hlosses.avg, epoch)
    writer.add_scalar('validation/R_loss_avg', Rlosses.avg, epoch)
    writer.add_scalar('validation/M_loss_avg', Mlosses.avg, epoch)
    writer.add_scalar('validation/H_diff_avg', Hdiff.avg, epoch)
    writer.add_scalar('validation/R_diff_avg', Rdiff.avg, epoch)

    print("#################################################### validation end ####################################################")
    return Hlosses.avg, Rlosses.avg, Mlosses.avg, Hdiff.avg, Rdiff.avg



def generation(opt, dataset, cov_loader, secret_img, Hnet):
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



def revealing(opt, dataset, con_loader, Rnet):
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



def detection(opt, dataset, data_loader, secret_img, Rnet):
    secret_img = np.array(secret_img, dtype='int32')

    if Rnet != None:
        Rnet.eval()
        idx = 0
        for container_batch in tqdm(data_loader):
            container_batch = container_batch.to(opt.device)

            rev_secret_bath = Rnet(container_batch)

            for _, rev_secret in enumerate(rev_secret_bath):
                img_name = os.path.basename(dataset.image_paths[idx])

                rev_secret = tensor2array(rev_secret)

                gap = np.abs(secret_img - rev_secret)
                #gap = np.clip(np.abs(secret_img - rev_secret) * 256, a_min=0, a_max=255)

                gap_img = Image.fromarray(gap.astype('uint8'))
                gap_img.save(os.path.join(opt.mask_pd_dir, img_name))
                idx += 1
    else:
        idx = 0
        for rev_secret_batch in tqdm(data_loader):
            rev_secret_batch = rev_secret_batch.to(opt.device)

            for _, rev_secret in enumerate(rev_secret_batch):
                img_name = os.path.basename(dataset.image_paths[idx])

                rev_secret = tensor2array(rev_secret)

                gap = np.abs(secret_img - rev_secret)
                #gap = np.clip(np.abs(secret_img - rev_secret) * 256, a_min=0, a_max=255)

                #gap_img = Image.fromarray(gap.astype('uint8'))
                gap_img = Image.fromarray(gap.astype('uint8')).convert('L')
                
                gap_img.save(os.path.join(opt.mask_pd_dir, img_name))
                idx += 1        
