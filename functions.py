import os
import time

from PIL import Image
from tqdm import tqdm

import numpy as np

import torch

from distortion import attack_layer
from utils.common import AverageMeter, save_result_pic, print_log, tensor2img, tensor2array



def forward_pass(opt, secret_img, cover_img, Hnet, Rnet, criterion_dict):
    secret = secret_img.to(opt.device)
    cover = cover_img.to(opt.device)

    watermark = Hnet(secret) * opt.Hnet_factor
    
    container = watermark + cover

    errH = criterion_dict['Hnet_loss'](container, cover)  # Hiding net image loss

    if opt.attack == 'Yes':
        container_tampered, container_mask = attack_layer(opt, container)
    else:
        container_tampered = container
        container_mask = torch.zeros_like(container_tampered)
    
    if opt.Rloss_mode == 'secret0': # 根据container_mask指示的位置，将secret中“对应”区域的像素置0
        secret_tampered = secret * (torch.ones_like(container_mask) - container_mask) 
    elif opt.Rloss_mode == 'secret1': # 根据container_mask指示的位置，将secret中“对应”区域的像素置1
        secret_tampered = secret * (torch.ones_like(container_mask) - container_mask) + container_mask
    elif opt.Rloss_mode == 'mask':
        secret_tampered = container_mask
    
    secret_retrieved = Rnet(container_tampered) 
    errR_img = criterion_dict['Rnet_imgloss'](secret_retrieved, secret_tampered)  # Reveal net image loss

    container_watermark = torch.ones_like(container_mask) - container_mask
    secret_watermark = torch.ones_like(secret_retrieved) - (secret - secret_retrieved).abs() 
    errR_wat = criterion_dict['Rnet_watloss'](container_watermark, secret_watermark) # Reveal net watermark loss

    #mask_pd = (retrieved-secret).abs() 
    #errM = criterion(mask_pd, mask) # Predicted mask loss

    diffC = (container - cover).abs().mean()*255
    diffS = (secret_retrieved - secret_tampered).abs().mean()*255

    image_dict = {
        'secret': secret,
        'cover': cover,
        'watermark': watermark,
        'container': container,
        'container_tampered': container_tampered,
        'secret_tampered': secret_tampered,
        'secret_retrieved': secret_retrieved,
        'mask_gt': container_mask,
        'mask_pd': secret_watermark
    }
    data_dict = {
        'errH': errH,
        'errR_img': errR_img,
        'errR_wat': errR_wat,
        'diffC': diffC, 
        'diffS': diffS
    }
    return image_dict, data_dict



def training(opt, train_loader, epoch, Hnet, Rnet, criterion_dict, optimizer_dict, writer, scheduler_dict):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    Hlosses = AverageMeter()  
    RIlosses = AverageMeter() 
    RWlosses = AverageMeter() 
    SumLosses = AverageMeter()
    
    Cdiff = AverageMeter()
    Sdiff = AverageMeter()

    # Switch to train mode
    Hnet.train()
    Rnet.train()

    start_time = time.time()

    for i, (secret_img, cover_img) in enumerate(train_loader):

        data_time.update(time.time() - start_time)

        image_dict, data_dict = forward_pass(opt, secret_img, cover_img, Hnet, Rnet, criterion_dict)

        Hlosses.update(data_dict['errH'].data, opt.bs_train)  # H loss
        RIlosses.update(data_dict['errR_img'].data, opt.bs_train)  # R image loss
        RWlosses.update(data_dict['errR_wat'].data, opt.bs_train)  # R watermark loss
        SumLosses.update(data_dict['errH'].data + data_dict['errR_img'].data + data_dict['errR_wat'].data, opt.bs_train) # H loss + R loss + M Loss
        Cdiff.update(data_dict['diffC'].data, opt.bs_train)
        Sdiff.update(data_dict['diffS'].data, opt.bs_train)

        # Loss, backprop, and optimization step
        err_sum = data_dict['errH'] + (data_dict['errR_img'] + data_dict['errR_wat']) * opt.Rnet_beta
        optimizer_dict['opt_Hnet'].zero_grad()
        optimizer_dict['opt_Rnet'].zero_grad()        
        err_sum.backward()
        optimizer_dict['opt_Hnet'].step()
        optimizer_dict['opt_Rnet'].step()

        # Time spent on one batch
        batch_time.update(time.time() - start_time)
        start_time = time.time()

        log = '[{:d}/{:d}][{:d}/{:d}] Loss_H: {:.6f} Loss_R(img): {:.6f} Loss_R(wat): {:.6f} Loss_Sum: {:.6f} L1_C: {:.4f} L1_S: {:.4f} datatime: {:.4f} batchtime: {:.4f}'.format(
            epoch, opt.max_epoch, i, opt.max_train_iters,
            Hlosses.val, RIlosses.val, RWlosses.val, SumLosses.val, Cdiff.val, Sdiff.val, 
            data_time.val, batch_time.val
        )

        if i % opt.logFrequency == 0:
            print(log)

        if epoch <= 0 and i % opt.resultPicFrequency == 0:
            save_result_pic(opt.dis_num, image_dict, epoch, i, opt.trainpics)

        if i == opt.max_train_iters-1:
            break

    scheduler_dict['Hnet_scheduler'].step(Hlosses.avg) 
    scheduler_dict['Rnet_scheduler'].step(RIlosses.avg + RWlosses.avg) 

    # To save the last batch only
    save_result_pic(opt.dis_num, image_dict, epoch, i, opt.trainpics)

    epoch_log = "Training[{:d}] Hloss={:.6f} R(img)loss={:.6f} R(wat)loss={:.6f} SumLoss={:.6f} Cdiff={:.4f} Sdiff={:.4f} H_lr={:.6f} R_lr={:.6f} Epoch time={:.4f}".format(
        epoch, Hlosses.avg, RIlosses.avg, RWlosses.avg, SumLosses.avg, Cdiff.avg, Sdiff.avg, 
        optimizer_dict['opt_Hnet'].param_groups[0]['lr'], optimizer_dict['opt_Rnet'].param_groups[0]['lr'], 
        batch_time.sum
    )
    print_log(epoch_log, opt.logPath)

    writer.add_scalar("lr/H_lr", optimizer_dict['opt_Hnet'].param_groups[0]['lr'], epoch)
    writer.add_scalar("lr/R_lr", optimizer_dict['opt_Rnet'].param_groups[0]['lr'], epoch)    
    writer.add_scalar('train/H_loss', Hlosses.avg, epoch)
    writer.add_scalar('train/R_img_loss', RIlosses.avg, epoch)
    writer.add_scalar('train/R_wat_loss', RWlosses.avg, epoch)
    writer.add_scalar('train/Sum_loss', SumLosses.avg, epoch)
    writer.add_scalar('train/C_diff', Cdiff.avg, epoch)
    writer.add_scalar('train/S_diff', Sdiff.avg, epoch)



def validation(opt, val_loader, epoch, Hnet, Rnet, criterion_dict, writer):
    print("#################################################### validation begin ####################################################")    
    batch_time = AverageMeter()
    Hlosses = AverageMeter()  
    RIlosses = AverageMeter() 
    RWlosses = AverageMeter() 
    
    Cdiff = AverageMeter()
    Sdiff = AverageMeter()

    start_time = time.time()

    Hnet.eval()
    Rnet.eval()

    for i, (secret_img, cover_img) in enumerate(val_loader):

        image_dict, data_dict = forward_pass(opt, secret_img, cover_img, Hnet, Rnet, criterion_dict)

        Hlosses.update(data_dict['errH'].data, opt.bs_train)  # H loss
        RIlosses.update(data_dict['errR_img'].data, opt.bs_train)  # R loss
        RWlosses.update(data_dict['errR_wat'].data, opt.bs_train)  # M loss
        Cdiff.update(data_dict['diffC'].data, opt.bs_train)
        Sdiff.update(data_dict['diffS'].data, opt.bs_train)
        
        if i == opt.max_val_iters-1:
            break

        batch_time.update(time.time() - start_time)
        start_time = time.time()

        val_log = "Validation[{:d}][{:d}/{:d}] val_Hloss = {:.6f} val_Rloss = {:.6f} val_Wloss = {:.6f} val_Cdiff = {:.4f} val_Sdiff={:.4f} batch time={:.4f}".format(
            epoch, i, opt.max_val_iters,
            Hlosses.val, RIlosses.val, RWlosses.val, Cdiff.val, Sdiff.val, 
            batch_time.val
        )
        if i % opt.logFrequency == 0:
            print(val_log)
    
    save_result_pic(opt.dis_num, image_dict, epoch, i, opt.validationpics)
    
    val_log = "Validation[{:d}] val_Hloss = {:.6f} val_R(img)loss = {:.6f} val_R(wat)loss = {:.6f} val_Cdiff = {:.4f} val_Sdiff={:.4f} batch time={:.4f}".format(
        epoch, Hlosses.avg, RIlosses.avg, RWlosses.avg, Cdiff.avg, Sdiff.avg, batch_time.sum)
    print_log(val_log, opt.logPath)

    writer.add_scalar('validation/H_loss_avg', Hlosses.avg, epoch)
    writer.add_scalar('validation/R_img_loss_avg', RIlosses.avg, epoch)
    writer.add_scalar('validation/R_wat_loss_avg', RWlosses.avg, epoch)
    writer.add_scalar('validation/C_diff_avg', Cdiff.avg, epoch)
    writer.add_scalar('validation/S_diff_avg', Sdiff.avg, epoch)

    losses_dict = {
        'Hlosses': Hlosses.avg,
        'RIlosses': RIlosses.avg,
        'RWlosses': RWlosses.avg,
        'Cdiff': Cdiff.avg,
        'Sdiff': Sdiff.avg
    }    

    print("#################################################### validation end ####################################################")
    return losses_dict



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
