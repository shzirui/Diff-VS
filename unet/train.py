'''
transformer模型训练程序
'''
import logging
import os
import sys

import numpy as np
import torch
import math
from helpers import data_helper
from helpers.noise_helper import Noise, Noise_adap
from torch import nn
from unet.evaluate import evaluate
from unet.losses import mse_loss, coe_loss
from unet.model import Model


'''
xavier_init函数
'''
def xavier_init(module):

    # 模型参数初始化
    cls_name = module.__class__.__name__
    if 'Linear' in cls_name or 'Conv' in cls_name:
        nn.init.xavier_uniform_(module.weight, gain=np.sqrt(2.0))
        if module.bias is not None:
            nn.init.constant_(module.bias, 0.1)


'''
adjust_learning_rate函数
    输入参数包括：
        optimizer，是模型优化器
        warmup_epoch，int型，是热启动epoch数量
        current_epoch，int型，是当前epoch数量
        lr, float型，是默认学习率
        now_loss，float型，是当前epoch损失
        prev_loss，float型，是上个epoch损失
'''
def adjust_learning_rate(optimizer, warmup_epoch, current_epoch, lr, now_loss=None, prev_loss=None):
    
    # 观察第一个出现损失上升的epoch的学习率，设置为初始学习率，热启动epoch数量一般为10
    if current_epoch <= warmup_epoch:
        lr = lr * current_epoch / warmup_epoch
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
    
    #  每当出现损失上升，把学习率置为之前的0.9倍
    elif now_loss != None and prev_loss != None:
        delta = prev_loss - now_loss
        if delta < 0:
            for param_group in optimizer.param_groups:
                param_group['lr'] *= 0.99


'''
train函数
    输入参数包括：
        args，是控制台参数
        split，是数据集的一种分割方案
        save_path，str型，是模型参数文件路径
'''
def train(args, split, save_path, logger):

    # max_val_fscore，float型，测试集上最高的f分数
    result = {'max_fscore': -1,
              'max_pre': -1,
              'max_rec': -1,
              'args': args}

    # 创建模型
    model = Model()
    device = torch.device(args.device)
    model = model.to(device)
    model.apply(xavier_init)
    
    # 损失函数初始化
    now_loss = None
    prev_loss = None

    # 设置模型优化器
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(parameters, lr=args.lr, weight_decay=args.decay)

    # 加载训练集并打乱顺序
    train_set = data_helper.VideoDataset(split['train_keys'])
    train_loader = data_helper.DataLoader(train_set, shuffle=False)

    # 噪声类
    if args.adap_noise == True:
        noise_helper = Noise_adap(args.times)
    else:
        noise_helper = Noise(args.times)

    # 遍历训练次数
    for epoch in range(args.epoch):

        # 模型调整为训练模式并调整学习率
        adjust_learning_rate(optimizer, args.warmup, epoch+1, args.lr, now_loss, prev_loss)
        model.train()

        # 创建AverageMeter类用于计算损失函数的均值
        stats = data_helper.AverageMeter('loss')

        # 从训练集遍历训练数据
        # key，str型，是索引对应的视频索引，形如'../custom_data/custom_dataset.h5/video_0'
        # seq，(采样帧数量, 特征向量维数)的numpy数组，记录了对采样帧进行特征提取得到的特征向量
        # gtscore，(采样帧数量,)的numpy数组，记录了ground truth
        # cps，(镜头数量, 2)的numpy数组，记录了每个镜头的起始帧下标和结束帧下标
        # n_frames，np.int32型，是原始视频的视频帧数量
        # nfps，(镜头数量,)的numpy数组，记录了每个镜头包含的视频帧数量
        # picks，(采样帧数量,)的numpy数组，记录了每个采样帧在原始视频的下标
        # user_summary，(标注人数, 视频帧数量)的numpy数组，记录了每个标注人员的视频摘要
        # user_score，(标注人数, 采样帧数量)的numpy数组，记录了每个标注人员对每个采样帧的打分
        for key, seq, gt, _, _, _, _, _, user_score in train_loader:
            # tvsum从user_score到训练数据的转换过程如下
            if user_score is not None:
                user_score = user_score[:, ::args.sample_rate]
                user_score -= np.expand_dims(np.min(user_score, axis=1), axis=1)
                user_score /= np.expand_dims(np.max(user_score, axis=1), axis=1)
            else:
                user_score = np.expand_dims(gt, axis=0)

            for gtscore in user_score:
                # 将训练数据放到指定设备
                # input, (1, 采样帧数量, 特征向量维数)
                # gtscore, (1, 采样帧数量, 1)
                input = torch.tensor(seq, dtype=torch.float32).unsqueeze(0).to(device)
                gtscore = torch.tensor(gtscore, dtype=torch.float32).unsqueeze(0).unsqueeze(-1).to(device)
                gtscore_norm = (gtscore * 2) - 1

                # 重要性图加噪
                # noise, (1, 采样帧数量, 1)
                # noise_step, (1)
                # noise_gtscore, (1, 采样帧数量, 1)
                noise = torch.randn_like(gtscore_norm).to(device)
                noise_step = torch.randint(0, args.times, (1, )).long().to(device)
                noise_gtscore = noise_helper.add_noise(gtscore_norm, noise, noise_step)

                # 前向传播
                # pred_noise, (1, 采样帧数量, 1)
                # prscore, (1, 采样帧数量, 1)
                pred_noise = model(noise_gtscore, input, noise_step)

                # 计算损失函数
                prscore_norm = noise_helper.min_noise(noise_gtscore, pred_noise, noise_step)
                prscore = (prscore_norm + 1) / 2
                loss = mse_loss(prscore, gtscore)
                # if key=="datasets/eccv16_dataset_summe_google_pool5.h5/video_19" or key=="datasets/eccv16_dataset_ovp_google_pool5.h5/video_3":
                #     continue
                # print(prscore.shape)
                # print(gtscore.shape)
                # print(prscore.squeeze(-1))
                # print(gtscore.squeeze(-1))
                # print(key)
                # print(loss)
                # print(loss.item())
                # print("-------------------------------")
                # 模型优化
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # 更新损失函数的均值
                stats.update(loss=loss.item())

        # 在测试集上进行测试，得到f分数
        eval = evaluate(args, split['test_keys'], None, model)

        # 如果测试集上的f分数有所提高，保存当前模型参数并覆盖掉原来的模型参数
        if result['max_fscore'] < eval['fscore']:
            result['max_fscore'] = eval['fscore']
            result['max_pre'] = eval['pre']
            result['max_rec'] = eval['rec']
            result['max_ken'] = eval['ken']
            result['max_spe'] = eval['spe']
            torch.save(model.state_dict(), str(save_path))

        # 日志输出训练轮次，学习率，损失函数，测试集上当前轮次的f分数和历次最高的f分数
        lr = optimizer.param_groups[0]['lr']
        logger.info(f'Epoch: {epoch}/{args.epoch} '
                    f'Lr: {lr:.7f} '
                    f'Loss: {stats.loss:.4f} '
                    f'F-score cur/max: {eval["fscore"]:.4f}/{result["max_fscore"]:.4f} '
                    f'Ken cur/max: {eval["ken"]:.4f}/{result["max_ken"]:.4f} '
                    f'Spe cur/max: {eval["spe"]:.4f}/{result["max_spe"]:.4f}')
        
        prev_loss = now_loss
        now_loss = stats.loss

    return result
