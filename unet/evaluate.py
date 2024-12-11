'''
transformer模型测试程序
'''
import sys
import torch
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import math
import h5py

from unet.model import Model
from helpers import data_helper, vsumm_helper
from helpers.noise_helper import Noise, Noise_adap


'''
evaluate函数
    输入参数包括：
        args，是控制台参数
        split，是数据集的一种分割方案
        save_path，str型，是模型参数文件路径
'''
def evaluate(args, split, ckpt_path, model=None):

    if model == None:

        # 创建模型
        model = Model()
        
        # 指定模型运行设备
        device = torch.device(args.device)
        model = model.to(device)

        # 加载模型参数
        state_dict = torch.load(str(ckpt_path), map_location=device)
        model.load_state_dict(state_dict)
    
    # 加载测试集
    val_set = data_helper.VideoDataset(split)
    val_loader = data_helper.DataLoader(val_set, shuffle=False)

    # 噪声类
    if args.adap_noise == True:
        noise_helper = Noise_adap(args.times)
    else:
        noise_helper = Noise(args.times)

    # 模型调整为测试模式
    device = torch.device(args.device)
    model = model.to(device)
    model.eval()

    # 创建AverageMeter类用于计算f分数和多样性分数的均值
    stats = data_helper.AverageMeter('fscore', 'pre', 'rec', 'ken', 'spe')

    with torch.no_grad():

        # 从测试集遍历训练数据
        # key，str型，是索引对应的视频索引，形如'../custom_data/custom_dataset.h5/video_0'
        # seq，(采样帧数量, 特征向量维数)的numpy数组，记录了对采样帧进行特征提取得到的特征向量
        # gtscore，(采样帧数量,)的numpy数组，记录了ground truth
        # cps，(镜头数量, 2)的numpy数组，记录了每个镜头的起始帧下标和结束帧下标
        # n_frames，np.int32型，是原始视频的视频帧数量
        # nfps，(镜头数量,)的numpy数组，记录了每个镜头包含的视频帧数量
        # picks，(采样帧数量,)的numpy数组，记录了每个采样帧在原始视频的下标
        # user_summary，(标注人数, 视频帧数量)的numpy数组，记录了每个标注人员的视频摘要
        # user_score，(标注人数, 采样帧)的numpy数组，记录了每个标注人员对每个采样帧的打分
        for test_key, seq, gtscore, cps, n_frames, nfps, picks, user_summary, user_score in val_loader:

            # prescore_list = []
            # for _ in range(user_score.shape[0]):
            # 将测试数据放到指定设备
            input = torch.tensor(seq, dtype=torch.float32).unsqueeze(0).to(device)
            if args.init_noise == 'gt':
                gtscore = torch.tensor(gtscore, dtype=torch.float32).unsqueeze(0).unsqueeze(-1).to(device)
                gtscore_norm = (gtscore * 2) - 1
                noise = torch.randn_like(gtscore_norm).to(device)
                noise_step = torch.tensor([args.times-1]).long().to(device)
                noise_prscore = noise_helper.add_noise(gtscore_norm, noise, noise_step)
            elif args.init_noise == 'noise':
                gtscore = torch.tensor(gtscore, dtype=torch.float32).unsqueeze(0).unsqueeze(-1).to(device)
                noise_prscore = torch.randn_like(gtscore).to(device)
            elif args.init_noise == 'exist':
                h5_res = h5py.File(args.exist_dir, 'r')
                noise_prscore = h5_res[test_key]['score'][...].astype(np.float32)
                noise_prscore -= noise_prscore.min()
                noise_prscore /= noise_prscore.max()
                noise_prscore = torch.tensor(noise_prscore, dtype=torch.float32).unsqueeze(0).unsqueeze(-1).to(device)
                noise_prscore = (noise_prscore * 2) - 1

            for step in reversed(range(0, args.times)):
                noise_step = torch.tensor([step]).long().to(device)

                # 前向传播
                pred_noise = model(noise_prscore, input, noise_step)

                # 去噪
                noise_prscore = noise_helper.sub_noise(noise_prscore, pred_noise, noise_step)
                
            # 生成摘要
            prscore = (noise_prscore + 1) / 2
            prscore = prscore.flatten().cpu().detach().numpy()
                # prescore_list.append(prscore)

            # prescore_list = np.array(prescore_list)
            # prscore = np.average(prescore_list, axis=0)
            pred_summ = vsumm_helper.get_keyshot_summ(prscore, cps, n_frames, nfps, picks)
            pred_score = vsumm_helper.get_keyshot_score(prscore, n_frames, picks)
            
            # 计算型预测的视频摘要和用户视频摘要的f分数
            eval_metric = 'max' if 'summe' in test_key else 'avg'
            fscore, pre, rec = vsumm_helper.get_summ_f1score(pred_summ, user_summary, eval_metric)
            # ken, spe = vsumm_helper.get_summ_coefficient(pred_score, user_score)
            video_num = test_key.split("/")[-1]
            if 'summe' in test_key:
                spe, ken = vsumm_helper.get_corr_coeff([pred_summ],[video_num],"summe",user_summary)
            elif 'tvsum' in test_key:
                spe, ken = vsumm_helper.get_corr_coeff([pred_score],[video_num],"tvsum",user_score)
        
            # 更新f分数和多样性分数的均值
            stats.update(fscore=fscore, pre=pre, rec=rec, ken=ken, spe=spe)

    # 返回f分数的均值和多样性分数的均值
    return {'fscore': stats.fscore, 'pre': stats.pre, 'rec': stats.rec, 'ken': stats.ken, 'spe': stats.spe}