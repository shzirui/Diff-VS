"""
功能性程序
"""
import sys
from typing import Iterable, List

import numpy as np
from ortools.algorithms.python import knapsack_solver as knapsack_solver0
from scipy.stats.stats import kendalltau, spearmanr
from scipy.stats import rankdata


'''
f1_score函数
    输入参数包括：
        pred，(视频帧数量,)的numpy数组，记录了模型预测的视频摘要
        test，(视频帧数量,)的numpy数组，记录了用户视频摘要
'''
def f1_score(pred: np.ndarray, test: np.ndarray) -> float:

    # 确保模型预测的视频摘要和用户视频摘要拥有一样的大小
    assert pred.shape == test.shape

    # 将模型预测的视频摘要和用户视频摘要转换为bool型numpy数组
    pred = np.asarray(pred, dtype=np.bool_)
    test = np.asarray(test, dtype=np.bool_)

    # 计算f1分数，分数越高越好
    overlap = (pred & test).sum()
    if overlap == 0:
        return [0.0, 0.0, 0.0]
    precision = overlap / pred.sum()
    recall = overlap / test.sum()
    f1 = 2 * precision * recall / (precision + recall)

    # f1，float型，是模型预测的视频摘要和用户视频摘要的f1分数
    return [float(f1), float(precision), float(recall)]


'''
knapsack函数
    输入参数包括：
        values，(镜头数量,)的numpy数组，记录了镜头级重要性分数
        weights，(镜头数量,)的numpy数组，记录了每个镜头包含的视频帧数量
        capacity，int型，根据域值指定视频摘要的长度限制
'''
def knapsack(values: Iterable[int],
             weights: Iterable[int],
             capacity: int
             ) -> List[int]:

    # 01背包算法，在weights之和不超过capacity的情况下，选取镜头尽可能使values最大
    knapsack_solver = knapsack_solver0.KnapsackSolver(
        knapsack_solver0.SolverType.KNAPSACK_DYNAMIC_PROGRAMMING_SOLVER, 'test'
    )

    values = list(values)
    weights = list(weights)
    capacity = int(capacity)

    knapsack_solver.init(values, [weights], [capacity])
    knapsack_solver.solve()
    packed_items = [x for x in range(0, len(weights))
                    if knapsack_solver.best_solution_contains(x)]

    # packed_items，(入选摘要的镜头数量,)的numpy数组，记录了入选摘要的镜头下标
    return packed_items


'''
get_keyshot_summ函数
    输入参数包括：
        pred: (采样帧数量,)的numpy数组，记录了模型预测的重要性分数
        cps，(镜头数量, 2)的numpy数组，记录了每个镜头的起始帧下标和结束帧下标，例如[0, 419]代表第0帧到第419帧共420帧
        n_frames，int型，是原始视频的视频帧数量
        nfps，(镜头数量,)的numpy数组，记录了每个镜头包含的视频帧数量
        picks，(采样帧数量,)的numpy数组，记录了每个采样帧在原始视频的下标
        proportion，float型，是视频摘要长度域值，用于指定视频摘要最长是原视频的多少倍
'''
def get_keyshot_summ(pred: np.ndarray,
                     cps: np.ndarray,
                     n_frames: int,
                     nfps: np.ndarray,
                     picks: np.ndarray,
                     proportion: float = 0.15
                     ) -> np.ndarray:

    # 确保模型预测重要性分数的长度等于采样帧数量
    assert pred.shape == picks.shape

    # 将采样帧在原始视频的下标转换为int32型的numpy数组
    picks = np.asarray(picks, dtype=np.int32)

    # 初始化帧级重要性分数
    # frame_scores，(视频帧数量,)的numpy数组，记录了帧级重要性分数
    frame_scores = np.zeros(n_frames, dtype=np.float32)

    # 遍历采样帧
    for i in range(len(picks)):

        # 获取采样帧对应的视频帧范围，pos_lo是下界，pos_hi是上界
        pos_lo = picks[i]
        pos_hi = picks[i + 1] if i + 1 < len(picks) else n_frames
        
        # 把范围内的帧级重要性分数置为采样帧的重要性分数
        frame_scores[pos_lo:pos_hi] = pred[i]

    # 初始化镜头级重要性分数
    # seg_scores，(镜头数量,)的numpy数组，记录了镜头级重要性分数
    seg_scores = np.zeros(len(cps), dtype=np.int32)

    # 将镜头范围内的帧级重要性分数的均值乘一千作为镜头级重要性分数
    for seg_idx, (first, last) in enumerate(cps):
        scores = frame_scores[first:last + 1]
        seg_scores[seg_idx] = int(1000 * scores.mean())

    # 根据域值指定视频摘要的长度限制
    limits = int(n_frames * proportion)

    # 使用01背包算法生成镜头级视频摘要
    # packed，(入选摘要的镜头数量,)的numpy数组，记录了入选摘要的镜头下标
    packed = knapsack(seg_scores, nfps, limits)

    # 初始化帧级视频摘要
    # summary，(视频帧数量,)的numpy数组，记录了帧级视频摘要
    summary = np.zeros(n_frames, dtype=np.bool_)

    # 遍历入选摘要的镜头
    for seg_idx in packed:

        # 获取镜头的起始帧下标和结束帧下标
        first, last = cps[seg_idx]

        # 将镜头范围内的帧都选入帧级视频摘要
        summary[first:last + 1] = True

    # summary，(视频帧数量,)的numpy数组，记录了帧级视频摘要
    return summary


def get_keyshot_score(pred: np.ndarray,
                     n_frames: int,
                     picks: np.ndarray,
                     ) -> np.ndarray:

    # 确保模型预测重要性分数的长度等于采样帧数量
    assert pred.shape == picks.shape

    # 将采样帧在原始视频的下标转换为int32型的numpy数组
    picks = np.asarray(picks, dtype=np.int32)

    # 初始化帧级重要性分数
    # frame_scores，(视频帧数量,)的numpy数组，记录了帧级重要性分数
    frame_scores = np.zeros(n_frames, dtype=np.float32)

    # 遍历采样帧
    for i in range(len(picks)):

        # 获取采样帧对应的视频帧范围，pos_lo是下界，pos_hi是上界
        pos_lo = picks[i]
        pos_hi = picks[i + 1] if i + 1 < len(picks) else n_frames
        
        # 把范围内的帧级重要性分数置为采样帧的重要性分数
        frame_scores[pos_lo:pos_hi] = pred[i]

    # summary，(视频帧数量,)的numpy数组，记录了帧级视频摘要
    return frame_scores


'''
get_frame_scores_summ函数
    输入参数包括：
        frame_scores: (视频帧数量,)的numpy数组，记录了标注的帧级重要性分数
        cps，(镜头数量, 2)的numpy数组，记录了每个镜头的起始帧下标和结束帧下标，例如[0, 419]代表第0帧到第419帧共420帧
        n_frames，int型，是原始视频的视频帧数量
        nfps，(镜头数量,)的numpy数组，记录了每个镜头包含的视频帧数量
        proportion，float型，是视频摘要长度域值，用于指定视频摘要最长是原视频的多少倍
'''
def get_frame_scores_summ(frame_scores: np.ndarray,
                          cps: np.ndarray,
                          n_frames: int,
                          nfps: np.ndarray,
                          proportion: float = 0.15
                          ) -> np.ndarray:

    # 初始化镜头级重要性分数
    # seg_scores，(镜头数量,)的numpy数组，记录了镜头级重要性分数
    seg_scores = np.zeros(len(cps), dtype=np.int32)

    # 将镜头范围内的帧级重要性分数的均值乘一千作为镜头级重要性分数
    for seg_idx, (first, last) in enumerate(cps):
        scores = frame_scores[first:last + 1]
        seg_scores[seg_idx] = int(1000 * scores.mean())

    # 根据域值指定视频摘要的长度限制
    limits = int(n_frames * proportion)

    # 使用01背包算法生成镜头级视频摘要
    # summary，(入选摘要的镜头数量,)的numpy数组，记录了入选摘要的镜头下标
    packed = knapsack(seg_scores, nfps, limits)

    # 初始化帧级视频摘要
    # summary，(视频帧数量,)的numpy数组，记录了帧级视频摘要
    summary = np.zeros(n_frames, dtype=np.bool_)

    # 遍历入选摘要的镜头
    for seg_idx in packed:

        # 获取镜头的起始帧下标和结束帧下标
        first, last = cps[seg_idx]

        # 将镜头范围内的帧都选入帧级视频摘要
        summary[first:last + 1] = True

    # summary，(视频帧数量,)的numpy数组，记录了帧级视频摘要
    return summary


'''
downsample_summ函数
    输入参数包括：
        summ，(视频帧数量,)的numpy数组,记录了模型预测的视频摘要
        sample_rate，int型，是采样率
'''
def downsample_summ(summ: np.ndarray, sample_rate: int) -> np.ndarray:

    # 根据采样率对模型预测的视频摘要进行降采样
    return summ[::sample_rate]


'''
get_summ_diversity函数
    输入参数包括：
        pred_summ，(采样帧数量,)的numpy数组,记录了采样后的模型预测的视频摘要
        features，(采样帧数量, 特征向量维数)的numpy数组，记录了对采样帧进行特征提取得到的特征向量
'''
def get_summ_diversity(pred_summ: np.ndarray,
                       features: np.ndarray
                       ) -> float:

    # 确保降采样后的视频摘要长度等于采样帧数量
    assert len(pred_summ) == len(features)

    # 将降采样后的视频摘要转换为bool型的numpy数组
    pred_summ = np.asarray(pred_summ, dtype=np.bool_)

    # 将入选视频摘要的采样帧的特征向量选出来
    pos_features = features[pred_summ]

    # 如果特征向量数量少于二，多样性分数为零
    if len(pos_features) < 2:
        return 0.0

    # 计算多样性分数，分数越高越好
    diversity = 0.0
    for feat in pos_features:
        diversity += (feat * pos_features).sum() - (feat * feat).sum()
    diversity /= len(pos_features) * (len(pos_features) - 1)
    
    # diversity，float型，是模型预测的视频摘要的多样性分数
    return diversity


'''
get_summ_f1score函数
    输入参数包括：
        pred_summ: (视频帧数量,)的numpy数组,记录了模型预测的视频摘要
        test_summ，(标注人数, 视频帧数量)的numpy数组,记录了用户的视频摘要
        eval_metric，str型，其值为'max'则输出最高的标注人员f分数，为'avg'则输出所有标注人员f分数的平均
'''
def get_summ_f1score(pred_summ: np.ndarray,
                     test_summ: np.ndarray,
                     eval_metric: str = 'avg'
                     ) -> float:

    # 将模型预测的视频摘要和用户的视频摘要转换为bool类型的numpy数组
    pred_summ = np.asarray(pred_summ, dtype=np.bool_)
    test_summ = np.asarray(test_summ, dtype=np.bool_)
    _, n_frames = test_summ.shape

    # pred_summ
    if pred_summ.size > n_frames:
        pred_summ = pred_summ[:n_frames]
    elif pred_summ.size < n_frames:
        pred_summ = np.pad(pred_summ, (0, n_frames - pred_summ.size))

    # 对每个标注人员的帧级用户视频摘要计算f1分数
    # f1s，列表型，记录了模型预测的视频摘要和用户视频摘要的f1分数
    eval = [f1_score(user_summ, pred_summ) for user_summ in test_summ]
    f1s = np.array(eval)[:,0]
    pres = np.array(eval)[:,1]
    recs = np.array(eval)[:,2]

    # 根据eval_metric选择f1分数的最大值或平均值
    if eval_metric == 'avg':
        final_f1 = np.mean(f1s)
        final_pre = np.mean(pres)
        final_rec = np.mean(recs)
    elif eval_metric == 'max':
        index = np.argmax(f1s)
        final_f1 = f1s[index]
        final_pre = pres[index]
        final_rec = recs[index]
    else:
        raise ValueError(f'Invalid eval metric {eval_metric}')

    # final_f1，float型，是模型预测的视频摘要和用户视频摘要的f1分数
    return float(final_f1), float(final_pre), float(final_rec)


def get_summ_coefficient(pred_score: np.ndarray,
                     test_score: np.ndarray
                     ) -> float:

    ken = [kendalltau(user_score, pred_score)[0] for user_score in test_score]
    ken = np.array(ken)
    ken = np.mean(ken)
    spe = [spearmanr(user_score, pred_score)[0] for user_score in test_score]
    spe = np.array(spe)
    spe = np.mean(spe)
    
    return float(ken), float(spe)


# Calculate Kendall's and Spearman's coefficients
def get_corr_coeff(pred_imp_scores, videos, dataset, user_scores=None):
    rho_coeff, tau_coeff = [], []
    if dataset=='summe':
        for pred_imp_score,video in zip(pred_imp_scores,videos):
            true = np.mean(user_scores,axis=0)
            rho_coeff.append(spearmanr(pred_imp_score,true)[0])
            tau_coeff.append(kendalltau(rankdata(pred_imp_score),rankdata(true))[0])
    elif dataset=='tvsum':
        for pred_imp_score,video in zip(pred_imp_scores,videos):
            pred_imp_score = np.squeeze(pred_imp_score).tolist()
            # user = int(video.split("_")[-1])

            # curr_user_score = user_scores[user-1]
            curr_user_score = user_scores

            tmp_rho_coeff, tmp_tau_coeff = [], []
            for annotation in range(len(curr_user_score)):
                true_user_score = curr_user_score[annotation]
                curr_rho_coeff, _ = spearmanr(pred_imp_score, true_user_score)
                curr_tau_coeff, _ = kendalltau(rankdata(pred_imp_score), rankdata(true_user_score))
                tmp_rho_coeff.append(curr_rho_coeff)
                tmp_tau_coeff.append(curr_tau_coeff)
            rho_coeff.append(np.mean(tmp_rho_coeff))
            tau_coeff.append(np.mean(tmp_tau_coeff))
    rho_coeff = np.array(rho_coeff).mean()
    tau_coeff = np.array(tau_coeff).mean()

    return rho_coeff, tau_coeff
