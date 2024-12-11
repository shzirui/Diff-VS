'''
transformer损失函数
'''
import sys

import torch
from torch.nn import functional as F
from scipy.stats.stats import kendalltau, spearmanr


def mse_loss(pred: torch.Tensor,
             gtscore: torch.Tensor
             ) -> torch.Tensor:

    mseloss = F.mse_loss(pred, gtscore)
    
    return mseloss


def coe_loss(pred: torch.Tensor,
             gtscore: torch.Tensor
             ) -> torch.Tensor:

    pred_matrix = torch.sub(pred.squeeze(0), pred.squeeze(-1))
    gtscore_matrix = torch.sub(gtscore.squeeze(0), gtscore.squeeze(-1))
    matrix = pred_matrix - gtscore_matrix
    masked = matrix < 0
    matrix = torch.masked_select(matrix, masked)
    coeloss = torch.mean(matrix ** 2)
    
    return coeloss
