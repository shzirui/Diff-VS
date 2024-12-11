"""
功能性程序
"""
import sys
import logging
import random
from pathlib import Path

import numpy as np
import torch


'''
set_random_seed函数
    输入参数包括：
        seed，int型，是随机数种子
'''
def set_random_seed(seed: int) -> None:

    # 分别给random，numpy，torch设置随机数种子
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


'''
init_logger函数
    输入参数包括：
        log_dir，str型，是存放模型的文件夹路径
        log_file，str型，是日志文件名称
'''
def init_logger(log_dir: str, log_file: str):

    # 启动日志
    logger = logging.getLogger()

    # 输出格式为时间+日志信息
    format_str = r'[%(asctime)s] %(message)s'

    # 设置时间格式
    logging.basicConfig(
        level=logging.INFO,
        datefmt=r'%Y/%m/%d %H:%M:%S',
        format=format_str
    )

    # 根据路径创建存放模型的文件夹
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    # 创建日志文件并开始记录日志
    fh = logging.FileHandler(str(log_dir / log_file))
    fh.setFormatter(logging.Formatter(format_str))
    logger.addHandler(fh)

    return logger
