"""
程序用于对模型进行训练
"""
import sys
import os
import argparse
import logging
from pathlib import Path

from unet.train import train as unet
from helpers import init_helper, data_helper

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
# 将模型名称和模型的训练程序对应起来
# TRAINER，字典型，键是模型名称，值是模型的训练程序
TRAINER = {
    'unet': unet
}


'''
get_trainer函数
    输入参数包括：
        model_type，str型，是模型名称
'''
def get_trainer(model_type):

    # 检查模型名称是否存在，并返回模型名称对应的训练程序
    assert model_type in TRAINER
    return TRAINER[model_type]


'''
主函数
'''
def main():

    # 初始化控制台参数
    parser = argparse.ArgumentParser()
    parser.add_argument('model', type=str, choices=('unet'))
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--seed', type=int, default=12345)
    parser.add_argument('--splits', type=str, nargs='+', default=[])
    parser.add_argument('--sample-rate', type=int, default=15)
    parser.add_argument('--model-dir', type=str, required=True)
    parser.add_argument('--exist-dir', type=str, default='')
    parser.add_argument('--log-file', type=str, default='log.txt')

    # 添加有关transformer模型的控制台参数
    parser.add_argument('--epoch', type=int, default=100)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--decay', type=float, default=0.01)
    parser.add_argument('--warmup', type=int, default=10)
    parser.add_argument('--times', type=int, default=100)
    parser.add_argument('--init-noise', type=str, default='exist')
    parser.add_argument('--adap-noise', action='store_true')
    args = parser.parse_args()

    # 初始化日志，并创建日志文件，日志会自动保存到日志文件当中
    logger = init_helper.init_logger(args.model_dir, args.log_file)

    # 日志输出控制台参数
    logger.info(vars(args))

    # 创建存放模型参数的文件夹，如果存在就报错
    model_dir = Path(args.model_dir)
    data_helper.get_ckpt_dir(model_dir).mkdir(parents=True, exist_ok=True)

    # 根据模型名称获取训练程序
    trainer = get_trainer(args.model)

    # 遍历.yml数据集分割文件
    for split_path in args.splits:

        # 打开.yml数据集分割文件
        split_path = Path(split_path)
        splits = data_helper.load_yaml(split_path)

        # 初始化AverageMeter类用于求f分数的均值
        stats = data_helper.AverageMeter('fscore', 'pre', 'rec', 'ken', 'spe')

        # 遍历.yml分割文件中的每一种分割方案
        for split_idx, split in enumerate(splits):

            # 初始化随机数种子
            init_helper.set_random_seed(args.seed)

            # 日志输出分割文件名称和分割方案序号
            logger.info(f'Start training on {split_path.stem}: split {split_idx}')

            # 获取对应的模型参数文件路径
            ckpt_path = data_helper.get_ckpt_path(model_dir, split_path, split_idx)

            # 训练并返回f分数作为训练结果
            result = trainer(args, split, ckpt_path, logger)

            # 更新f分数均值
            stats.update(fscore=result['max_fscore'], pre=result['max_pre'], rec=result['max_rec'], ken=result['max_ken'], spe=result['max_spe'])

        # 日志输出分割文件名称和f分数均值
        logger.info(f'Training done on {split_path.stem}. F-score: {stats.fscore:.4f}. Pre: {stats.pre:.4f}. Rec: {stats.rec:.4f}. Ken: {stats.ken:.4f}. Spe: {stats.spe:.4f}')
        with open(os.path.join(model_dir, 'results.txt'), 'a') as f:
            f.write(f'{result["args"]}\n')
            f.write(f'{stats.fscore:.4f},{stats.pre:.4f},{stats.rec:.4f},{stats.ken:.4f},{stats.spe:.4f}\n')
    
    logging.shutdown()


if __name__ == '__main__':
    main()
