"""
程序用于对模型进行测试
"""
import sys
import argparse
import logging
from pathlib import Path

from unet.evaluate import evaluate as unet
from helpers import init_helper, data_helper


# 将模型名称和模型的测试程序对应起来
# EVALUATOR，字典型，键是模型名称，值是模型的测试程序
EVALUATOR = {
    'unet': unet
}


'''
get_evaluator函数
    输入参数包括：
        model_type，str型，是模型名称
'''
def get_evaluator(model_type):

    # 检查模型名称是否存在，并返回模型名称对应的训练程序
    assert model_type in EVALUATOR
    return EVALUATOR[model_type]


'''
主函数
'''
def main():

    # 初始化控制台参数
    parser = argparse.ArgumentParser()
    parser.add_argument('model', type=str, choices=('transformer'))
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--seed', type=int, default=12345)
    parser.add_argument('--splits', type=str, nargs='+', default=[])
    parser.add_argument('--sample-rate', type=int, default=15)
    parser.add_argument('--model-dir', type=str, required=True)
    parser.add_argument('--log-file', type=str, default='log.txt')

    # 添加有关transformer模型的控制台参数
    parser.add_argument('--num-head', type=int, default=8)
    parser.add_argument('--num-feature', type=int, default=1024)
    parser.add_argument('--num-hidden', type=int, default=128)
    parser.add_argument('--num-layers', type=int, default=1)
    args = parser.parse_args()

    # 初始化日志，并创建日志文件
    logger = init_helper.init_logger(args.model_dir, args.log_file)

    # 日志输出控制台参数
    logger.info(vars(args))

    # 存放模型的文件夹
    model_dir = Path(args.model_dir)

    # 根据模型名称获取测试程序
    evaluator = get_evaluator(args.model)

    # 遍历.yml数据集分割文件
    for split_path in args.splits:

        # 打开.yml数据集分割文件
        split_path = Path(split_path)
        splits = data_helper.load_yaml(split_path)

        # 初始化AverageMeter类用于求f分数的均值
        stats = data_helper.AverageMeter('fscore', 'pre', 'rec')

        # 遍历.yml分割文件中的每一种分割方案
        for split_idx, split in enumerate(splits):

            # 初始化随机数种子
            init_helper.set_random_seed(args.seed)

            # 日志输出分割文件名称和分割方案序号
            logger.info(f'Start testing on {split_path.stem}: split {split_idx}')

            # 获取对应的模型参数文件路径
            ckpt_path = data_helper.get_ckpt_path(model_dir, split_path, split_idx)

            # 分别对f分数最高的模型参数以及训练次数最多的模型参数进行测试并返回f分数
            eval = evaluator(args, split['test_keys'], ckpt_path, None)
            
            # 更新f分数均值
            stats.update(fscore=eval['fscore'], pre=eval['pre'], rec=eval['rec'])

            # 日志输出分割文件名称，分割方案序号和f分数
            logger.info(f'{split_path.stem} split {split_idx}: '
                        f'F-score: {eval["fscore"]:.4f}. Pre: {eval["pre"]:.4f}. Rec: {eval["pre"]:.4f}')

        # 日志输出分割文件名称和f分数均值
        logger.info(f'{split_path.stem}: '
                    f'F-score: {stats.fscore:.4f}. Pre: {stats.pre:.4f}. Rec: {stats.rec:.4f}')
    
    logging.shutdown()


if __name__ == '__main__':
    main()
