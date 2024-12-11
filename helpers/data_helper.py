"""
功能性程序
"""
import sys
import random
from os import PathLike
from pathlib import Path
from typing import Any, List, Dict

import h5py
import numpy as np
import yaml


'''
VideoDataset类
    初始化参数包括：
        keys，字符串列表型，记录了数据集包含的视频数据的视频索引，形如['../custom_data/custom_dataset.h5/video_0',...]
    __getitem__函数用于根据索引取出数据集中的数据，输入参数包括：
        index，int型，是索引
    __len__函数用于计算数据集大小
    get_datasets函数用于加载数据集涉及到的.h5文件，输入参数包括：
        keys，字符串列表型，记录了数据集包含的视频数据的视频索引，形如['../custom_data/custom_dataset.h5/video_0',...]
'''
class VideoDataset(object):
    def __init__(self, keys: List[str]):

        # self.keys，字符串列表型，记录了数据集包含的视频数据的视频索引，形如['../custom_data/custom_dataset.h5/video_0',...]
        self.keys = keys

        # 根据视频索引确定并加载视频数据所在的.h5文件
        # self.datasets，字典型，键是.h5文件路径，值是打开的.h5文件
        self.datasets = self.get_datasets(keys)

    def __getitem__(self, index):

        # 根据索引取出对应的视频索引
        # key，str型，是索引对应的视频索引，形如'../custom_data/custom_dataset.h5/video_0'
        key = self.keys[index]
        
        # 根据视频索引得到对应的视频数据
        # dataset_name，str型，是视频数据所对应的.h5文件路径
        # video_name，str型，是.h5文件内的视频索引
        # video_file，h5py._hl.group.Group类，记录了视频索引所对应的视频数据
        video_path = Path(key)
        dataset_name = str(video_path.parent)
        video_name = video_path.name
        video_file = self.datasets[dataset_name][video_name]

        # 将视频数据取出
        # seq，(采样帧数量, 特征向量维数)的numpy数组，记录了对采样帧进行特征提取得到的特征向量
        # gtscore，(采样帧数量,)的numpy数组，记录了ground truth
        # cps，(镜头数量, 2)的numpy数组，记录了每个镜头的起始帧下标和结束帧下标
        # n_frames，np.int32型，是原始视频的视频帧数量
        # nfps，(镜头数量,)的numpy数组，记录了每个镜头包含的视频帧数量
        # picks，(采样帧数量,)的numpy数组，记录了每个采样帧在原始视频的下标
        # user_summary，(标注人数, 视频帧数量)的numpy数组，记录了每个标注人员的视频摘要
        # user_score，(标注人数, 视频帧数量)的numpy数组，记录了每个标注人员对每个视频帧的打分
        seq = video_file['features'][...].astype(np.float32)
        gtscore = video_file['gtscore'][...].astype(np.float32)
        cps = video_file['change_points'][...].astype(np.int32)
        n_frames = video_file['n_frames'][...].astype(np.int32)
        nfps = video_file['n_frame_per_seg'][...].astype(np.int32)
        picks = video_file['picks'][...].astype(np.int32)
        user_summary = None
        user_score = None
        if 'user_summary' in video_file:
            user_summary = video_file['user_summary'][...].astype(np.float32)
        if 'user_score' in video_file:
            user_score = video_file['user_score'][...].astype(np.float32)
            # tvsum从user_score到gtscore的转换过程如下
            # user_score = np.average(user_score, axis=0)
            # user_score = user_score[::15]
            # user_score -= user_score.min()
            # user_score /= user_score.max()

        # 将ground truth的下界设置为零，上界设置为一
        # gtscore，(采样帧数量,)的numpy数组，记录了ground truth
        gtscore -= gtscore.min()
        gtscore /= gtscore.max()

        # key，str型，是索引对应的视频索引，形如'../custom_data/custom_dataset.h5/video_0'
        # images，(采样帧数量, 3, 224, 224)的numpy数组，记录了经过预处理后的采样帧
        # seq，(采样帧数量, 特征向量维数)的numpy数组，记录了对采样帧进行特征提取得到的特征向量
        # gtscore，(采样帧数量,)的numpy数组，记录了ground truth
        # cps，(镜头数量, 2)的numpy数组，记录了每个镜头的起始帧下标和结束帧下标
        # n_frames，np.int32型，是原始视频的视频帧数量
        # nfps，(镜头数量,)的numpy数组，记录了每个镜头包含的视频帧数量
        # picks，(采样帧数量,)的numpy数组，记录了每个采样帧在原始视频的下标
        # user_summary，(标注人数, 视频帧数量)的numpy数组，记录了每个标注人员对每个视频帧的打分
        # video_name，np.string_型，是视频名称
        return key, seq, gtscore, cps, n_frames, nfps, picks, user_summary, user_score

    def __len__(self):

        # len(self.keys)，int型，是数据集大小
        return len(self.keys)

    @staticmethod
    def get_datasets(keys: List[str]) -> Dict[str, h5py.File]:

        # 根据视频索引确定并加载视频数据所在的.h5文件
        # dataset_paths，字典型，记录了视频数据所在的.h5文件的路径
        # datasets，字典型，键是视频数据所在的.h5文件的路径，值是打开的.h5文件
        dataset_paths = {str(Path(key).parent) for key in keys}
        datasets = {path: h5py.File(path, 'r') for path in dataset_paths}
        
        # datasets，字典型，键是视频数据所在的.h5文件的路径，值是打开的.h5文件
        return datasets


'''
DataLoader类
    初始化参数包括：
        dataset，VideoDataset类，是数据集
        shuffle，bool型，指定数据集在输出数据时是否打乱顺序
    __iter__函数用于返回迭代器
    __next__函数用于返回下一条数据
'''
class DataLoader(object):
    def __init__(self, dataset: VideoDataset, shuffle: bool):

        # self.dataset，VideoDataset类，是数据集
        self.dataset = dataset

        # self.shuffle，bool型，指定数据集在输出数据时是否打乱顺序
        self.shuffle = shuffle

        # self.data_idx，列表，给数据集中的数据创建了索引
        self.data_idx = list(range(len(self.dataset)))

    def __iter__(self):

        # 迭代器置零
        self.iter_idx = 0
        
        # 如果需要打乱顺序，将数据集中的数据索引打乱
        if self.shuffle:
            random.shuffle(self.data_idx)
        
        return self

    def __next__(self):

        # 如果迭代器已经超出了数据集的大小，退出迭代
        if self.iter_idx == len(self.dataset):
            raise StopIteration

        # 用迭代器找到当前数据的索引
        curr_idx = self.data_idx[self.iter_idx]

        # 根据索引从数据集中获取数据
        # batch，元组，包含了VideoDataset类__getitem__方法的返回值
        batch = self.dataset[curr_idx]

        # 迭代器加一
        self.iter_idx += 1

        # batch，元组，包含了VideoDataset类__getitem__方法的返回值
        return batch


'''
AverageMeter类
    初始化参数包括：
        *keys，多个str型，是需要计算均值的变量名称
    update函数用于添加新的变量值，输入参数包括：
        **kwargs，多个键值对，键是str型的变量名称，值是float型的变量值
    __getattr__函数用于计算并返回变量的均值，输入参数包括：
        attr，str型，是变量名称
    _check_attr函数用于检查变量是否存在，输入参数包括：
        attr，str型，是变量名称
'''
class AverageMeter(object):
    def __init__(self, *keys: str):

        # self.totals，字典型，用于计算变量值的和，键是变量名称，值是零
        self.totals = {key: 0.0 for key in keys}

        # self.counts，字典型，用于计录变量值的个数，键是变量名称，值是零
        self.counts = {key: 0 for key in keys}

    def update(self, **kwargs: float) -> None:

        # 变量所有需要更新的变量
        for key, value in kwargs.items():

            # 检查变量是否存在
            self._check_attr(key)

            # 变量值的和更新
            self.totals[key] += value

            # 变量值的个数更新
            self.counts[key] += 1

    def __getattr__(self, attr: str) -> float:

        # 检查变量是否存在
        self._check_attr(attr)

        # 获取变量值的和
        total = self.totals[attr]

        # 获取变量值的个数
        count = self.counts[attr]

        # 返回变量的均值
        return total / count if count else 0.0

    def _check_attr(self, attr: str) -> None:

        # 检查变量名称是否存在于self.totals字典和self.counts字典中
        assert attr in self.totals and attr in self.counts


'''
get_ckpt_dir函数
    输入参数包括：
        model_dir，PathLike型，是存放模型的文件夹路径
'''
def get_ckpt_dir(model_dir: PathLike) -> Path:

    # 返回存放模型参数的文件夹路径
    return Path(model_dir) / 'checkpoint'


'''
get_ckpt_path函数
    输入参数包括：
        model_dir，PathLike型，是存放模型参数的文件夹路径
        split_path，PathLike型，是.yml数据集分割文件的路径
        split_index，int型，指定是第几种分割方案
'''
def get_ckpt_path(model_dir: PathLike,
                  split_path: PathLike,
                  split_index: int) -> Path:
    
    # 组合并返回分割方案所对应的模型参数文件路径
    split_path = Path(split_path)
    return get_ckpt_dir(model_dir) / f'{split_path.name}.{split_index}.pt'


'''
load_yaml函数
    输入参数包括：
        path，PathLike型，是.yml数据集分割文件的路径
'''
def load_yaml(path: PathLike) -> Any:

    # 根据路径打开.yml数据集分割文件
    with open(path) as f:
        obj = yaml.safe_load(f)
    
    # obj，list型，记录了.yml数据集分割文件中存放的各种分割方案
    return obj


'''
dump_yaml函数
    输入参数包括：
        obj，任意型，是需要保存的数据
        path，PathLike型，是.yml文件的路径
'''
def dump_yaml(obj: Any, path: PathLike) -> None:

    # 将数据根据路径保存到.yml文件
    with open(path, 'w') as f:
        yaml.dump(obj, f)
