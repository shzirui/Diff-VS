'''
功能性程序
'''
import sys
from os import PathLike
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from numpy import linalg
from torch import nn
from torchvision import transforms, models

from kts.cpd_auto import cpd_auto



'''
FeatureExtractor类
    初始化参数包括：
        device，指定设备
    run函数用于对采样帧进行预处理和特征提取，输入参数包括：
        img，(高, 宽, 3)的numpy数组，记录了一个采样帧
'''
class FeatureExtractor(object):
    def __init__(self, device):

        # 指定设备
        self.device = device

        # 创建transforms类，用于对采样帧进行预处理
        # transforms.Resize(256)，将采样帧按比例缩放成256*256大小
        # transforms.CenterCrop(224)，将采样帧中心裁剪成224*224大小
        # transforms.ToTensor()，将采样帧转换为tensor类型
        # transforms.Normalize()，将采样帧进行正则化处理
        self.preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        # 创建预训练好的googlenet模型，用于对采样帧进行特征提取
        self.model = models.googlenet(pretrained=True)

        # 将googlenet模型的最后两层删除，仅保留前面用于特征提取的层
        self.model = nn.Sequential(*list(self.model.children())[:-2])

        # 将googlenet模型设置成测试模式并放到指定设备上
        self.model = self.model.to(device).eval()

    def run(self, img: np.ndarray) -> np.ndarray:
        
        # 使用PIL库打开采样帧
        img = Image.fromarray(img)

        # 使用transforms类对采样帧进行预处理，并添加batch_size维度
        # img，(3, 224, 224)的tensor，记录了预处理后的采样帧
        # batch，(1, 3, 224, 224)的tensor，记录了添加batch_size维度后的采样帧
        img = self.preprocess(img)
        batch = img.unsqueeze(0)

        # 使用googlenet模型对采样帧进行特征提取，并将特征向量转换为numpy类型
        # feat，(特征向量维数,)的numpy数组，记录了采样帧的特征向量
        with torch.no_grad():
            feat = self.model(batch.to(self.device))
            feat = feat.squeeze().cpu().numpy()
        assert feat.shape == (1024,), f'Invalid feature shape {feat.shape}: expected 1024'

        # 对特征向量进行正则化处理
        # feat，(特征向量维数,)的numpy数组，记录了采样帧的特征向量
        feat /= linalg.norm(feat) + 1e-10
        
        # img，(3, 224, 224)的numpy数组，记录了预处理后的采样帧
        # feat，(特征向量维数,)的numpy数组，记录了采样帧的特征向量
        return feat


'''
VideoPreprocessor类
    初始化参数包括：
        sample_rate，int型，是采样率
        device，指定设备
    get_features函数用于对原始视频进行帧采样，预处理和特征提取，输入参数包括：
        video_path，PathLike类，记录了原始视频路径
    kts函数用于对原始视频进行镜头划分，输入参数包括：
        n_frames，int型，是视频帧数量
        features，(采样帧数量, 特征向量维数)的numpy数组，记录了对采样帧进行特征提取得到的特征向量
    run函数用于先对原始视频进行帧采样和特征提取，再对原始视频进行镜头划分，输入参数包括：
        video_path，PathLike类，是原始视频路径
'''
class VideoPreprocessor(object):
    def __init__(self, sample_rate: int, device) -> None:

        # 创建特征提取模型
        self.model = FeatureExtractor(device)

        # 采样率
        self.sample_rate = sample_rate

    def get_features(self, video_path: PathLike, count):

        # 用cv2库打开原始视频
        video_path = Path(video_path)
        cap = cv2.VideoCapture(str(video_path))
        frame_num = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_num += 1
        cap.release()
        cap = cv2.VideoCapture(str(video_path))
        fps_change = frame_num/count
        assert cap is not None, f'Cannot open video: {video_path}'

        # 初始化预处理采样帧数组，特征向量数组和视频帧数量
        features = []
        n_frames = 0
        real_frames = 0
        flag = 0

        # 遍历视频帧
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            real_frames += 1
            if real_frames == frame_num:
                num = int(count - flag)
                flag = count
            else:
                num = int(int(real_frames / fps_change) - flag)
                flag = int(real_frames / fps_change)

            for i in range(num):
                # 当视频帧下标是采样率的倍数，对该视频帧进行采样和处理
                if n_frames % self.sample_rate == 0:

                    # 将采样帧转化为RGB格式
                    # frame，(高, 宽, 3)的numpy数组，记录了RGB格式的采样帧
                    cvt_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                    # 对采样帧进行预处理和特征提取，并构建预处理采样帧数组和特征向量数组
                    # img，(3, 224, 224)的numpy数组，记录了预处理后的采样帧
                    # feat，(特征向量维数,)的numpy数组，记录了采样帧的特征向量
                    feat = self.model.run(cvt_frame)
                    features.append(feat)

                # 视频帧数量加一
                n_frames += 1

        # 用cv2库关闭原始视频
        cap.release()

        # 将特征向量数组转换为numpy格式
        # features，(采样帧数量, 特征向量维数)的numpy数组，记录了对采样帧进行特征提取得到的特征向量
        features = np.array(features)
        
        # n_frames，int型，是视频帧数量
        # features，(采样帧数量, 特征向量维数)的numpy数组，记录了对采样帧进行特征提取得到的特征向量
        return n_frames, features

    def kts(self, n_frames, features):

        # 根据采样帧数量和采样率，计算每个采样帧在原始视频的下标
        # seq_len，int型，是采样帧数量
        # picks，(采样帧数量,)的numpy数组，记录了每个采样帧在原始视频的下标，下标从0开始
        seq_len = len(features)
        picks = np.arange(0, seq_len) * self.sample_rate

        # KTS镜头分割算法，得到镜头分割结果
        # change_points，(镜头数量, 2)的numpy数组，记录了每个镜头的起始帧下标和结束帧下标，例如[0, 419]代表第0帧到第419帧共420帧
        kernel = np.matmul(features, features.T)
        change_points, _ = cpd_auto(kernel, seq_len - 1, 1, verbose=False)
        change_points *= self.sample_rate
        change_points = np.hstack((0, change_points, n_frames))
        begin_frames = change_points[:-1]
        end_frames = change_points[1:]
        change_points = np.vstack((begin_frames, end_frames - 1)).T

        # 计算每个镜头包含的视频帧数量
        # n_frame_per_seg，(镜头数量,)的numpy数组，记录了每个镜头包含的视频帧数量
        n_frame_per_seg = end_frames - begin_frames
        
        # change_points，(镜头数量, 2)的numpy数组，记录了每个镜头的起始帧下标和结束帧下标，例如[0, 419]代表第0帧到第419帧共420帧
        # n_frame_per_seg，(镜头数量,)的numpy数组，记录了每个镜头包含的视频帧数量
        # picks，(采样帧数量,)的numpy数组，记录了每个采样帧在原始视频的下标，下标从0开始
        return change_points, n_frame_per_seg, picks

    def run(self, video_path: PathLike, count):

        # 对原始视频进行帧采样和特征提取
        # n_frames，int型，是视频帧数量
        # images，(采样帧数量, 3, 224, 224)的numpy数组，记录了预处理后的采样帧
        # features，(采样帧数量, 特征向量维数)的numpy数组，记录了对采样帧进行特征提取得到的特征向量
        n_frames, features = self.get_features(video_path, count)

        # 对原始视频进行镜头划分
        # cps，(镜头数量, 2)的numpy数组，记录了每个镜头的起始帧下标和结束帧下标，例如[0, 419]代表第0帧到第419帧共420帧
        # nfps，(镜头数量,)的numpy数组，记录了每个镜头包含的视频帧数量
        # picks，(采样帧数量,)的numpy数组，记录了每个采样帧在原始视频的下标，下标从0开始
        cps, nfps, picks = self.kts(n_frames, features)
        
        # n_frames，int型，是视频帧数量
        # images，(采样帧数量, 3, 224, 224)的numpy数组，记录了预处理后的采样帧
        # features，(采样帧数量, 特征向量维数)的numpy数组，记录了对采样帧进行特征提取得到的特征向量
        # cps，(镜头数量, 2)的numpy数组，记录了每个镜头的起始帧下标和结束帧下标，例如[0, 419]代表第0帧到第419帧共420帧
        # nfps，(镜头数量,)的numpy数组，记录了每个镜头包含的视频帧数量
        # picks，(采样帧数量,)的numpy数组，记录了每个采样帧在原始视频的下标，下标从0开始
        return n_frames, features, cps, nfps, picks
