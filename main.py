# -*- coding: UTF-8 -*- #
"""
@filename:main.py
@author:Young
@time:2024-01-13
"""
import os, warnings
import numpy as np
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")
import torch
import torch.nn as nn
from torch.utils.data import DataLoader,random_split,Dataset
import torchaudio
import librosa
from torchaudio import transforms
from torch import Tensor
from torchaudio import datasets
from train_utils import *
from model import *
from torch.utils.data import DataLoader, random_split, Dataset

# 关断言
# import os
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
# ——————————————————————————————————————————————————————————————————————————
# 参数配置
num_epochs = 50
BATCH_SIZE = 128
lr = 0.01
max_lr = 0.01
txt_name = "Lstm_MelSpectrogram_50_128_0.01"
model_name = "LSTM"

# 数据转换模式及模型选择
# 2 还没修好BUG
data_transform = 3

if data_transform == 1:
    print("MFCC Features classification")
    train_audio_transforms = nn.Sequential(
            torchaudio.transforms.MFCC(log_mels=False)
            )
    net = CNN1(num_class=35)
elif data_transform == 2:
    print("Mel Spectogram Features classification")
    train_audio_transforms = nn.Sequential(
            torchaudio.transforms.MelSpectrogram()
            )
    net = CNN2(num_class=35)
elif data_transform == 3:
    print("Mel Spectogram Features classification")
    train_audio_transforms = nn.Sequential(
            torchaudio.transforms.MelSpectrogram()
            )
    net = Lstm(num_class=35)
else:
    train_audio_transforms = None
    net = NN(num_class=35)

# ——————————————————————————————————————————————————————————————————————————
# 下载数据
train_audio_path = './data/SpeechCommands/speech_commands_v0.02/'
labels_dict=os.listdir(train_audio_path)
data = torchaudio.datasets.SPEECHCOMMANDS('./data/' , url = 'speech_commands_v0.02',
                                       folder_in_archive= 'SpeechCommands', download = False)

# 后端支持读写的文件
# print(torchaudio.list_audio_backends())

# 展示样本信息
# filename = "./data/SpeechCommands/speech_commands_v0.02/backward/0165e0e8_nohash_0.wav"
# waveform, sample_rate = torchaudio.load(filename, format='wav')
# print("Shape of waveform: {}".format(waveform.size()))
# print("Sample rate of waveform: {}".format(sample_rate))
# plt.figure()
# plt.plot(waveform.t().numpy())
# plt.plot(dataset[0][0].t())
# plt.show()

# 每一项都是一个元组，形式为：
# waveform、sample_rate、label、speaker_id、utterance_number
# 筛采样率，取波形和标签
wave = []
labels = []
#三十五类： 105829
#前十类：24793
for i in range(0,105829):
    if data[i][0].shape == (1, 16000):
        wave.append(data[i][0])
        labels.append(data[i][2])

# 音频处理的流行功能包括：
# Mel-Spectogram（Spectrogram和MelScale的组合）
# MFCC
# 可以使用 torchaudio 库获得所有这三种转换以及更多转换

# 将MFCC更改为MelSpectogram以获得梅尔标度频谱图
# specgram = torchaudio.transforms.MFCC()(wave[0])
# print("Shape of spectrogram: {}".format(specgram.size()))
# plt.figure(figsize=(10,5))
# plt.imshow(specgram[0,:,:].numpy())
# plt.colorbar()
# plt.show()

# 对比spectogram和MFCC
# specgram = torchaudio.transforms.MelSpectrogram()(wave[0])
# mfcc = torchaudio.transforms.MFCC()(wave[0])
# fig,ax = plt.subplots(1,2)
# ax[0].imshow(specgram[0,:,:].numpy())
# ax[1].imshow(mfcc[0,:,:].numpy())

# 音频数据加载器
# list_dir：标签列表
# 返回转换后的波形和标签
class SpeechDataLoader(Dataset):

    def __init__(self, data, labels, list_dir, transform=None, sequence_length=32):
        self.data = data
        self.labels = labels
        self.label_dict = list_dir
        self.transform = transform
        self.sequence_length = sequence_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        waveform = self.data[idx]

        if self.transform != None:
            waveform = self.transform(waveform)

        if self.labels[idx] in self.label_dict:
            out_labels = self.label_dict.index(self.labels[idx])

        return waveform, out_labels


# 设置训练数据集
dataset= SpeechDataLoader(wave,labels,labels_dict, train_audio_transforms)
print(labels_dict)

traindata, testdata = random_split(dataset, [round(len(dataset)*.8), round(len(dataset)*.2)])

trainloader = torch.utils.data.DataLoader(traindata, BATCH_SIZE, shuffle=True)

testloader = torch.utils.data.DataLoader(testdata, BATCH_SIZE, shuffle=True)

# 训练基本设置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # 决定用GPU还是CPU训练
net = net.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(),lr)
scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr,
                                              steps_per_epoch=int(len(trainloader)),
                                              epochs=num_epochs,
                                              anneal_strategy='linear')

# 开始训练
for epoch in range(0, num_epochs):
    train(net, trainloader, optimizer, scheduler, criterion, epoch, device)
    best_acc = test(net, testloader, optimizer, criterion, epoch, device, txt_name, model_name)