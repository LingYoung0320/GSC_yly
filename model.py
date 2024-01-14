# -*- coding: UTF-8 -*- #
"""
@filename:model.py
@author:Young
@time:2024-01-13
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.sparse import issparse


class NN(nn.Module):
    def __init__(self, num_class):
        super(NN, self).__init__()

        self.conv1 = nn.Conv1d(in_channels=1, out_channels=8, kernel_size=13, stride=1)
        self.dropout1 = nn.Dropout(0.3)

        self.conv2 = nn.Conv1d(in_channels=8, out_channels=16, kernel_size=11, stride=1)
        self.dropout2 = nn.Dropout(0.3)

        self.conv3 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=9, stride=1)
        self.dropout3 = nn.Dropout(0.3)

        self.conv4 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=7, stride=1)
        self.dropout4 = nn.Dropout(0.3)

        self.fc1 = nn.Linear(12416, 256)
        self.dropout5 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(256, 128)
        self.dropout6 = nn.Dropout(0.3)
        self.fc3 = nn.Linear(128, num_class)

    def forward(self, x):
        x = F.max_pool1d(F.relu(self.conv1(x)), kernel_size=3)
        x = self.dropout1(x)

        x = F.max_pool1d(F.relu(self.conv2(x)), kernel_size=3)
        x = self.dropout2(x)

        x = F.max_pool1d(F.relu(self.conv3(x)), kernel_size=3)
        x = self.dropout3(x)

        x = F.max_pool1d(F.relu(self.conv4(x)), kernel_size=3)
        x = self.dropout4(x)

        x = F.relu(self.fc1(x.reshape(-1, x.shape[1] * x.shape[2])))
        x = self.dropout5(x)

        x = F.relu(self.fc2(x))
        x = self.dropout6(x)

        x = self.fc3(x)

        # print(x.shape)
        return x

    # AdvDetectionModel
class CNN1(nn.Module):
    def __init__(self, num_class):
        super(CNN1, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=(2, 2), stride=(1, 1), padding=(0, 0))
        self.batch_norm1 = nn.BatchNorm2d(64)
        self.max_pool1 = nn.MaxPool2d((1, 3), padding=(0, 0))

        self.conv2 = nn.Conv2d(64, 64, kernel_size=(2, 2), stride=(1, 1), padding=(0, 0))
        self.batch_norm2 = nn.BatchNorm2d(64)
        self.max_pool2 = nn.MaxPool2d((2, 2), padding=(0, 0))

        self.conv3 = nn.Conv2d(64, 32, kernel_size=(2, 2), stride=(1, 1), padding=(0, 0))
        self.batch_norm3 = nn.BatchNorm2d(32)
        self.max_pool3 = nn.MaxPool2d((2, 2), padding=(0, 0))
        self.dropout = nn.Dropout(0.4)

        self.flatten = nn.Flatten()
        self.dense1 = nn.Linear(1440, 128)
        self.dropout2 = nn.Dropout(0.5)
        self.dense2 = nn.Linear(128, 35)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.batch_norm1(x)
        x = self.max_pool1(x)

        x = F.relu(self.conv2(x))
        x = self.batch_norm2(x)
        x = self.max_pool2(x)

        x = F.relu(self.conv3(x))
        x = self.batch_norm3(x)
        x = self.max_pool3(x)
        x = self.dropout(x)

        x = self.flatten(x)
        x = F.relu(self.dense1(x))
        x = self.dropout2(x)

        x = self.dense2(x)
        return F.softmax(x, dim=1)

# TrojaningAttacksModel
class CNN2(nn.Module):
    def __init__(self, num_class):
        super(CNN2, self).__init__()
        # TODO: Make this configurable
        input_shape = (1, 100, 40)
        self.conv1 = nn.Conv2d(1, 96, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.max_pool1 = nn.MaxPool2d((2, 2))

        self.conv2 = nn.Conv2d(96, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.max_pool2 = nn.MaxPool2d((2, 2))

        self.conv3 = nn.Conv2d(256, 384, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv4 = nn.Conv2d(384, 384, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv5 = nn.Conv2d(384, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.max_pool3 = nn.MaxPool2d((3, 3), stride=(2, 2))

        self.flatten = nn.Flatten()
        self.dense1 = nn.Linear(256 * 6 * 15, 256)
        self.dropout1 = nn.Dropout(0.3)
        self.dense2 = nn.Linear(256, 128)
        self.dropout2 = nn.Dropout(0.2)
        # TODO: Make this configurable
        self.dense3 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.max_pool1(x)

        x = F.relu(self.conv2(x))
        x = self.max_pool2(x)

        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = self.max_pool3(x)

        x = self.flatten(x)
        x = F.relu(self.dense1(x))
        x = self.dropout1(x)

        x = F.relu(self.dense2(x))
        x = self.dropout2(x)

        x = self.dense3(x)
        return F.softmax(x, dim=1)

# LstmAttModel
class LstmAtt(nn.Module):
        def __init__(self, num_class):
            super().__init__()

            self.conv1 = nn.Conv2d(1, 10, kernel_size=(5, 1), padding='same')
            self.bn1 = nn.BatchNorm2d(10)
            self.conv2 = nn.Conv2d(10, 1, kernel_size=(5, 1), padding='same')
            self.bn2 = nn.BatchNorm2d(1)

            self.lstm1 = nn.LSTM(40, 64, bidirectional=True, batch_first=True)
            self.lstm2 = nn.LSTM(128, 64, bidirectional=True, batch_first=True)

            self.linear1 = nn.Linear(128, 128)
            self.linear2 = nn.Linear(64, 64)
            self.linear3 = nn.Linear(64, 32)
            self.linear4 = nn.Linear(32, 10)

        def forward(self, x):
            x = F.relu(self.bn1(self.conv1(x)))
            x = F.relu(self.bn2(self.conv2(x)))

            # 获取x的维度信息
            batch_size, channels, seq_len, features = x.size()

            if x.dim() == 3:
                x = x.permute(0, 2, 1)
            elif x.dim() == 4:
                x = x.permute(0, 3, 1, 2)

            x = x.contiguous().view(batch_size, seq_len, -1)

            x, _ = self.lstm1(x)
            x, _ = self.lstm2(x)

            x_first = x[:, -1, :]

            query = self.linear1(x_first)
            att_scores = torch.matmul(query, x.transpose(1, 2))
            att_scores = F.softmax(att_scores, dim=1)

            att_vector = torch.matmul(att_scores, x)

            x = F.relu(self.linear2(att_vector))
            x = F.dropout(x, p=0.5)
            x = self.linear3(x)
            x = self.linear4(x)

            return x

class Lstm(torch.nn.Module):

    def __init__(self, num_class):
        super().__init__()
        self.seq = torch.nn.Sequential(
            torch.nn.Conv2d(1, 4, (1, 3), (1, 2), (0, 1)),
            torch.nn.BatchNorm2d(4),
            torch.nn.ReLU(),
            torch.nn.Conv2d(4, 4, (1, 3), (1, 2), (0, 1)),
            torch.nn.BatchNorm2d(4),
            torch.nn.ReLU(),
            torch.nn.Conv2d(4, 4, (1, 3), (1, 2), (0, 1)),
            torch.nn.BatchNorm2d(4),
            torch.nn.ReLU(),
            torch.nn.Conv2d(4, 8, 7, 2, 1),
            torch.nn.BatchNorm2d(8),
            torch.nn.ReLU(),
            torch.nn.Conv2d(8, 16, 3, 1, 0),
            torch.nn.BatchNorm2d(16),
            torch.nn.ReLU(),
            #torch.nn.Conv2d(8, 1, (10, 1)),
            #torch.nn.ReLU(),

        )

        self.rnn = nn.LSTM(32, 128, 2, batch_first=True, bidirectional=False)
        self.output_layer = nn.Linear(128, 35)
        self.apply(weight_init)

    def forward(self, x):
        h = self.seq(x)
        #print(h.shape)
        _n, _c, _h, _w = h.shape
        _x = h.permute(0, 2, 3, 1)
        #print(_x.shape)
        _x = _x.reshape(_n,_h,_w*_c)
        #print(_x.shape)
        h0 = torch.zeros(2 * 1, _n, 128).cuda()  # 初始化反馈值 num_layers * num_directions ,batch, hidden_size
        c0 = torch.zeros(2 * 1, _n, 128).cuda()
        hsn, (hn, cn) = self.rnn(_x, (h0, c0))
        out = self.output_layer(hsn[:, -1, :])

        return out#.reshape(-1, 10)

def weight_init(m):
    if (isinstance(m, nn.Conv2d)):
        nn.init.kaiming_normal_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif (isinstance(m, nn.Linear)):
        nn.init.kaiming_normal_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)