# -*- coding: utf8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary

import numpy as np
import sys


def upsample(x, out_size):
    return F.interpolate(x, size=out_size, mode='linear', align_corners=False)


def bn_relu_conv(in_, out_, kernel_size=3, stride=1, bias=False):
    padding = kernel_size // 2
    return nn.Sequential(nn.BatchNorm1d(in_),
                         nn.ReLU(inplace=True),
                         nn.Conv1d(in_, out_, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias))

####################   double-branch ##############
class FCNGRU(nn.Module):
    """FCN for motif mining"""
    def __init__(self):
        super(FCNGRU, self).__init__()
        # encode process
        self.conv1 = nn.Conv1d(in_channels=4, out_channels=64, kernel_size=13, padding=6)
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=7, padding=3)
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=5, padding=2)
        self.pool3 = nn.MaxPool1d(kernel_size=2, stride=2)
        # decode process
        self.gru = nn.GRU(input_size=64, hidden_size=64, batch_first=True)
        self.gru_drop = nn.Dropout(p=0.5)
        # self.aap = nn.AdaptiveAvgPool1d(1)
        self.blend3 = bn_relu_conv(64, 64, kernel_size=3)
        self.blend2 = bn_relu_conv(64, 4, kernel_size=3)
        self.blend1 = bn_relu_conv(4, 1, kernel_size=3)
        # regression head
        c_in = 256
        self.linear1 = nn.Linear(c_in, 64)
        self.drop = nn.Dropout(p=0.5)
        self.linear2 = nn.Linear(64, 1)
        # general functions
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=0.2)
        self._init_weights()

    def _init_weights(self):
        """Initialize the new built layers"""
        for layer in self.modules():
            if isinstance(layer, (nn.Conv1d, nn.Linear)):
                # nn.init.kaiming_uniform_(layer.weight, mode='fan_in', nonlinearity='relu')
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0)
            elif isinstance(layer, nn.BatchNorm1d):
                nn.init.constant_(layer.weight, 1)
                nn.init.constant_(layer.bias, 0)

    def forward(self, data):
        """Construct a new computation graph at each froward"""
        b, _, _ = data.size()
        # encode process
        skip1 = data
        out1 = self.conv1(data)
        out1 = self.relu(out1)
        out1 = self.pool1(out1)
        out1 = self.dropout(out1)
        skip2 = out1
        out1 = self.conv2(out1)
        out1 = self.relu(out1)
        out1 = self.pool2(out1)
        out1 = self.dropout(out1)
        skip3 = out1
        out1 = self.conv3(out1)
        out1 = self.relu(out1)
        out1 = self.pool3(out1)
        out1 = self.dropout(out1)

        skip4 = out1.permute(0, 2, 1)
        up4, _ = self.gru(skip4)
        up4 = self.gru_drop(up4)
        up4 = up4.permute(0, 2, 1)

        # decode process

        up3 = upsample(up4, skip3.size()[-1])
        up3 = up3 + skip3
        up3 = self.blend3(up3)
        up2 = upsample(up3, skip2.size()[-1])
        up2 = up2 + skip2
        up2 = self.blend2(up2)
        up1 = upsample(up2, skip1.size()[-1])
        up1 = up1 + skip1
        up1 = self.blend1(up1)
        out_dense = self.sigmoid(up1)
        out_dense = out_dense.view(b, -1)

        # regression
        out2 = skip4.reshape(b, -1)
        out2 = self.linear1(out2)
        out2 = self.relu(out2)
        out2 = self.drop(out2)
        out_regression = self.linear2(out2)

        return out_regression, out_dense


# ## #################  single branch #################
# class FCNGRU(nn.Module):
#     """FCN for motif mining"""
#     def __init__(self, motiflen=13):
#         super(FCNGRU, self).__init__()
#         # encode process
#         self.conv1 = nn.Conv1d(in_channels=4, out_channels=64,  kernel_size=motiflen, padding=motiflen//2)
#         self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)
#         self.conv2 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=7, padding=3)
#         self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)
#         self.conv3 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=5, padding=2)
#         self.pool3 = nn.MaxPool1d(kernel_size=2, stride=2)
#         # decode process
#         self.gru = nn.GRU(input_size=64, hidden_size=64, batch_first=True)
#         self.gru_drop = nn.Dropout(p=0.5)
#         # self.aap = nn.AdaptiveAvgPool1d(1)
#         self.blend3 = bn_relu_conv(64, 64, kernel_size=3)
#         self.blend2 = bn_relu_conv(64, 4, kernel_size=3)
#         self.blend1 = bn_relu_conv(4, 1, kernel_size=3)
#         # regression head
#         # general functions
#         self.sigmoid = nn.Sigmoid()
#         self.relu = nn.ReLU(inplace=True)
#         self.dropout = nn.Dropout(p=0.2)
#         self._init_weights()
#
#     def _init_weights(self):
#         """Initialize the new built layers"""
#         for layer in self.modules():
#             if isinstance(layer, (nn.Conv1d, nn.Linear)):
#                 # nn.init.kaiming_uniform_(layer.weight, mode='fan_in', nonlinearity='relu')
#                 nn.init.xavier_uniform_(layer.weight)
#                 if layer.bias is not None:
#                     nn.init.constant_(layer.bias, 0)
#             elif isinstance(layer, nn.BatchNorm1d):
#                 nn.init.constant_(layer.weight, 1)
#                 nn.init.constant_(layer.bias, 0)
#
#     def forward(self, data):
#         """Construct a new computation graph at each froward"""
#         b, _, _ = data.size()
#         # encode process
#         skip1 = data
#         out1 = self.conv1(data)
#         out1 = self.relu(out1)
#         out1 = self.pool1(out1)
#         out1 = self.dropout(out1)
#         skip2 = out1
#         out1 = self.conv2(out1)
#         out1 = self.relu(out1)
#         out1 = self.pool2(out1)
#         out1 = self.dropout(out1)
#         skip3 = out1
#         out1 = self.conv3(out1)
#         out1 = self.relu(out1)
#         out1 = self.pool3(out1)
#         out1 = self.dropout(out1)
#
#         skip4 = out1.permute(0, 2, 1)
#         up4, _ = self.gru(skip4)
#         up4 = self.gru_drop(up4)
#         up4 = up4.permute(0, 2, 1)
#
#         # decode process
#
#         up3 = upsample(up4, skip3.size()[-1])
#         up3 = up3 + skip3
#         up3 = self.blend3(up3)
#         up2 = upsample(up3, skip2.size()[-1])
#         up2 = up2 + skip2
#         up2 = self.blend2(up2)
#         up1 = upsample(up2, skip1.size()[-1])
#         up1 = up1 + skip1
#         up1 = self.blend1(up1)
#         out_dense = self.sigmoid(up1)
#         out_dense = out_dense.view(b, -1)
#
#         # regression
#
#
#         return  out_dense
