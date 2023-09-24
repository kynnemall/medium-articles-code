#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 24 22:01:01 2023

@author: martin
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def same_move(state, next_state, last_memory):
    condition1 = torch.eq(state, last_memory.state).all()
    condition2 = torch.eq(next_state, last_memory.next_state).all()
    return condition1 and condition2


def encode_state(board):
    board_flat = [0 if e == 0 else int(math.log(e, 2))
                  for e in board.flatten()]
    board_flat = torch.LongTensor(board_flat)
    board_flat = F.one_hot(board_flat, num_classes=16).float().flatten()
    board_flat = board_flat.reshape(1, 4, 4, 16).permute(0, 3, 1, 2)
    return board_flat


class DepthwiseSeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(DepthwiseSeparableConv2d, self).__init__()

        # Depthwise convolution
        self.depthwise = nn.Conv2d(
            in_channels, in_channels, kernel_size, stride=stride, padding=padding, groups=in_channels)

        # Pointwise convolution
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


class ConvBlock(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(ConvBlock, self).__init__()
        d = output_dim // 4
        self.conv1 = DepthwiseSeparableConv2d(input_dim, d, 1, padding='same')
        self.conv2 = DepthwiseSeparableConv2d(input_dim, d, 2, padding='same')
        self.conv3 = DepthwiseSeparableConv2d(input_dim, d, 3, padding='same')
        self.conv4 = DepthwiseSeparableConv2d(input_dim, d, 4, padding='same')

    def forward(self, x):
        output1 = self.conv1(x)
        output2 = self.conv2(x)
        output3 = self.conv3(x)
        output4 = self.conv4(x)
        return torch.cat((output1, output2, output3, output4), dim=1)


class DQN(nn.Module):

    def __init__(self):
        super(DQN, self).__init__()
        self.conv1 = ConvBlock(16, 2048)
        self.conv2 = ConvBlock(2048, 2048)
        self.conv3 = ConvBlock(2048, 2048)
        self.dense1 = nn.Linear(2048 * 16, 1024)
        self.dense6 = nn.Linear(1024, 4)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = nn.Flatten()(x)
        x = F.dropout(self.dense1(x))
        return self.dense6(x)
