import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, activation='relu'):
        super(ConvBlock, self).__init__()
        self.pad = nn.ReflectionPad2d(padding=kernel_size // 2)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride)
        self.instnorm = nn.InstanceNorm2d(out_channels)
        self.activation = activation
    def forward(self, x):
        if self.activation == 'relu':
            return F.relu(self.instnorm(self.conv(self.pad(x))))
        elif self.activation == 'sigmoid':
            return F.sigmoid(self.instnorm(self.conv(self.pad(x))))
        else:
            return self.instnorm(self.conv(self.pad(x)))


class ResBlock(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(ResBlock, self).__init__()
        self.conv1 = ConvBlock(in_channels, out_channels, 3, 1, 'relu')
        self.conv2 = ConvBlock(out_channels, out_channels, 3, 1, 'linear')

    def forward(self, x):
        identity = x
        x = self.conv2(self.conv1(x))
        x += identity
        return x

class Upsample(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(Upsample, self).__init__()
        self.upsample = nn.UpsamplingNearest2d(scale_factor=2)
        self.conv = ConvBlock(in_channels, out_channels, 3, 1, 'relu')

    def forward(self, x):
        x = self.conv(self.upsample(x))
        return x