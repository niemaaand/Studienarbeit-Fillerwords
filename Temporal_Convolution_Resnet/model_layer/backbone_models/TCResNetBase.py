import torch
from einops import rearrange
from torch import nn as nn

from Temporal_Convolution_Resnet.model_layer.blocks.DSCNN import DepthwiseSeparable
from Temporal_Convolution_Resnet.model_layer.blocks.ResidualCNN import Residual, ResidualDSMix


class TCResNetBase(nn.Module):
    def __init__(self, bins, n_channels, p_dropout=0.2, leaky_relu=False, ds=False):
        """
                    Args:
                        bin: frequency bin or feature bin
                """
        super(TCResNetBase, self).__init__()

        if not ds:
            self.conv = nn.Conv2d(
                bins, n_channels[0], kernel_size=(1, 3), padding=(0, 1), bias=False)
        else:
            self.conv = DepthwiseSeparable(
                bins, n_channels[0], kernel_size_depth=(1, 3), padding=(0, 1), bias=False)

        layers = []
        for in_channels, out_channels in zip(n_channels[0:-1], n_channels[1:]):
            if not ds:
                layers.append(Residual(in_channels, out_channels))
            else:
                layers.append(ResidualDSMix(in_channels, out_channels, leaky_relu=leaky_relu))

        self.layers = nn.Sequential(*layers)

        self.pool1 = nn.AdaptiveAvgPool2d((None, None))
        self.drop1 = nn.Dropout2d(p_dropout)

        if not ds:
            self.conv2 = nn.Conv2d(n_channels[-1], n_channels[-1], (1, 2))
        else:
            self.conv2 = DepthwiseSeparable(n_channels[-1], n_channels[-1], (1, 2))

        self.pool2 = nn.AdaptiveAvgPool2d(1)

    def forward(self, inputs):
        """
        Args:
            :param inputs: [B, 1, H, W] ~ [B, 1, freq, time]
            reshape -> [B, freq, 1, time]
        """
        B, C, H, W = inputs.shape
        inputs = rearrange(inputs, "b c f t -> b f c t", c=C, f=H)
        out = self.conv(inputs)
        out = self.layers(out)
        out = self.pool1(out)
        out = self.drop1(out)
        out = self.conv2(out)

        out = self.pool2(out)
        out = torch.reshape(out, (out.shape[0], -1))
        return out
