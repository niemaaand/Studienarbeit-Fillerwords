import torch.nn as nn


class DepthwiseSeparable(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size_depth, kernel_size_point=1, stride=1, bias=True,
                 padding=0):
        super(DepthwiseSeparable, self).__init__()
        self.depth_conv = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=kernel_size_depth,
                                    groups=in_channels, stride=stride, bias=bias, padding=padding)
        self.point_conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size_point,
                                    stride=1, bias=bias, padding=(0, 0))

    def forward(self, inputs):
        out = self.depth_conv(inputs)
        out = self.point_conv(out)
        return out


