from torch import nn

from Temporal_Convolution_Resnet.model_layer.blocks.DSCNN import DepthwiseSeparable


class ResidualBase(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(1, 9), padding=(0, 4), leaky_relu=False, ds=False):
        super(ResidualBase, self).__init__()

        relu = nn.ReLU if not leaky_relu else nn.LeakyReLU

        if in_channels != out_channels:
            stride = 2
            self.residual = nn.Sequential(
                nn.Conv2d(
                    in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
                relu())
        else:
            stride = 1
            self.residual = nn.Sequential()

        if ds:
            self.conv1 = DepthwiseSeparable(
                in_channels, out_channels, kernel_size_depth=kernel_size, stride=stride, padding=padding, bias=False)
            self.conv2 = DepthwiseSeparable(
                out_channels, out_channels, kernel_size_depth=kernel_size, stride=1, padding=padding, bias=False)
        else:
            self.conv1 = nn.Conv2d(
                in_channels, out_channels, kernel_size=(1, 9), stride=stride, padding=(0, 4), bias=False)
            self.conv2 = nn.Conv2d(
                out_channels, out_channels, kernel_size=(1, 9), stride=1, padding=(0, 4), bias=False)

        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = relu()

    def forward(self, inputs):
        out = self.conv1(inputs)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        res = self.residual(inputs)
        out = self.relu(out + res)
        return out


class ResidualDSMix(ResidualBase):
    def __init__(self, in_channels, out_channels, kernel_size=(1, 9), padding=(0, 4), leaky_relu=False):
        super(ResidualDSMix, self).__init__(in_channels, out_channels, kernel_size, padding, leaky_relu, ds=True)


class Residual(ResidualBase):
    def __init__(self, in_channels, out_channels, kernel_size=(1, 9), padding=(0, 4), leaky_relu=False):
        super(Residual, self).__init__(in_channels, out_channels, kernel_size, padding, leaky_relu, ds=False)






