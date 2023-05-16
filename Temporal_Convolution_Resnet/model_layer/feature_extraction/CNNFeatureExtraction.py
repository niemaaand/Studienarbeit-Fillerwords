import torch.nn as nn
from einops import rearrange

from Temporal_Convolution_Resnet.model_layer.blocks.DSCNN import DepthwiseSeparable


class CNNFeatureExtraction(nn.Module):
    def __init__(self, out_channels, kernel_size=[(1, 64)], stride=[8], padding=(0, 0), pool_size=[(1,2)]):
        super(CNNFeatureExtraction, self).__init__()
        models = []
        self.conv1 = DepthwiseSeparable(in_channels=1, out_channels=out_channels[0], kernel_size_depth=(1, kernel_size[0]), stride=(1,stride[0]))
        self.pool1 = nn.MaxPool2d(kernel_size=(1,pool_size[0]))

        for i in range(len(kernel_size) - 1):
            models.append(DepthwiseSeparable(in_channels=out_channels[i], out_channels=out_channels[i+1], kernel_size_depth=(1,kernel_size[i+1]), stride=(1, stride[i+1])))
            models.append(nn.MaxPool2d(kernel_size=(1,pool_size[i+1])))

        self.models_list = nn.ModuleList(models)

    def forward(self, wav):
        out = wav.unsqueeze(1)

        out = self.conv1(out)
        out = self.pool1(out)
        for m in self.models_list:
            out = m(out)

        B, C, H, W = out.shape
        out = rearrange(out, "b c f t -> b f c t", c=C, f=H)
        return out

