import torch.nn as nn

from Temporal_Convolution_Resnet.model_layer.backbone_models.TCResNetBase import TCResNetBase
from Temporal_Convolution_Resnet.model_layer.blocks.ResidualCNN import ResidualDSMix


class DSMixTCResnet(TCResNetBase):
    def __init__(self, bins, n_channels, p_dropout=0.2, leaky_relu=False):
       super(DSMixTCResnet, self).__init__(bins, n_channels, p_dropout, leaky_relu, ds=True)


class DSMixTCResnetPool(DSMixTCResnet):
    def __init__(self, bins, n_channels, p_dropout=0.2, last_res_pool=1, leaky_relu=False):
        super(DSMixTCResnetPool, self).__init__(bins, n_channels, p_dropout, leaky_relu=leaky_relu)
        self.pool1 = nn.AvgPool2d(kernel_size=(1,3))
        self.pool2 = nn.AdaptiveAvgPool2d(output_size=(1, last_res_pool))


class DSMixTCResnetPoolDrop(DSMixTCResnetPool):
    def __init__(self, bins, n_channels, p_dropout=0.2, last_res_pool=1, p_dropout_in_res=0.2, leaky_relu=False):
        super(DSMixTCResnetPoolDrop, self).__init__(bins, n_channels, p_dropout, last_res_pool, leaky_relu=leaky_relu)

        layers = []
        for in_channels, out_channels in zip(n_channels[0:-1], n_channels[1:]):
            layers.append(ResidualDSMix(in_channels, out_channels, leaky_relu=leaky_relu))
            layers.append(nn.Dropout2d(p_dropout_in_res))
        self.layers = nn.Sequential(*layers)


