import torch.nn as nn

from Temporal_Convolution_Resnet.model_layer.backbone_models.TCResNetBase import TCResNetBase
from Temporal_Convolution_Resnet.model_layer.blocks.ResidualCNN import Residual


class TCResNet(TCResNetBase):
    def __init__(self, bins, n_channels, n_class, p_dropout=0.2, last_res_pool=1):
        super(TCResNet, self).__init__(bins, n_channels, p_dropout)
        """
        Args:
            bin: frequency bin or feature bin
        """

        self.pool2 = nn.AdaptiveAvgPool2d(last_res_pool)
        self.linear1 = nn.Linear(n_channels[-1], n_class)

    def forward(self, inputs):
        out = super(self).forward(inputs)
        out = self.linear1(out)
        return out


class TCResnetWithoutLinearLayer(TCResNetBase):
    def __init__(self, bins, n_channels, p_dropout=0.2):
        super(TCResnetWithoutLinearLayer, self).__init__(bins, n_channels, p_dropout)
        """
            Args:
                bin: frequency bin or feature bin
        """


class TCResnetWithoutLinearLayerPool(TCResnetWithoutLinearLayer):
    def __init__(self, bins, n_channels, p_dropout=0.2, last_res_pool=1):
        super(TCResnetWithoutLinearLayerPool, self).__init__(bins, n_channels, p_dropout)
        self.pool1 = nn.AvgPool2d(kernel_size=(1, 3))
        self.pool2 = nn.AdaptiveAvgPool2d(output_size=(1, last_res_pool))


class TCResnetWithoutLinearLayerPoolDrop(TCResnetWithoutLinearLayerPool):
    def __init__(self, bins, n_channels, p_dropout=0.2, last_res_pool=1, p_dropout_in_res=0.2):
        super(TCResnetWithoutLinearLayerPoolDrop, self).__init__(bins, n_channels, p_dropout, last_res_pool)

        layers = []
        for in_channels, out_channels in zip(n_channels[0:-1], n_channels[1:]):
            layers.append(Residual(in_channels, out_channels))
            layers.append(nn.Dropout2d(p_dropout_in_res / len(n_channels)))
        self.layers = nn.Sequential(*layers)



