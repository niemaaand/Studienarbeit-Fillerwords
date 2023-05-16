import copy

import torch
import torch.nn as nn
from torchaudio.transforms import MFCC
from typing import Tuple, Iterator

from .classifiers.fully_connected import FNN
from .feature_extraction.STFTnnAudio import STFTnnAudio
from .backbone_models.DSMix_TCResnet import DSMixTCResnet, DSMixTCResnetPool, DSMixTCResnetPoolDrop
from .backbone_models.TCResnet import TCResNet, TCResnetWithoutLinearLayerPoolDrop
from .feature_extraction.Wav2Vec2Based import Wav2Vec2CNNs, Wav2Vec2CNNsAndTransformers
from .feature_extraction.CNNFeatureExtraction import CNNFeatureExtraction
from ..options import OptionsInformation
from Temporal_Convolution_Resnet.model_layer.feature_extraction.SincNet import SincNet


def __use_multiple_models_in_parallel__(opt: OptionsInformation) -> bool:
    return opt.n_model > 1 and (opt.n_model == opt.n_classifiers)


def build_model(opt: OptionsInformation):
    feature_extraction: nn.Module = nn.Sequential()
    backbone_model: nn.Module = nn.Sequential()
    classifier: nn.Module = nn.Sequential()

    if opt.feature_extraction == "nnAudio_stft" or opt.feature_extraction == "stft":
        feature_extraction = STFTnnAudio(opt.filter_length, opt.hop_length, opt.bins, opt.num_classes,
                                         opt.trainable_feature_extraction, opt.freq_scale)
    elif opt.feature_extraction == "mfcc":
        feature_extraction = MFCC(sample_rate=opt.sampling_rate, n_mfcc=opt.bins, log_mels=True)
    elif opt.feature_extraction == "sinc":
        # https://github.com/mravanelli/SincNet/blob/d74164861d8bd8ab9cb02f102d677f5605edfbf4/speaker_id.py
        # https://github.com/mravanelli/SincNet/blob/d74164861d8bd8ab9cb02f102d677f5605edfbf4/cfg/SincNet_TIMIT.cfg

        cnn_arch = {'input_dim': int(opt.sampling_rate * opt.clip_duration),
                    'fs': opt.sampling_rate,
                    'cnn_N_filt': opt.sinc_options.cnn_N_filt,
                    'cnn_len_filt': opt.sinc_options.cnn_len_filt,
                    'cnn_max_pool_len': opt.sinc_options.cnn_max_pool_len,
                    'cnn_use_laynorm_inp': opt.sinc_options.cnn_use_laynorm_inp,
                    'cnn_use_batchnorm_inp': opt.sinc_options.cnn_use_batchnorm_inp,
                    'cnn_use_laynorm': [opt.sinc_options.cnn_use_laynorm] * len(opt.sinc_options.cnn_N_filt),
                    'cnn_use_batchnorm': [opt.sinc_options.cnn_use_batchnorm] * len(opt.sinc_options.cnn_N_filt),
                    'cnn_act': opt.sinc_options.cnn_act,
                    'cnn_drop': [opt.p_dropout] * len(opt.sinc_options.cnn_N_filt),
                    }

        feature_extraction = SincNet(cnn_arch)
    elif opt.feature_extraction == "wav2vec2_CNNs":
        feature_extraction = Wav2Vec2CNNs()
    elif opt.feature_extraction == "wav2vec2_CNNs_andTransformers":
        feature_extraction = Wav2Vec2CNNsAndTransformers()
    elif opt.feature_extraction == "cnn":
        feature_extraction = CNNFeatureExtraction(opt.sinc_options.cnn_N_filt, stride=opt.sinc_options.cnn_stride,
                                                  kernel_size=opt.sinc_options.cnn_len_filt,
                                                  pool_size=opt.sinc_options.cnn_max_pool_len)
    elif opt.feature_extraction == "":
        raise NotImplementedError

    models = []
    classifiers = []

    for i in range(opt.n_model):
        if opt.model == "dsmix_tcresnet":
            backbone_model = DSMixTCResnet(opt.bins, opt.channels, opt.p_dropout)
        elif opt.model == "dsmix_tcresnet_pool":
            backbone_model = DSMixTCResnetPool(opt.bins, opt.channels, opt.p_dropout, opt.last_res_pool)
        elif opt.model == "dsmix_tcresnet_pool_drop":
            backbone_model = DSMixTCResnetPoolDrop(opt.bins, opt.channels, opt.p_dropout, opt.last_res_pool,
                                                   opt.p_dropout_in_res, opt.leaky_relu)
        elif opt.model == "tcresnet":
            backbone_model = TCResNet(opt.bins, opt.channels, opt.num_classes, opt.p_dropout)
        elif opt.model == "tcresnet_withoutLinearLayer_pool_drop":
            backbone_model = TCResnetWithoutLinearLayerPoolDrop(opt.bins, opt.channels, opt.p_dropout,
                                                                opt.last_res_pool, opt.p_dropout_in_res)
        else:
            raise NotImplementedError

        if opt.classifier == "fnn":
            classifier = FNN(opt.classifier_sizes, opt.classifier_activations, opt.classifier_residual)
        elif opt.classifier is None:
            pass
        else:
            raise NotImplementedError

        models.append(copy.deepcopy(backbone_model))
        classifiers.append(copy.deepcopy(classifier))

    if __use_multiple_models_in_parallel__(opt):
        backbone_model = models
        classifier = classifiers

    return Blueprint(feature_extraction, backbone_model, classifier, opt)


def load_model(options: OptionsInformation) -> nn.Module:
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if isinstance(options, list):
        models = []
        for o in options:
            m = build_model(o)
            m.load_state_dict(torch.load(o.reload_model, map_location=torch.device(device)))
            models.append(m)
        return MultipleModels(models)
    else:
        model = build_model(options)
        model.load_state_dict(torch.load(options.reload_model, map_location=torch.device(device)))
        return model


class MultipleModels(nn.Module):
    def __init__(self, models: list):
        super(MultipleModels, self).__init__()
        self.models = nn.ModuleList(models)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, input):
        outs = []
        for m in self.models:
            out = m(input)
            out = self.softmax(out)
            outs.append(out)

        out = sum(outs)
        out = self.softmax(out)
        # for i in range(out.shape[0]):
        #    if out[i][0] > 0.51:
        #        out[i][0] = 1.0
        #        out[i][1] = 0.0
        #    else:
        #        out[i][0] = 0.0
        #        out[i][1] = 1.0
        return out


class Blueprint(nn.Module):
    def __init__(self, feature_extraction: nn.Module, model: nn.Module, classifier: nn.Module,
                 opt: OptionsInformation = None):
        super(Blueprint, self).__init__()
        self.opt = opt
        self.feature_extraction = feature_extraction

        if __use_multiple_models_in_parallel__(opt):
            self.model = nn.ModuleList(model)
            self.classifier = nn.ModuleList(classifier)
            self.softmax = nn.Softmax(dim=1)
        else:
            self.model = model
            self.classifier = classifier

    def forward(self, wav):
        out = self.feature_extraction(wav)

        if __use_multiple_models_in_parallel__(self.opt):
            outs = []
            for m, c in zip(self.model, self.classifier):
                o = m(out)
                o = c(o)
                o = self.softmax(o)
                outs.append(o)

            out = sum(outs)
        else:
            out = self.model(out)
            out = self.classifier(out)

        return out

    def parameters_own(self, recurse: bool = True) -> Iterator[nn.Parameter]:
        for name, param in self.named_parameters_own(recurse=recurse):
            yield param

    def named_parameters_own(self, prefix: str = '', recurse: bool = True) -> Iterator[Tuple[str, nn.Parameter]]:
        gen = self._named_members(
            lambda module: module._parameters.items(),
            prefix=prefix, recurse=recurse)
        for elem in gen:
            if self.opt and not self.opt.trainable_feature_extraction:
                if elem[0].startswith("feature_extraction"):
                    continue
            yield elem
