import copy
import json
import os


class SincOptions:
    def __init__(self, cnn_N_filt=[64, 129],
                 cnn_len_filt=[128, 64],
                 cnn_max_pool_len=[16, 8],
                 cnn_stride = [4,4],
                 cnn_act=["leaky_relu", "leaky_relu", "leaky_relu"],
                 cnn_use_laynorm_inp=True,
                 cnn_use_batchnorm_inp=False,
                 cnn_use_laynorm=True,
                 cnn_use_batchnorm=False
                 ):
        self.cnn_N_filt = cnn_N_filt
        self.cnn_len_filt = cnn_len_filt
        self.cnn_max_pool_len = cnn_max_pool_len
        self.cnn_stride = cnn_stride
        self.cnn_act = cnn_act
        self.cnn_use_laynorm_inp = cnn_use_laynorm_inp
        self.cnn_use_batchnorm_inp = cnn_use_batchnorm_inp
        self.cnn_use_laynorm = cnn_use_laynorm
        self.cnn_use_batchnorm = cnn_use_batchnorm


class OptionsInformation:
    def __init__(self, feature_extraction="nnAudio_stft",
                 freq_scale="mel", trainable_feature_extraction: bool = True,
                 filter_length=256,
                 hop_length=129,

                 bins=129,
                 sinc_options=None,

                 # backbone model
                 n_model=1,
                 n_classifiers=1,

                 # resnet
                 __channels__=[16, 24, 32, 48],
                 channels=[16, 24, 32, 48],
                 model="dsmix_tcresnet_pool",
                 channel_scale=4,
                 p_dropout=0.2,
                 last_res_pool=1,
                 p_dropout_in_res=0.0,
                 leaky_relu=False,

                 # classification
                 classifier="fnn",
                 num_classes=2,
                 classifier_sizes=None,
                 classifier_activations=["none", "none"],
                 classifier_residual=[True, False],

                 # general
                 sampling_rate=16000,
                 clip_duration=2,  # seconds
                 use_opt=True,
                 batch: int = 128,
                 epoch=200,
                 step=30,  # number of epochs before adapt learning rate (per scheduler)
                 save_freq=5,
                 lr=0.05,
                 optim="sgd",
                 scheduler_lr_gamma=0.1,
                 lr_momentum=0.9,
                 reload_model: str = None,
                 save="",
                 k_fold: int = 1,
                 train_data_size: int = None,
                 retraining = False
                 ):
        # feature extraction
        self.feature_extraction = feature_extraction
        self.freq_scale: str = freq_scale
        self.trainable_feature_extraction: bool = trainable_feature_extraction
        self.filter_length = filter_length
        self.hop_length = hop_length
        self.bins = bins
        self.sinc_options = sinc_options
        if sinc_options:
            self.sinc_options = SincOptions(**sinc_options)

        # backbone model
        self.n_model = n_model
        self.n_classifiers = n_classifiers

        # resnet
        self.__channels__ = __channels__
        self.model = model
        self.channel_scale = channel_scale
        self.channels = copy.deepcopy(self.__channels__)
        self.p_dropout = p_dropout
        self.last_res_pool = last_res_pool
        self.p_dropout_in_res = p_dropout_in_res
        self.leaky_relu = leaky_relu

        # classification
        self.classifier = classifier
        self.num_classes = num_classes
        self.classifier_sizes = classifier_sizes
        self.classifier_activations = classifier_activations
        self.classifier_residual = classifier_residual

        # general
        self.sampling_rate = sampling_rate
        self.clip_duration = clip_duration  # seconds
        self.use_opt = use_opt
        self.batch: int = batch
        self.epoch = epoch
        self.step = step  # number of epochs before adapt learning rate (per scheduler)
        self.save_freq = save_freq
        self.lr = lr
        self.optim = optim
        self.scheduler_lr_gamma = scheduler_lr_gamma
        self.lr_momentum = lr_momentum
        self.reload_model: str = reload_model
        self.save: str = save
        self.k_fold: int = k_fold
        self.train_data_size: int = train_data_size
        self.retraining = retraining

    def scale_channels(self):
        self.channels = [int(cha * self.channel_scale) for cha in self.__channels__]

    def set_save(self):
        self.save = "{}_{}".format(self.feature_extraction, self.model)

    def set_classifier_sizes(self):
        self.classifier_sizes = [self.channels[-1] * self.last_res_pool, self.channels[-1] * self.last_res_pool,
                                 self.num_classes]

    def set_everything(self):
        self.scale_channels()
        self.set_save()
        self.set_classifier_sizes()

    def write_options_to_json(self, path: str):
        dict_opt = vars(self)
        with open(path, "w") as f:
            json.dump(dict_opt, f, default=lambda o: o.__dict__, indent=4)


def read_options_from_json(path: str) -> OptionsInformation:
    file = open(path, "r")
    dict_options = json.load(file)
    json_s = json.dumps(dict_options)
    o = OptionsInformation(**json.loads(json_s))
    return o
