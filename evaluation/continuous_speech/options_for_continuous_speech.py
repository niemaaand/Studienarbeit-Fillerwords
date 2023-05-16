import os

from Temporal_Convolution_Resnet.options import read_options_from_json


class OptionsForContinuousSpeech:
    def __init__(self, save_path="../../pretrained_models/chapter_6_5",
                 wav_path="../../example_data/CEO Interview 2017_Pat Finn.wav"):
        reload_path = os.path.join(save_path, "01/options.json")
        o1 = read_options_from_json(reload_path)
        o1.reload_model = os.path.join(save_path,
                                       "01/best_dsmix_tcresnet_pool_drop27_tensor(0.8928, device='cuda_0').pt")

        reload_path = os.path.join(save_path, "02/options.json")
        o2 = read_options_from_json(reload_path)
        o2.reload_model = os.path.join(save_path,
                                       "02/best_dsmix_tcresnet_pool_drop41_tensor(0.9055, device='cuda_0').pt")

        self.save_path = save_path
        self.wav_path = wav_path
        self.options = [o1, o2]
