import torch
import torch.nn as nn
import numpy as np
import nnAudio
from nnAudio import features as nnASpec
import librosa


class STFTnnAudio(nn.Module):
    def __init__(self, filter_length, hop_length, bins, num_classes,
                 trainable_stft=False, freq_scale="no"):
        super(STFTnnAudio, self).__init__()

        self.sampling_rate = 16000
        self.filter_length = filter_length
        self.hop_length = hop_length
        self.bins = bins
        self.num_classes = num_classes

        intern_freq_scale = None

        if not freq_scale in ["no", "linear", "log", "log2"]:
            intern_freq_scale = freq_scale
            freq_scale = "no"

        self.stft_layer_nnAudio = nnASpec.STFT(n_fft=self.filter_length, win_length=self.filter_length,
                                               freq_bins=self.bins, hop_length=self.hop_length, window="hann",
                                               sr=self.sampling_rate, output_format="Magnitude",
                                               trainable=trainable_stft,
                                               freq_scale=freq_scale)

        if intern_freq_scale:
            if intern_freq_scale == "mel":
                # scale to mel-frequency
                create_fourier_kernel_scaled_to_bins2freq(
                    librosa.mel_frequencies(n_mels=self.bins, fmin=20.0, fmax=8000).tolist(),
                    self.stft_layer_nnAudio, self.sampling_rate)

    def forward(self, waveform):
        spectro = self.stft_layer_nnAudio(waveform)
        spectro = spectro.unsqueeze(1)
        return spectro


def create_fourier_kernel_scaled_to_bins2freq(bins2freq, layer, sampling_rate):

    s = np.arange(0, layer.n_fft, 1.0)

    layer.binslist = []
    kernel_sin = np.empty((layer.freq_bins, 1, layer.n_fft))
    kernel_cos = np.empty((layer.freq_bins, 1, layer.n_fft))

    for k in range(len(bins2freq)):
        layer.binslist.append(bins2freq[k] * layer.n_fft / sampling_rate)  # self.filter_length = n_fft
        kernel_sin[k, 0, :] = np.sin(2 * np.pi * layer.binslist[k] * s / layer.n_fft)
        kernel_cos[k, 0, :] = np.cos(2 * np.pi * layer.binslist[k] * s / layer.n_fft)

    kernel_sin = kernel_sin.astype(np.float32)
    kernel_cos = kernel_cos.astype(np.float32)

    kernel_sin = torch.tensor(kernel_sin, dtype=torch.float)
    kernel_cos = torch.tensor(kernel_cos, dtype=torch.float)

    wsin = kernel_sin * layer.window_mask.squeeze(-1).unsqueeze(0)
    wcos = kernel_cos * layer.window_mask.squeeze(-1).unsqueeze(0)

    if not layer.trainable:
        layer.register_buffer("wsin", wsin)
        layer.register_buffer("wcos", wcos)
    elif layer.trainable:
        wsin = nn.Parameter(wsin, requires_grad=layer.trainable)
        wcos = nn.Parameter(wcos, requires_grad=layer.trainable)
        layer.register_parameter("wsin", wsin)
        layer.register_parameter("wcos", wcos)

    layer.bins2freq = bins2freq

