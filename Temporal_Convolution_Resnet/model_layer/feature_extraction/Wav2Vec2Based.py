import torch.nn as nn
import torchaudio


class Wav2Vec2CNNsAndTransformers(nn.Module):
    def __init__(self):
        super(Wav2Vec2CNNsAndTransformers, self).__init__()
        bundle = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H
        self.wav2vec2_model = bundle.get_model()
        self.wav2vec2_model.aux = None

    def forward(self, x):
        out = x.squeeze(1)
        out = self.wav2vec2_model(out)[0]
        out = out.unsqueeze(1)
        return out


class Wav2Vec2CNNs(Wav2Vec2CNNsAndTransformers):
    def __init__(self):
        super(Wav2Vec2CNNs, self).__init__()
        self.wav2vec2_model.encoder = IdentityEncoder()


class IdentityEncoder(nn.Module):
    def __init__(self):
        super(IdentityEncoder, self).__init__()

    def forward(self, x, length):
        return x



