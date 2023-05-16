import torch
import torchaudio
from Inspection.ModelInspection import get_model_size


class GreedyCTCDecoder(torch.nn.Module):
    def __init__(self, labels, blank=0):
        super().__init__()
        self.labels = labels
        self.blank = blank

    def forward(self, emission: torch.Tensor) -> str:
        """Given a sequence emission over labels, get the best path string
        Args:
          emission (Tensor): Logit tensors. Shape `[num_seq, num_label]`.

        Returns:
          str: The resulting transcript
        """
        indices = torch.argmax(emission, dim=-1)  # [num_seq,]
        indices = torch.unique_consecutive(indices, dim=-1)
        #indices = [i for i in indices if i != self.blank]
        indices = indices[0]
        return "".join([self.labels[i] for i in indices])


def __get_bundle__():
    return torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H


def print_model_size():
    n_params, size = get_model_size(__get_bundle__().get_model())
    print('Params: ' + str(n_params) + '\nModel size: {:.3f}MB'.format(size))


def execute(waveform, sample_rate) -> str:

    # setting device on GPU if available, else CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    bundle = __get_bundle__()
    model = bundle.get_model().to(device)
    waveform = waveform.to(device)

    emission = model(waveform)
    decoder = GreedyCTCDecoder(bundle.get_labels())
    transcript = decoder(emission[0])
    return transcript
