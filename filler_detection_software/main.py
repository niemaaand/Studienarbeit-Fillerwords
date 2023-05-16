from Temporal_Convolution_Resnet.options import OptionsInformation
from evaluation.continuous_speech import predictions
from utils import converter
from evaluation.continuous_speech.predictions import ScoreType
from filler_detection_software import visualiziation
from evaluation.continuous_speech.options_for_continuous_speech import OptionsForContinuousSpeech


def start(opt: OptionsInformation, wav_path: str, resolution_factor: int = 10, show_in_gui: bool = False):

    predictor = predictions.Predictor(opt, wav_path, resolution_factor)
    predictor.predict_fillers()
    filler_occurrences = predictor.predict_fillers_per_hop(ScoreType.THREE)

    if isinstance(opt, list):
        opt = opt[0]

    filler_occurrences = converter.convert_hops_to_filler_occurrences(filler_occurrences, opt.clip_duration / (resolution_factor * opt.clip_duration))

    if show_in_gui:
        v = visualiziation.Visualizer(filler_occurrences, wav_path)
        v.visualize_filler_occurrences()
    else:
        for filler_occ in filler_occurrences:
            print(filler_occ)


opt_cont = OptionsForContinuousSpeech("../pretrained_models/chapter_6_5", "../example_data/CEO Interview 2017_Pat Finn.wav")
start(opt_cont.options, opt_cont.wav_path, show_in_gui=True)










