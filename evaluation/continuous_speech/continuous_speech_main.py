from paths_for_running_in_colab import PathsForRunningInColabSingleton
from datapreprocessing.visualize_fillers_from_master_csv import get_corrected_labels_for_episode
from evaluation.continuous_speech import predictions, metric
from evaluation.continuous_speech.predictions import ScoreType
from evaluation.continuous_speech.options_for_continuous_speech import OptionsForContinuousSpeech

corrected_labels_path = "../../example_data/CEO Interview 2017_Pat Finn.txt"


def get_predictions_for_scores(opt, wav_path, corrected_labels_path=None):
    """
    This function can be used to get scores (f1, recall, precision).
    Use this function by setting breakpoint in line with "predictor.predict_fillers_per_hop(...)".
    The prediction itself (done before this line) takes some time, but the evaluation and calculation of metrics are
    fast.
    So if you want to have multiple metrics, as soon as you reach this breakpoint, you can build/calculate new metrics.
    :return:
    """

    paths_for_running_in_colab = PathsForRunningInColabSingleton.get_instance()

    fillers = get_corrected_labels_for_episode(paths_for_running_in_colab.master_csvfile,
                                               wav_path,
                                               corrected_labels_path)

    predictor = predictions.Predictor(opt, wav_path, 10)
    predictor.predict_fillers()
    filler_occurrences = predictor.predict_fillers_per_hop(ScoreType.THREE)
    m2 = metric.Metrics2(filler_occurrences, fillers, 1.2, predictor.hop_size / predictor.sr)
    f1 = m2.f1()


opt_cont = OptionsForContinuousSpeech("../../pretrained_models/chapter_6_5", "../../example_data/CEO Interview 2017_Pat Finn.wav")
get_predictions_for_scores(opt_cont.options, opt_cont.wav_path, corrected_labels_path)
