import datetime
import math
from enum import Enum

import torch
import torch.nn as nn
import torchaudio

from Temporal_Convolution_Resnet.model_layer import ModelBuilder
from Temporal_Convolution_Resnet.options import OptionsInformation
from utils import converter
from utils.filler_occurence import FillerOccurrence, sort_fillers


class ScoreType(Enum):
    """
    None: Every hop, where probability for filler is high enough, is treated as filler candidate. There have to be
    enough filler candidate following each other, so that it is considered an actual filler.
    There have to be (n_hop_samples - n_max_gap_to_next_detected_filler) following filler candidates.

    ONE: Similar to NONE, but with usage of max(average, pool) probabilities.

    TWO: For every hop, the highest probability of two surrounding hops and averaged probability over all windows,
    containing this hop is calculated. Value for hop is the maximum value.

    THREE: Averaged Probability over all windows, containing this hop is calculated. (Described in paper.)
    """
    NONE = 0
    ONE = 1
    TWO = 2
    THREE = 3


class Predictor:
    def __init__(self, options: OptionsInformation, wav_path: str, resolution_factor: int = 4):
        self.options = options
        self.wav_path = wav_path
        self.resolution_factor = resolution_factor

        self.model: nn.Module = ModelBuilder.load_model(self.options)
        self.audio_file, self.sr = torchaudio.load(wav_path)
        self._all_probs = []

        if isinstance(self.options, list):
            self.options = self.options[0]

        if self.sr != self.options.sampling_rate:
            raise ValueError

        self.hop_size = int(self.sr / resolution_factor) # samples
        self._filler_occurrences = []

        self.hops_per_sample = resolution_factor * self.options.clip_duration

    def __align_filler_occurrences__(self, filler_occurrences: list, n_max_gap_to_next_dected_filler=None) -> list[FillerOccurrence]:
        """
        Alignment for ScoreType.NONE.
        :param filler_occurrences:
        :param n_max_gap_to_next_dected_filler:
        :return:
        """
        filler_occurrences = sort_fillers(filler_occurrences)
        filler_occurrences_stacked = []
        i: int = 0

        if not n_max_gap_to_next_dected_filler:
            n_max_gap_to_next_dected_filler = int(math.sqrt(self.hops_per_sample / 2)) + 1

        while i < len(filler_occurrences):

            t_start: float = None
            t_end: float = None

            f = filler_occurrences[i]
            broke = False

            max_gap = 0
            next_filler = False
            p: float = 0
            j: int = 0
            complete_windows_factor = 0
            while max_gap < n_max_gap_to_next_dected_filler and not next_filler:
                idx = i + complete_windows_factor * self.hops_per_sample + j
                if (idx + 1) >= len(filler_occurrences):
                    break

                time_diff = filler_occurrences[idx + 1].start_time - filler_occurrences[idx].start_time
                time_diff_factor = int(round(time_diff / (self.hop_size / self.options.sampling_rate), 0))
                if int(time_diff_factor - 1) > n_max_gap_to_next_dected_filler:
                    next_filler = True
                else:
                    max_gap += int(time_diff_factor - 1)

                if (idx + 1) >= len(filler_occurrences):
                    break
                else:
                    p += filler_occurrences[idx].probability

                    if j >= self.hops_per_sample:
                        complete_windows_factor += 1
                        j = 0
                        max_gap = 0
                    j += 1

            j_complete = j + complete_windows_factor * self.hops_per_sample
            if j_complete > self.hops_per_sample - n_max_gap_to_next_dected_filler - (n_max_gap_to_next_dected_filler - max_gap - 1):
                # Observation showed, that start of filler is recognized more precisely than end of filler.
                # This is why for the start, less buffer (2) is calculated than for the end (3).
                t_start = filler_occurrences[i].end_time - (2 * (self.hop_size / self.options.sampling_rate))
                t_end = filler_occurrences[i + j_complete - 1].start_time + (3 * (self.hop_size / self.options.sampling_rate))

            if t_start and t_end and (t_start < t_end):
                filler_occurrences_stacked.append(FillerOccurrence(t_start, t_end, p / j_complete))

            if max_gap % n_max_gap_to_next_dected_filler > 2 or next_filler:
                i += j_complete
            else:
                i += max(j_complete - 1, 1)

        return filler_occurrences_stacked

    def __average__(self):
        n_hop_samples = self.__calc_n_hop_samples()
        average_score = [0.0] * n_hop_samples
        for i in range(len(self._all_probs)):
            for h in range(self.hops_per_sample):
                average_score[i + h] += self._all_probs[i] / self.hops_per_sample
                # if hops_scores[i + h] < self._all_probs[i]:
                #    hops_scores[i + h] += self._all_probs[i]

        return average_score

    def __max_pool__(self):
        n_hop_samples = self.__calc_n_hop_samples()

        max_pool_score = [0.0] * n_hop_samples
        max_pool_score[0] = max(self._all_probs[0], self._all_probs[1])
        max_pool_score[-1] = max(self._all_probs[-1], self._all_probs[-2])
        for i in range(1, len(self._all_probs) - 1):
            max_pool_score[i] = max(self._all_probs[i - 1], self._all_probs[i], self._all_probs[i + 1])

        return max_pool_score

    def __max_average_pool__(self) -> list[float]:
        average_score = self.__average__()
        max_pool_score = self.__max_pool__()

        n_hop_samples = self.__calc_n_hop_samples()

        total_score = [0.0] * n_hop_samples
        for i in range(n_hop_samples):
            if max_pool_score[i] >= 0.5 or average_score[i] >= 0.5:
                total_score[i] = max(max_pool_score[i], average_score[i])

        return total_score

    def __pure_score_one__(self, probabilities: list):
        filler_occurrences = []
        n_hop_samples = self.__calc_n_hop_samples()

        for i in range(len(probabilities)):
            if probabilities[i] >= 0.5:
                t_start = i * self.hop_size
                t_end = t_start + (self.options.clip_duration * self.sr)

                filler_occurrences.append(FillerOccurrence(t_start / self.sr, t_end / self.sr, self._all_probs[i]))

        fillers_aligned = self.__align_filler_occurrences__(filler_occurrences)
        delta_t = self.options.clip_duration / self.hops_per_sample

        return converter.convert_filler_occurrences_to_hops(fillers_aligned, delta_t, n_hop_samples, 0.0, 0.0, None)

    def __calc_n_hop_samples(self):
        return int(((self.options.clip_duration * self.sr + self.audio_file[0].size()[0]) / self.hop_size) + 1)

    def __calc_prob_per_hop__(self, score: ScoreType, before_factor=0.3, after_factor=0.25):
        """
        Calculates probability per hop. Calculation methodology is selected based on score.
        :param score:
        :param before_factor: needed with score = ScoreType.TWO
        :param after_factor: needed with score = ScoreType.TWO
        :return:
        """
        n_hop_samples = self.__calc_n_hop_samples()

        if not score or score == ScoreType.NONE:
            return self.__pure_score_one__(self._all_probs)
        elif score == ScoreType.ONE:
            total_score = self.__max_average_pool__()
            return self.__pure_score_one__(total_score)
        elif score == ScoreType.TWO:

            total_score = self.__max_average_pool__()

            if before_factor and after_factor:
                i: int = 0
                while i < n_hop_samples:
                    j = i

                    while total_score[j] >= 0.5:
                        j += 1

                    first_positive = i
                    last_positive = j - 1
                    diff = last_positive - first_positive

                    if diff > math.ceil(self.hops_per_sample * (before_factor + after_factor)):
                        for k in range(int(math.ceil(self.hops_per_sample * before_factor))):
                            total_score[first_positive + k] = 0.4  # just a number < 0.5
                        for k in range(int(math.ceil(self.hops_per_sample * after_factor))):
                            total_score[last_positive - k] = 0.4  # just a number < 0.5
                    elif diff > 0:
                        for k in range(diff):
                            total_score[i + k] = 0.4  # just a number < 0.5

                    i = j + 1

            return total_score
        elif score == ScoreType.THREE:
            return self.__average__()


    def __calc_probabilities_per_window(self):
        """
        Result is saved in self._all_probs
        :return:
        """
        self.model.eval()
        with torch.no_grad():
            audio_file = self.audio_file[0]

            # self._filler_occurrences = []

            n_clips = int(audio_file.size()[0] / self.hop_size) - 1
            audio_file = torch.cat([audio_file, torch.zeros(int(self.options.clip_duration * self.sr))])

            softmax = nn.Softmax(dim=1)
            for i in range(n_clips):
                t_start = i * self.hop_size
                t_end = t_start + (self.options.clip_duration * self.sr)
                clip = audio_file[t_start: t_end]
                probs = softmax(self.model(clip.unsqueeze(0).unsqueeze(0)))[0]  # probs[0]: filler, probs[1]: non_filler
                self._all_probs.append(probs[0].item())

    def predict_fillers(self):
        """
        Execute this before predict_fillers_per_hop because here some internal values are set.
        :param score:
        :return:
        """
        t0 = datetime.datetime.now()
        self.__calc_probabilities_per_window()
        print("Duration predicting fillers: {}".format(datetime.datetime.now() - t0))

    def predict_fillers_per_hop(self, score=0) -> list[float]:
        """
        Should only be executed, if predict_fillers has run before.
        :param score:
        :return:
        """
        filler_occurrences_stacked = self.__calc_prob_per_hop__(score)
        return filler_occurrences_stacked


