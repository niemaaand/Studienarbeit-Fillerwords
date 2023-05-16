from utils import filler_occurence, converter
from utils.filler_occurence import FillerOccurrence


class MetricsBase:
    def __init__(self):
        self._true_positive: int = 0
        self._false_positive: int = 0
        self._false_negative: int = 0

    def print_information(self):
        print("Total: {}".format(len(self._predicted)))
        print("True Positive: {}".format(self._true_positive))
        print("False Positive: {}".format(self._false_positive))
        print("False Negative: {}".format(self._false_negative))

    def recall(self):
        return self._true_positive / (self._true_positive + self._false_negative)

    def precision(self):
        return self._true_positive / (self._true_positive + self._false_positive)

    def f1(self):
        return 2 * (self.precision() * self.recall()) / (self.precision() + self.recall())


class Metrics(MetricsBase):
    """
    Metric based on overlapping of prediction and truth.
    IMPLEMENTATION NOT WORKING PROPERLY
    """
    def __init__(self, predicted_fillers: list[FillerOccurrence], actual_fillers: list[FillerOccurrence],
                 offset_shorter=0.3, offset_longer=0.6, window_size=2.0):
        super(Metrics, self).__init__()
        self._predicted = filler_occurence.sort_fillers(predicted_fillers)
        self._actual = filler_occurence.sort_fillers(actual_fillers)
        self._offset_shorter = offset_shorter
        self._offset_longer = offset_longer

        for i in range(len(self._predicted)):
            self._predicted[i] = (self._predicted[i], False)

        for i in range(len(self._actual)):
            self._actual[i] = (self._actual[i], False)

        for i in range(len(self._predicted)):

            predicted_filler: FillerOccurrence = self._predicted[i][0]

            for j in range(len(self._actual)):

                actual_filler: FillerOccurrence = self._actual[j][0]

                start_approved = False
                end_approved = False
                if actual_filler.start_time - self._offset_longer < predicted_filler.start_time < actual_filler.start_time + self._offset_shorter:
                    start_approved = True
                elif actual_filler.start_time - self._offset_longer > predicted_filler.start_time:
                    break

                if start_approved:
                    k: int = j # set to filler-occurrency which approved start-time
                    go_on = True
                    while go_on:
                        if not (k+1) < len(self._actual):
                            go_on = False
                            break

                        if self._actual[k][0].end_time - self._offset_shorter < predicted_filler.end_time < self._actual[k][0].end_time + self._offset_longer and \
                            not (self._actual[k+1][0].end_time - self._offset_shorter < predicted_filler.end_time):
                            end_approved = True
                            go_on = False
                        elif self._actual[k + 1][0].start_time < self._actual[k][0].end_time + window_size + 2 * self._offset_shorter:
                            k += 1
                        else:
                            break

                    if start_approved and end_approved:
                        self._predicted[i] = (self._predicted[i][0], True)

                        for _k in range(k - j + 1):
                            self._actual[j + _k] = (self._actual[j + _k][0], True)

        true_positive_1 = 0
        true_positive_2 = 0
        for _, correct in self._predicted:
            if correct:
                true_positive_1 += 1
            else:
                self._false_positive += 1

        for _, correct in self._actual:
            if correct:
                true_positive_2 += 1
            else:
                self._false_negative += 1

        self._true_positive = max(true_positive_1, true_positive_2)
        self.print_information()


class Metrics2(MetricsBase):
    """
    Calculate scores per hop.
    """
    def __init__(self, predicted_fillers: list[float], actual_fillers: list[FillerOccurrence], offset=0.6, delta_t=0.1):
        """

        :param predicted_fillers: Probability per hop.
        :param actual_fillers:
        :param offset:
        :param delta_t:
        """
        super(Metrics2, self).__init__()
        self._predicted = predicted_fillers
        self._actual = filler_occurence.sort_fillers(actual_fillers)
        self._offset_longer = offset

        self._truth = converter.convert_filler_occurrences_to_hops(actual_fillers, delta_t, length=len(self._predicted),
                                                                   offset_beginning=self._offset_longer,
                                                                   offset_end=self._offset_longer)

        for truth, predicted in zip(self._truth, self._predicted):
            if truth == -1 and predicted < 0.5:
                # true negative
                pass
            elif truth == -1 and predicted >= 0.5:
                self._false_positive += 1
            elif truth == 1 and predicted < 0.5:
                self._false_negative += 1
            elif truth == 1 and predicted >= 0.5:
                self._true_positive += 1
            elif truth == 0 and predicted >= 0.5:
                self._true_positive += 1
            else:
                pass

        self.print_information()




