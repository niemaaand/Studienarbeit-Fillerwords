

class FillerOccurrence:
    def __init__(self, start_time: float, end_time: float, probability: float):
        self._start_time: float = start_time
        self._end_time: float = end_time
        self._probability: float = probability

    @property
    def start_time(self) -> float:
        return self._start_time

    @start_time.setter
    def start_time(self, value: float):
        self._start_time = value
        if self.start_time > self.end_time:
            self.end_time = self.start_time

    @property
    def end_time(self) -> float:
        return self._end_time

    @end_time.setter
    def end_time(self, value: float):
        self._end_time = value
        if self.end_time < self.start_time:
            self.start_time = self.end_time

    @property
    def probability(self) -> float:
        return self._probability

    @probability.setter
    def probability(self, value: float):
        self._probability = value

    def __str__(self):
        return "Start: {} End: {} Prob: {}".format(self.start_time, self.end_time, self.probability)


def sort_fillers(fillers: list):
    def clips_comparison(c: FillerOccurrence):
        return c.start_time

    _fillers = sorted(fillers, key=clips_comparison)
    return _fillers
