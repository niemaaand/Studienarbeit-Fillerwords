import math

from utils.filler_occurence import FillerOccurrence


def convert_filler_occurrences_to_hops(fillers: list[FillerOccurrence], delta_t: float, length: int,
                                       offset_beginning: float = 0.2, offset_end: float = 0.2, values=[-1, 0, 1]) -> list[float]:
    """

    :param fillers:
    :param delta_t: duration of hop in seconds
    :param length: number of hops
    :param offset_beginning: offset (in seconds) before the start of the filler. Any prediction in this area is always considered correct.
    :param offset_end: offset (in seconds) after the end of the filler. Any prediction in this area is always considered correct.
    :param values: If values=None then probabilities of fillers are used.
    :return:
    """

    if not values or not len(values) == 3:
        values = None

    hops_probs = [(values[0] if values else -1)] * length

    for i in range(len(fillers)):
        n_start = math.floor(fillers[i].start_time / delta_t)
        n_end = math.ceil(fillers[i].end_time / delta_t) + 1

        for j in range(n_start, n_end):
            hops_probs[j] = values[2] if values else fillers[i].probability

        for j in range(n_start - math.ceil(offset_beginning / delta_t), n_start):
            if hops_probs[j] < values[1]:
                hops_probs[j] = values[1] if values else fillers[i].probability

        for j in range(n_end, n_end + math.ceil(offset_end / delta_t)):
            if hops_probs[j] < values[1]:
                hops_probs[j] = values[1] if values else fillers[i].probability

    return hops_probs


def convert_hops_to_filler_occurrences(fillers: list[float], delta_t: float) -> list[FillerOccurrence]:
    filler_occ = []
    i: int = 0
    t_start: float = 0.0
    t_end: float = 0.0
    p: float = 0.0
    while i < (len(fillers)):
        t_start = i * delta_t
        p = 0.0

        j = i
        while fillers[j] >= 0.5:
            p += fillers[i]
            j += 1

        t_end = j * delta_t
        if t_end > t_start:
            filler_occ.append(FillerOccurrence(t_start, t_end, p / (j - i)))

        i = j + 1

    return filler_occ




