import glob
import os

import pandas as pd


class TimesHolder:
    def __init__(self, event_time_mean, event_time_max, event_time_min, fillerevent_time_mean, fillerevent_time_max,
                 fillerevent_time_min):
        self.event_time_mean = event_time_mean
        self.event_time_max = event_time_max
        self.event_time_min = event_time_min
        self.fillerevent_time_mean = fillerevent_time_mean
        self.fillerevent_time_max = fillerevent_time_max
        self.fillerevent_time_min = fillerevent_time_min

    def __str__(self):
        return "Any event: mean: {} min: {} max: {} Filler: mean: {} min: {} max: {}".format(
            self.event_time_mean, self.event_time_min, self.event_time_max,
            self.fillerevent_time_mean, self.fillerevent_time_min, self.fillerevent_time_max)


def calc_mean_event_time(master_csv_path: str, clip_wav_path: str) -> TimesHolder:
    event_df = pd.read_csv(master_csv_path)
    avail_clips = glob.glob(os.path.join(clip_wav_path, "*", "**/*.wav"))

    if len(avail_clips) == 0:
        raise FileNotFoundError

    event_time_total: float = 0
    events_total: int = 0

    fillerevent_time_total: float = 0
    fillerevents_total: int = 0

    times_holder: TimesHolder = TimesHolder(0.0, 0.0, None,
                                            0.0, 0.0, None)

    for clip in avail_clips:
        pfID = get_wav_id_from_file_name(clip)

        event = event_df.loc[pfID]
        if event["pfID"] != pfID:
            raise ValueError

        t = event["event_end_inepisode"] - event["event_start_inepisode"]

        if t > times_holder.event_time_max:
            times_holder.event_time_max = t

        if (not times_holder.event_time_min) or (t < times_holder.event_time_min):
            times_holder.event_time_min = t

        if event["label_consolidated_vocab"] == "Uh" or event["label_consolidated_vocab"] == "Um":
            if t > times_holder.fillerevent_time_max:
                times_holder.fillerevent_time_max = t

            if (not times_holder.fillerevent_time_min) or (t < times_holder.fillerevent_time_min):
                times_holder.fillerevent_time_min = t

            fillerevent_time_total += t
            fillerevents_total += 1

        event_time_total += t
        events_total += 1

    times_holder.event_time_mean = event_time_total / events_total
    times_holder.fillerevent_time_mean = fillerevent_time_total / fillerevents_total
    print(times_holder)
    return times_holder


def get_wav_id_from_file_name(wav_path: str) -> int:
    return int(os.path.splitext(os.path.basename(wav_path))[0])
