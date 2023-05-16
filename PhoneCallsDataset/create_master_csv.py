import csv
import glob, os
import random

import pandas as pd
from pandas import DataFrame
from pydub import AudioSegment

from Temporal_Convolution_Resnet.model_dataloader.FillersDatasets import filler_categories, non_filler_categories


def create_master_csv(full_wav_path: str, transcrib_path: str, master_csv: str):
    clips: list(ClipInformationHolder) = list()
    rand = random.Random()

    trans_cha = glob.glob(os.path.join(transcrib_path, "*.cha"))
    full_wav_files = glob.glob(os.path.join(full_wav_path, "**/*.wav"))
    avail_full_wav_files = dict()

    for w in full_wav_files:
        avail_full_wav_files[os.path.splitext(os.path.basename(w))[0]] = w

    for cha in trans_cha:
        with open(cha, "r") as f:

            current_clips: list(ClipInformationHolder) = list()
            podcast_path = avail_full_wav_files[os.path.splitext(os.path.basename(cha))[0]]
            max_duration = AudioSegment.from_wav(podcast_path).duration_seconds
            podcast_filename: str = str(os.path.splitext(os.path.basename(podcast_path))[0])
            subset: str = os.path.basename(os.path.dirname(podcast_path))
            lines = f.readlines()

            # get filler clips
            i: int = 0
            while i < len(lines):

                line = lines[i]
                if line.startswith("*"):  # or line.startswith("\t"):

                    filler_in_line, clip_start_in_episode, clip_end_in_episode, i = __transcribed_filler_in_segment__(
                        lines, i, max_duration)
                    if filler_in_line:
                        # save clip

                        event_start_in_episode = clip_start_in_episode - 0.5
                        if event_start_in_episode < 0:
                            event_start_in_episode = 0

                        event_end_in_episode = clip_start_in_episode + 2

                        current_clips.append(
                            ClipInformationHolder(len(clips) + len(current_clips), "filler", podcast_filename,
                                                  clip_start_in_episode, clip_end_in_episode, event_start_in_episode,
                                                  event_end_in_episode, subset))

                i += 1

            # get non-filler clips
            cnt: int = 0
            total_trys: int = 0
            n_fillers_in_file = len(current_clips)
            while cnt <= n_fillers_in_file + 1:  # get some more non-filler clips (for example if another file
                # is too short for creating non-fillers)

                r = round(rand.uniform(0, max_duration - 1.5), 3)

                r_is_ok = True
                for c in current_clips:
                    if (c.clip_start_in_episode < r < c.clip_end_in_episode) or \
                            (c.clip_start_in_episode < (r + 1) < c.clip_end_in_episode):
                        # beginning of new clip during existing clip
                        # ending of new clip during existing clip
                        r_is_ok = False
                        break

                if r_is_ok:
                    current_clips.append(
                        ClipInformationHolder(len(clips) + len(current_clips), "non_filler", podcast_filename,
                                              r, r + 1, r - 0.5, r + 1.5, subset))
                    cnt += 1

                if total_trys > 5 * n_fillers_in_file + 2:  # some buffer / files with no filler-clips
                    # emergency exit, for example if wav file is too short to create enough samples
                    break

                total_trys += 1

            # add current clips of current file to global clips
            for c in current_clips:
                clips.append(c)

            __write_csv__(clips, master_csv)


def __write_csv__(clips: list, master_csv: str):
    # sort clips

    def clips_comparison(c: ClipInformationHolder):
        return c.pfId

    clips = sorted(clips, key=clips_comparison)

    # write file
    with open(master_csv, "w", newline="", encoding="UTF-8") as csv_file:
        file_writer = csv.writer(csv_file, delimiter=",")
        file_writer.writerow(
            ["clip_name", "pfID", "label_full_vocab", "label_consolidated_vocab", "podcast_filename",
             "event_start_inepisode", "event_end_inepisode",
             "clip_start_inepisode", "clip_end_inepisode", "confidence", "episode_split_subset",
             "clip_split_subset"])

        for c in clips:
            file_writer.writerow(
                ["{}.wav".format(c.pfId), str(c.pfId), c.label, c.label,
                 c.podcast_filename,
                 c.event_start_in_episode, c.event_end_in_episode,
                 c.clip_start_in_episode, c.clip_end_in_episode, "0.85", c.subset, c.subset])


class ClipInformationHolder:
    def __init__(self, pfId: int, label: str, podcast_filename: str, clip_start_in_episode: float,
                 clip_end_in_episode: float,
                 event_start_in_episode: float, event_end_in_episode: float, subset: str):
        self.pfId = pfId
        self.label = label
        self.podcast_filename: str = podcast_filename
        self.clip_start_in_episode = clip_start_in_episode
        self.clip_end_in_episode = clip_end_in_episode
        self.event_start_in_episode = event_start_in_episode
        self.event_end_in_episode = event_end_in_episode
        self.subset = subset


all_fillers = [" uh", "&uh", "%uh", " um", "&um", "%um", " eh", "&eh", "%eh", " er", "&er", "%er", " mm", "&mm", "%mm"
                                                                                                                 " hm",
               "&hm", "%hm"]


def __transcribed_filler_in_segment__(lines: list[str], i: int, max_duration) -> (bool, float, float, int):
    filler_in_line = False
    segment_end = False
    position = 0
    i_start = i

    while not segment_end:
        line = lines[i].replace("uhhuh", "")

        for filler in all_fillers:
            if filler in line:
                position = line.index(filler)
                filler_in_line = True
                break

        if lines[i].startswith("@End") or lines[i].endswith("\x15\n") or lines[i + 1].startswith("@End"):
            segment_end = True
        else:
            i += 1

    if filler_in_line:
        if "\x15" in line:
            start_time, end_time = lines[i].split("\x15")[-2].split("_")
            start_time = int(start_time) / 1000
            end_time = int(end_time) / 1000

            length: int = 0
            for j in range(i_start, i):
                length += len(lines[j])

            length += len(lines[i][:lines[i].index("\x15")])

            while end_time - start_time > 1:

                if position > (length / 2):
                    start_time += (end_time - start_time) / 2
                    position -= length / 2
                else:
                    end_time -= (end_time - start_time) / 2

                length /= 2

            # set to exactly one second
            missing_time = (1 - (end_time - start_time)) / 2
            end_time += missing_time
            start_time -= missing_time

            if start_time < 0:
                start_time = 0
                end_time = 1

            if end_time >= max_duration:
                end_time = max_duration
                start_time = max_duration - 1

            return filler_in_line, start_time, end_time, i
        else:
            # should not occur
            pass
    return False, None, None, i


def __duration_to_two_seconds(start_time: float, end_time: float) -> (float, float):
    act_dur = end_time - start_time
    if not act_dur == 2.0:
        missing_duration = 2.0 - act_dur
        return start_time - (missing_duration / 2), end_time + (missing_duration / 2)
    return start_time, end_time


def __get_clip_information_from_filename__(file_path: str, event_df: DataFrame, recreate: bool = False, not_ok: bool = False) -> ClipInformationHolder:
    file_name = os.path.splitext(os.path.basename(file_path))[0]

    pfId: int
    new_label: str

    if not recreate:
        pfId =  int(file_name.split("_")[0])

        if not_ok:
            if file_name.endswith("_non_filler"):
                new_label = "filler"
            elif file_name.endswith("_filler"):
                new_label = "non_filler"
            else:
                end = file_name.split("_")[-1]
                if end in filler_categories:
                    new_label = "filler"
                elif end in non_filler_categories:
                    new_label = "non_filler"
                else:
                   raise ValueError
        else:
            new_label = event_df.loc[pfId, "label_consolidated_vocab"]
    else:
        pfId = int(file_name)
        new_label = event_df.loc[pfId, "label_consolidated_vocab"]

    event = event_df.loc[pfId]

    if not event["pfID"] == pfId:
        raise ValueError

    clip_start_in_episode = event["clip_start_inepisode"]
    clip_end_in_episode = event["clip_end_inepisode"]

    clip_start_in_episode, clip_end_in_episode = __duration_to_two_seconds(clip_start_in_episode, clip_end_in_episode)

    if "event_start_in_episode" in event:
        event_start_in_episode = event["event_start_in_episode"]
        event_end_in_episode = event["event_end_in_episode"]
        event_start_in_episode, event_end_in_episode = __duration_to_two_seconds(event_start_in_episode,
                                                                                 event_end_in_episode)
    else:
        event_start_in_episode, event_end_in_episode = clip_start_in_episode, clip_end_in_episode

    if event_start_in_episode != clip_start_in_episode or event_end_in_episode != clip_end_in_episode:
        # TODO
        pass

    return ClipInformationHolder(pfId, new_label, event["podcast_filename"],
                                 clip_start_in_episode, clip_end_in_episode,
                                 event_start_in_episode, event_end_in_episode,
                                 event["clip_split_subset"])


def update_master_csv_from_manual_check(master_csv_path: str, nok_path: str, ok_path: str, csv_target_name: str):
    clips: list(ClipInformationHolder) = list()

    event_df = pd.read_csv(master_csv_path)
    nok_files = glob.glob(os.path.join(nok_path, "*.wav"))
    ok_files = glob.glob(os.path.join(ok_path, "*.wav"))

    for nok_f in nok_files:
        clips.append(__get_clip_information_from_filename__(nok_f, event_df, not_ok=True))

    for ok_f in ok_files:
        clips.append(__get_clip_information_from_filename__(ok_f, event_df, not_ok=False))

    __write_csv__(clips, os.path.join(os.path.dirname(master_csv_path), csv_target_name))


def recreate_master_csv_with_two_second_clips():
    """
    Method to recreate master-csv of CallHomeFillers with accurate start/end times.
    This was possible, because no randomness was used to create 2s clips.

    :return:
    """
    # TODO: remove absolute paths?
    clips: list(ClipInformationHolder) = list()
    event_df = pd.read_csv("C:/CallHomeDataset/metadata/CallHomeDataset.csv")
    files = glob.glob(os.path.join("C:/CallHomeDataset/audio/clip_wav_regenerate_2s", "*", "**/*.wav"))
    for f in files:
        clips.append(__get_clip_information_from_filename__(f, event_df, recreate=True))

    __write_csv__(clips, "C:/CallHomeDataset/metadata/CallHomeDataset_2s.csv")

