# https://github.com/gzhu06/PodcastFillers_Utils/blob/f3f54c5e3001bc77c0701112af63cb358ecd0620/preprocessing_script.py
# Added multicore processing. Use librosa to get duration of audio file. Use PyDub to cut clips.

import glob, os
import json
import math
import random
import shutil
import subprocess
import threading

import librosa
import pandas as pd
import soundfile
import multiprocessing
from pydub import AudioSegment

from pandas import DataFrame

from paths_for_running_in_colab import PathsForRunningInColabSingleton

gone_wrong: [] = []  # Current implementation does work without lock for this variable.

vary_location_in_clip: bool = False
episode_wavs = dict()
event_df: DataFrame # Lock necessary!
event_df_lock = threading.Lock()


def __ffmpeg_convert__(input_audiofile, output_audiofile, sr=16000):
    """
    Convert an audio file to a resampled audio file with the desired
    sampling rate specified by `sr`.
    Parameters
    ----------
    input_audiofile : string
            Path to the video or audio file to be resampled.
    output_audiofile
            Path for saving the resampled audio file. Should have .wav extension.
    sr : int
            The sampling rate to use for resampling (e.g. 16000, 44100, 48000).
    Returns
    -------
    completed_process : subprocess.CompletedProcess
            A process completion object. If completed_process. Returncode is 0 it
            means the process completed successfully. 1 means it failed.
    """

    # fmpeg command
    cmd = ["ffmpeg", "-loglevel", "warning", "-i", input_audiofile, "-ac", "1", "-af", "aresample=resampler=soxr", "-ar", str(sr), "-y", output_audiofile]
    completed_process = subprocess.run(cmd)

    # confirm process completed successfully
    assert completed_process.returncode == 0

    # confirm new file has desired sample rate
    assert soundfile.info(output_audiofile).samplerate == sr


def __reformat__(mp3_folder, full_wav_folder, sr=16000, cores=None):
    """
    convert full-length MP3 files into wav files.
    Parameters
    ----------
    mp3_folder : str
            folder path for full-length podcast episodes in the original MP3 format
    full_wav_folder : str
            folder path for saving full-length podcast episodes in converted WAV format
    sr : int, optional
            The target sampling rate to use for resampling
    """
    audiofiles = glob.glob(os.path.join(mp3_folder, "**/*.mp3"), recursive=True)

    __multi_core_processing__(__reformat_to_wav__, audiofiles, None, full_wav_folder, None, sr, cores=cores)
    #__reformat_to_wav__(audiofiles, None, full_wav_folder, None, sr, None)


def __generate_clip_wav__(master_csvfile, full_wav_folder, clip_folder, duration=1.0, cores=None,
                          vary_loc_in_clip: bool = False, csv_target_name: str = "NewMasterCSV.csv") -> list[dict]:
    """
    generate clips files from full podcast episodes for training
    filler classifier
    Args:
            master_csvfile (str): master csv filepath
            full_wav_folder (str): folder path for full length podcast episode in converted wav format
            clip_folder (str): folder path for event wav clips
            duration (float): amount of time to increase or decrease over the original one second clip
    """
    global vary_location_in_clip, event_df, gone_wrong, episode_wavs
    vary_location_in_clip = vary_loc_in_clip

    gone_wrong = []
    gone_wrong.append(dict())

    # use caching of audio files in global variable
    episode_wavs = dict()
    for episode in glob.glob(os.path.join(full_wav_folder, "**/*.wav")):
        episode_wavs[episode] = [AudioSegment.from_wav(episode), librosa.get_duration(path=episode)]

    event_df = pd.read_csv(master_csvfile)
    event_list = []
    for i, event in event_df.iterrows():
        event_list.append(event)

    __multi_core_processing__(__cut_clip_wav__, event_list, clip_folder, full_wav_folder, duration, None, cores=cores)
    #__cut_clip_wav__(event_list, clip_folder, full_wav_folder, duration, None, 0)

    event_df_lock.acquire()
    paths_for_running_in_colab = PathsForRunningInColabSingleton.get_instance()

    event_df.to_csv(os.path.join(clip_folder, csv_target_name), index=False)

    new_master_csv_meta_path = os.path.join(paths_for_running_in_colab.metadata_folder, csv_target_name)
    shutil.copyfile(os.path.join(clip_folder, csv_target_name), new_master_csv_meta_path)

    paths_for_running_in_colab.master_csv_name = csv_target_name
    paths_for_running_in_colab.master_csvfile = new_master_csv_meta_path

    event_df_lock.release()

    return gone_wrong


def __multi_core_processing__(target, data: list, clip_folder: str, full_wav_folder: str, duration: float, sr: int, cores = None):

    global gone_wrong

    if not cores:
        cores = multiprocessing.cpu_count() - 1

    event_lists = []
    threads = []
    for i in range(0, cores):
        event_lists.append([])
        gone_wrong.append(dict())

    i = 0
    for event in data:
        event_lists[i % len(event_lists)].append(event)
        i += 1

    for i in range(0, cores):
        threads.append(threading.Thread(target=target, args=(event_lists[i], clip_folder, full_wav_folder, duration, sr, i)))
        threads[i].start()

    for i in range(0, cores):
        threads[i].join()


def __cut_clip_wav__(data: list, clip_folder: str, full_wav_folder: str, duration: float, sr: int, idx = 0):
    global vary_location_in_clip, event_df, gone_wrong, episode_wavs

    paths_for_running_in_colab = PathsForRunningInColabSingleton.get_instance()

    for event in data:

        episode_subset = event["episode_split_subset"]
        clip_subset = event["clip_split_subset"]

        if clip_subset == "extra":
            continue

        tar_folder: str
        if paths_for_running_in_colab.n_subfolders != 0:
            tar_folder = os.path.join(clip_folder, clip_subset, str(event["pfID"] % paths_for_running_in_colab.n_subfolders))
        else:
            tar_folder = os.path.join(clip_folder, clip_subset)

        os.makedirs(tar_folder, exist_ok=True)

        src_filepath = os.path.join(
            full_wav_folder, episode_subset, str(event["podcast_filename"]) + ".wav"
        )
        start_time_clip = event["clip_start_inepisode"]
        end_time_clip = event["clip_end_inepisode"]

        tar_filepath = os.path.join(tar_folder, event["clip_name"])

        if os.path.exists(tar_filepath):
            continue

        # cut wav into clips based on filler metainfo
        act_start_time: int # in ms
        act_end_time: int # in ms

        if vary_location_in_clip:
            r: float = 0.1
            start_time_event = event["event_start_inepisode"] - r
            end_time_event = event["event_end_inepisode"] + r

            missing_to_duration = duration - (end_time_event - start_time_event)
            if not missing_to_duration == -0.2:
                # Take random element into time span.
                r = round(random.uniform(0, missing_to_duration), 3)
            else:
                # Keep original time span.
                r = -0.1

            act_start_time = int((start_time_event - r) * 1000)
        else:
            if end_time_clip - start_time_clip < duration:
                duration_offset = (duration - 1.0) / 2.0
                act_start_time = int((start_time_clip - duration_offset) * 1000)
            else:
                act_start_time = int(start_time_clip * 1000)

        if act_start_time < 0:
            act_start_time = 0

        act_end_time = int(act_start_time + duration * 1000)

        new_clip, episode_duration = episode_wavs[src_filepath]
        if act_end_time > episode_duration and \
                math.isclose(act_end_time, episode_duration, abs_tol=duration + 1e-5):
            # abs_tol=duration because it might happen, that event is timed after end of episode
            act_end_time = math.floor(episode_duration * 1000)
            act_start_time = int(act_end_time - duration * 1000)
        # else: clip cannot be extracted, so it will be listed in faults.json

        sth_went_wrong: bool = False
        try:
            if act_end_time > len(new_clip):
                sth_went_wrong = True
            else:
                new_clip = new_clip[act_start_time:act_end_time]
                new_clip.export(tar_filepath, format="wav")  # Exports to a wav file in the current path.
        except:
            sth_went_wrong = True

        # confirm process completed successfully
        actual_duration: float = 0
        try:
            if not sth_went_wrong:
                actual_duration = librosa.get_duration(path=tar_filepath)
        except FileNotFoundError:
            sth_went_wrong = True

        if actual_duration != duration or sth_went_wrong:  # or completed_process.returncode != 0: (with use of ffmpeg)
            gone_wrong[idx][tar_filepath] = src_filepath

        event_df_lock.acquire()
        event_df.loc[event["pfID"], "clip_start_inepisode"] = act_start_time / 1000
        event_df.loc[event["pfID"], "clip_end_inepisode"] = act_end_time / 1000
        event_df_lock.release()


def __reformat_to_wav__(data: list, clip_folder: str, full_wav_folder: str, duration: float, sr: int, idx = 0):
    for audiofile in data:

        folder_path = os.path.join(full_wav_folder, os.path.basename(os.path.dirname(audiofile)))
        os.makedirs(folder_path, exist_ok=True)
        opt_audiofile = os.path.join(folder_path, (os.path.splitext(os.path.basename(audiofile)))[0] + ".wav")

        if os.path.exists(opt_audiofile):
            continue

        __ffmpeg_convert__(audiofile, opt_audiofile, sr)


def cut_clips(dataset_path: str, vary_loc_in_clip=True, duration=2.0):
    """
    Proposed (by Authors of PodcastFillers) preprocessing of PodcastFillers dataset: converting wav to mp3 and cutting clips from wav.
    :param duration:
    :param vary_loc_in_clip:
    :param dataset_path:
    :return:
    """
    paths_for_running_in_colab = PathsForRunningInColabSingleton.get_instance()

    master_csvfile = paths_for_running_in_colab.master_csvfile
    full_mp3_folder = os.path.join(dataset_path, "audio", "episode_mp3")
    full_wav_folder = os.path.join(dataset_path, "audio", "episode_wav_regenerate")
    clip_folder = os.path.join(dataset_path, "audio", "clip_wav_regenerate_2s")

    sample_rate = 16000

    # convert full-length MP3 into WAV
    __reformat__(full_mp3_folder, full_wav_folder, sr=sample_rate)
    print("mp3 to wav done")

    # generate clip wavs from full length WAV
    # it's required to run "reformat" stage first, otherwise path-creation in module has to be adapted
    gone_wrong = __generate_clip_wav__(master_csvfile, full_wav_folder, clip_folder, duration=duration,
                                       vary_loc_in_clip=vary_loc_in_clip)
    if len(gone_wrong) > 0:
        n_gone_wrong = 0
        for g in gone_wrong:
            n_gone_wrong += len(g)
        print("Gone wrong creating clip: " + str(n_gone_wrong))

        with open(os.path.join(PathsForRunningInColabSingleton.get_instance().clip_folder_path, "goneWrongCreatingClip.json"), "w", encoding="UTF-8") as f:
            json.dump(gone_wrong, f, indent=4, ensure_ascii=False)

    print("clip-extraction done")
    print("")
