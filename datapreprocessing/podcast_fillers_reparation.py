import csv
import datetime
import glob, os
import json
import random
import shutil
import time

import librosa
import pandas as pd
from pandas import DataFrame, Series

from paths_for_running_in_colab import PathsForRunningInColabSingleton


class CSVRepairer:
    def __init__(self, dataset_path, csv_path):
        self.dataset_path = dataset_path
        self.csv_path = csv_path

        self.full_mp3_folder = os.path.join(dataset_path, "audio", "episode_mp3")
        self.event_df: DataFrame = self.load_csv()
        self.audiofiles = glob.glob(os.path.join(self.full_mp3_folder, "**/*.mp3"), recursive=True)

    def load_csv(self) -> DataFrame:
        self.event_df = pd.read_csv(self.csv_path)
        return self.event_df

    def __get_avail_audiofiles(self) -> dict:
        avail_audiofiles = dict()
        for audiofile in self.audiofiles:
            avail_audiofiles[audiofile.split("\\")[-1].split(".mp3")[0]] = audiofile

        return avail_audiofiles

    def adapt_filenames(self, target_path, adapt_csv_path: bool = True) -> int:
        """
        Adapt filenames in master-csv. Before, there were signs like ?, |, \\ used in the master-csv as filenames of
        podcast-episodes. After running this function, these characters, which are not allowed in file-system,
        are replaced with _ .
        :param adapt_csv_path: if true, internal variable is to new csv path.
        :param target_path: path to save new csv-file
        :return: number of repaired + broken files
        """

        avail_audio_files = self.__get_avail_audiofiles()

        broken_files = dict()
        repairable_files = dict()

        for i, event in self.event_df.iterrows():

            name = str(event["podcast_filename"])

            if name in avail_audio_files:
                # good, do nothing
                continue
            elif __repair_name__(name) in avail_audio_files:
                # file can be repaired easily
                if name not in repairable_files:
                    repairable_files[name] = list()
                repairable_files[name].append(i)
                continue
            else:
                # file has to be inspected separately
                if name not in broken_files:
                    broken_files[name] = list()
                broken_files[name].append(i)

        if len(repairable_files) > 0:
            for filename in repairable_files:
                clips = repairable_files[filename]
                for index in clips:
                    self.event_df.loc[index, "podcast_filename"] = __repair_name__(str(event["podcast_filename"]))

            self.event_df.to_csv(target_path, index=False)

        if adapt_csv_path:
            self.csv_path = target_path

        return len(repairable_files) + len(broken_files)

    def find_train_versus_validation_versus_test_inconsistencies(self, target_path: str) -> int:
        """
        All clips generated from the same podcast should be aligned to the same subset. The subset is ruled by episode.
        :param target_path:
        :return:
        """
        self.load_csv()

        inconsistencies: dict = dict()
        avail_audio_files: dict = self.__get_avail_audiofiles()

        # check if episode-subset matches clip-subset
        for i, event in self.event_df.iterrows():
            clip_split_subset = event["clip_split_subset"]
            episode_split_subset = event["episode_split_subset"]

            full_episode_path = avail_audio_files[event['podcast_filename']]
            actual_episode_subset = __get_train_validation_test_type_from_file__(full_episode_path)

            if episode_split_subset != actual_episode_subset or (
                    clip_split_subset != episode_split_subset and clip_split_subset != "extra"):
                # this event has a strange subset-distribution
                inconsistencies[i] = InconsistencyHolder(event, full_episode_path)

        # check if all exceptions match identified pattern
        b = __is_all_train_validation_combo__(inconsistencies)
        if not b:
            # unknown pattern
            print("New additional adjustments needed!!!")

        if len(inconsistencies) > 0:
            # save new file
            for i in inconsistencies:
                incons: InconsistencyHolder = inconsistencies[i]
                self.event_df.loc[i, "clip_split_subset"] = incons.actual_episode_subset
                self.event_df.loc[i, "episode_split_subset"] = incons.actual_episode_subset

            self.event_df.to_csv(target_path, index=False)

        return len(inconsistencies)

    def count_subsets(self) -> dict:
        # TODO: do not count from master-csv but from dataloader-files.
        #  Only clips, that were actually created/creatable should be counted.
        self.load_csv()

        nums = {
            "test": 0,
            "train": 0,
            "validation": 0,
            "extra": 0
        }

        for i, event in self.event_df.iterrows():
            if event["label_consolidated_vocab"] != "None":
                nums[event["clip_split_subset"]] += 1

        return nums


def __is_all_train_validation_combo__(inconsistencies) -> bool:
    """
    Checks if the inconsistencies follow all the same pattern.
    :param inconsistencies:
    :return:
    """
    others = dict()

    for i in inconsistencies:
        incons = inconsistencies[i]
        val_val_train = incons.clip_subset == "validation" and incons.episode_subset == "validation" and incons.actual_episode_subset == "train"
        extra_val_train = incons.clip_subset == "extra" and incons.episode_subset == "validation" and incons.actual_episode_subset == "train"
        if not val_val_train and not extra_val_train:
            key = incons.event["pfID"]
            others[key] = incons

    return False if len(others) > 0 else True


def __get_train_validation_test_type_from_file__(path: str):
    ret = path.split("\\")[-2]
    return ret


def __repair_name__(name: str) -> str:
    """

    :param name: filename to be repaired
    :return: filename where special characters (characters which are not allowed as filename) are replaced with _
    """
    return name.replace("|", "_").replace("?", "_").replace("\"", "_")


class InconsistencyHolder:
    def __init__(self, event: Series, episode_path: str):
        self.event = event
        self.episode_path = episode_path

        self.clip_subset = event["clip_split_subset"]
        self.episode_subset = event["episode_split_subset"]
        self.actual_episode_subset = __get_train_validation_test_type_from_file__(episode_path)


def remove_clips_with_no_length(clip_wav_path: str, expected_duration: float = 1.0, instant_delete=True) -> list:
    """
    Creates file "faults.json" which contains the found clips.
    :param clip_wav_path:
    :param expected_duration:
    :param instant_delete: If true, the found clips are instantly deleted.
    :return:
    """
    info = "Removing clips with no length"
    print(info)

    audio_files = glob.glob(os.path.join(clip_wav_path, "**/*.wav"), recursive=True)

    faults = []
    actual_duration: int
    for f in audio_files:
        try:
            if os.path.getsize(f) != 0 and \
                    ((expected_duration and (librosa.get_duration(path=f) == expected_duration))
                     or not expected_duration):
                continue
        except:
            pass

        faults.append(f)

    with open(os.path.join(clip_wav_path, "faults.json"), "w", encoding="UTF-8") as f:
        json.dump(faults, f, indent=4, ensure_ascii=False)

    if instant_delete:
        print("{} faulty clip-files found".format(str(len(faults))))
        remove_clips_according_to_json(os.path.join(clip_wav_path, "faults.json"))

    print("-DONE: {}".format(info))
    print("")
    return faults


def __read_all_subset_files__(train, test, valid) -> list:
    """

    :param train:
    :param test:
    :param valid:
    :return: list of 3-tuples: file-name, subset, label
    """

    all_lines = []

    with open(train, "r") as train_file:
        for line in train_file.readlines():
            all_lines.append([os.path.basename(line.replace("\n", "")), "train", os.path.dirname(line)])

    with open(test, "r") as test_file:
        for line in test_file.readlines():
            all_lines.append([os.path.basename(line.replace("\n", "")), "test", os.path.dirname(line)])

    with open(valid, "r") as valid_file:
        for line in valid_file.readlines():
            all_lines.append([os.path.basename(line.replace("\n", "")), "validation", os.path.dirname(line)])

    return all_lines


def check_labels(clip_wav_path: str, csv_path: str, n_vals: int, dest_path: str) -> (list, str):
    """
    Takes random clip-wav-samples and writes them in a csv-file.
    Basis for selection of the samples are the subset-files.

    :param clip_wav_path:
    :param csv_path:
    :param n_vals:
    :param dest_path:
    :return:
    """

    paths_for_running_in_colab = PathsForRunningInColabSingleton.get_instance()

    # TODO: move to separate class since this can also be used for CallHome-dataset
    considered_audio_files = __read_all_subset_files__(paths_for_running_in_colab.train_data_file,
                                                       paths_for_running_in_colab.test_data_file,
                                                       paths_for_running_in_colab.valid_data_file)

    ran_nums = list()

    if n_vals < len(considered_audio_files):
        i: int = 0
        rand = random.Random()
        while i < n_vals:
            r = rand.randint(0, len(considered_audio_files) - 1)
            if not r in ran_nums:
                ran_nums.append(r)
                i += 1
        ran_nums.sort()
    else:
        ran_nums = range(0, len(considered_audio_files))

    snap_samples = []
    for r in ran_nums:
        id = int(os.path.splitext(considered_audio_files[r][0])[0])
        snap_samples.append(LabelCheckData(id,
                                           os.path.join(
                                               clip_wav_path,
                                               considered_audio_files[r][1],
                                               str(id % paths_for_running_in_colab.n_subfolders),
                                               considered_audio_files[r][0]),
                                           considered_audio_files[r][2]))

    file_name = os.path.join(dest_path, "{}_LabelChecks.csv".format(datetime.datetime.now().strftime("%Y%m%d-%H%M%S")))
    with open(file_name, "w", newline="") as label_check_file:
        file_writer = csv.writer(label_check_file, delimiter=",")
        file_writer.writerow(["Id", "Path", "Label"])

        for sample in snap_samples:
            file_writer.writerow([sample.wavId, sample.clip_wav_path, sample.label])

    return snap_samples, file_name


class LabelCheckData:
    def __init__(self, wavId: int, clip_wav_path: str, label: str):
        self.wavId = wavId
        self.clip_wav_path = clip_wav_path
        self.label = label


def prepare_files_for_manual_check(csv_path: str, dest_path: str):
    """
    Copy clip-wavs, specified in csv_path, to dest_path.
    Create folder structure for manual checking, to check if labels of selected files are correct.
    The copied files have the naming convention: fileName_label.wav

    These clip-files should then be moved to corresponding folders manually:
        ___NotOk: clips with wrong label
        __NotSure: clips, whose label is neither certainly correct nor wrong
        OK: clips with correct label
        _(_)Here should land nothing (2): Folders as separators between the other folders. These folders should stay
        empty.

    :param csv_path: Path to csv-file with selected clips to manually check.
    :param dest_path: Path, were the files are copied to. This path should not exist or at least be an empty directory.
    :return:
    """
    info = "Copying files for manual check"
    print(info)

    if not os.path.isdir(dest_path):
        os.mkdir(dest_path)

    if not len(os.listdir(dest_path)) == 0:
        raise FileExistsError

    # copy csv
    shutil.copyfile(csv_path, os.path.join(dest_path, os.path.basename(csv_path)))

    # copy files
    dest_path = os.path.join(dest_path, "files")
    os.mkdir(dest_path)
    os.mkdir(os.path.join(dest_path, "_Moved"))
    os.mkdir(os.path.join(dest_path, "_Moved", "___NotOk"))
    os.mkdir(os.path.join(dest_path, "_Moved", "__Here should land nothing"))
    os.mkdir(os.path.join(dest_path, "_Moved", "__Not sure"))
    os.mkdir(os.path.join(dest_path, "_Moved", "_Here should land nothing 2"))
    os.mkdir(os.path.join(dest_path, "_Moved", "OK"))

    csv_file = pd.read_csv(csv_path)
    for i, check_data in csv_file.iterrows():
        file_name_with_label = os.path.join(dest_path, "{}_{}.wav".format(
            os.path.splitext(os.path.basename(check_data["Path"]))[0], check_data["Label"]))

        actual_path: str = check_data["Path"]
        if not os.path.exists(check_data["Path"]):
            possible_files = glob.glob(check_data["Path"])
            if len(possible_files) != 1:
                raise FileNotFoundError  # I don't know if throwing error is the best solution here. Alternatively, comment this line.
                continue
            actual_path = possible_files[0]

        shutil.copyfile(actual_path, file_name_with_label)

    print("-DONE: {}".format(info))


def remove_clips_according_to_json(faults_json_path: str):
    info: str = "Deleting faulty clip files"
    print(info)

    file = open(faults_json_path, "r", encoding="UTF-8")
    faults = json.load(file)

    for f in faults:
        if os.path.exists(f):
            os.remove(f)

    print("-DONE: {}".format(info))


def repair(dataset_path: str):
    paths_for_running_in_colab = PathsForRunningInColabSingleton.get_instance()

    target_path = os.path.join(paths_for_running_in_colab.metadata_folder, "PodcastFillers_repaired.csv")
    csv_repairer = CSVRepairer(dataset_path, paths_for_running_in_colab.master_csvfile)
    compromised_files = csv_repairer.adapt_filenames(target_path)
    paths_for_running_in_colab.master_csvfile = target_path
    paths_for_running_in_colab.master_csv_name = os.path.basename(target_path)
    print("Wrong file names: " + str(compromised_files))
    out = csv_repairer.find_train_versus_validation_versus_test_inconsistencies(target_path)
    print("Wrong train/test/validation-set: " + str(out))


def get_mp3_from_corrupted(data_path, clips_wav_path, master_csv_file, faults_path: str):
    """
    Get mp3 files to corrupted clips. For example, it occurred that there were clip-files with 0 Bytes or 0s length.
    This Methode gives per episode.mp3 the amount of such clips.
    :param data_path:
    :param clips_wav_path:
    :param master_csv_file:
    :return:
    """
    import pandas as pd
    event_df = pd.read_csv(master_csv_file, encoding="utf-8")

    event_arr = []
    for event in event_df.iterrows():
        event_arr.append(event)

    file = open(faults_path, "r")
    faults = json.load(file)

    parent_mp3_s = dict()
    for fault in faults:
        file_id = os.path.splitext(os.path.basename(fault))[0]
        parent = str((event_arr[int(file_id)])[1]["podcast_filename"])

        if not parent in parent_mp3_s:
            parent_mp3_s[parent] = list()

        parent_mp3_s[parent].append(file_id)

    paths_for_running_in_colab = PathsForRunningInColabSingleton.get_instance()

    try:
        with open(os.path.join(paths_for_running_in_colab.clip_folder_path, "mp3FromBrokenClips.json"), "w", encoding="UTF-8") as f:
            json.dump(parent_mp3_s, f, indent=4, ensure_ascii=False)
        print("")
    except:
        pass

    return parent_mp3_s
