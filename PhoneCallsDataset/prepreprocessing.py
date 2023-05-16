import glob
import math
import os
import shutil

from pydub import AudioSegment

from paths_for_running_in_colab import PathsForRunningInColabSingleton
from sklearn.model_selection import train_test_split

from PhoneCallsDataset import create_master_csv


class CallHomeIntoStructure:
    def __init__(self, dataset_path: str = "C:/CallHomeDataset/"):
        """
        :param dataset_path: Path to CallHomeDataset.
        """

        self.dataset_path: str = dataset_path
        self.episode_wav_path: str = os.path.join(dataset_path, "audio/episode_wav_regenerate")
        self.episode_transcripts_path: str = os.path.join(dataset_path, "metadata/eng")
        self.original_master_csv: str = os.path.join(dataset_path,
                                                "metadata/CallHomeDataset_2s.csv")  # paths_for_running_in_colab.master_csvfile
        self.new_master_csv_name: str = "CallHomeDataset.csv"
        self.manual_checking_path: str = os.path.join(dataset_path,
                                                 "ManualChecking/files_20230322-143559_LabelChecks/files/_Moved")

    def bring_to_structure(self):
        """
        Before executing this function:
            1. Build directory-hierarchy: CallHomeDataset/audio/episode_wav_regenerate
            2. Download all episode-wav-fiels of CallHome dataset to directory episode_wav_regenerate.
            3. Set dataset_path to absolute path of CallHomeDataset (constructor).
        :return:
        """

        info: str = "Bringing CallHome dataset to structure (Convert, rename, split, create master-csv). "
        print(info)

        __convert_files_at_directory_to_mono__(self.episode_wav_path)
        __rename__(self.episode_wav_path, self.episode_transcripts_path)
        __split_into_subsets__(self.episode_wav_path)
        create_master_csv.create_master_csv(self.episode_wav_path, self.episode_transcripts_path, self.original_master_csv)

        print("-Done: {}".format(info))

    def update_master_csv(self):
        """
        Before running this function, a ManualChecking-directory-hierarchy should be build and the manual checking
        should have been performed.
        :return:
        """
        info: str = "Updating master csv. Src: {} Target: {}".format(self.original_master_csv, self.new_master_csv_name)
        print(info)

        create_master_csv.update_master_csv_from_manual_check(self.original_master_csv,
                                                              os.path.join(self.manual_checking_path, "___NotOk"),
                                                              os.path.join(self.manual_checking_path, "OK"),
                                                              self.new_master_csv_name)

        print("-Done: {}".format(info))


def __split_into_subsets__(full_wav_path: str, train: float = 0.5, valid: float = 0.25, test: float=0.25):

    paths_for_running_in_colab = PathsForRunningInColabSingleton.get_instance()

    wav_files = glob.glob(os.path.join(full_wav_path, "*.wav"))

    if not math.isclose(train + valid + test, 1.0, abs_tol=1e-8):
        raise ValueError

    if len(wav_files) > 0:
        subsets: [] = [list, list, list]
        subsets[0], remaining = train_test_split(wav_files, train_size=train)

        if len(remaining) > 0:
            subsets[1], subsets[2] = train_test_split(remaining, test_size=test/(test + valid))

        folder_names: [] = [paths_for_running_in_colab.train_clips_folder,
                            paths_for_running_in_colab.valid_clips_folder,
                            paths_for_running_in_colab.test_clips_folder]

        for i in range(len(subsets)):
            subset = subsets[i]

            sub_dir_target = os.path.join(full_wav_path, folder_names[i])
            if not os.path.exists(sub_dir_target):
                os.mkdir(sub_dir_target)

            for f in subset:
                shutil.move(f, os.path.join(sub_dir_target, os.path.basename(f)))
    pass


def __rename__(full_wav_path: str, transcrib_path: str, rename_wavs=False, rename_trans=False):
    """
    Add underscore to all wav-files at full_wav_path and to all cha-files at transcrib_path.
    Without underscore, these file-names are only numeric which is converted to an int by creating/reading csv-files
    for train, test, valid.
    :param full_wav_path:
    :param transcrib_path:
    :param rename_wavs:
    :param rename_trans:
    :return:
    """
    if rename_wavs:
        wav_files = glob.glob(os.path.join(full_wav_path, "**/*.wav"))

        for f in wav_files:
            os.rename(f, os.path.join(os.path.dirname(f), "_{}".format(os.path.basename(f))))

    if rename_trans:
        trans_files = glob.glob(os.path.join(transcrib_path, "*.cha"))

        for f in trans_files:
            os.rename(f, os.path.join(os.path.dirname(f), "_{}".format(os.path.basename(f))))


def __convert_files_at_directory_to_mono__(folder_path: str):
    """
    Convert all wav-files at folder_path and in depth of two sub-directories to mono-channel and 16kHz.
    This method should be executed before cutting clips. (But can also be done afterwards.)
    Overwrites all files!
    :param folder_path:
    :return:
    """
    wav_files_1 = glob.glob(os.path.join(folder_path, "*.wav"))
    wav_files_2 = glob.glob(os.path.join(folder_path, "**/*.wav"))
    wav_files_3 = glob.glob(os.path.join(folder_path, "*", "**/*.wav"))

    __convert_to_mono__(wav_files_1)
    __convert_to_mono__(wav_files_2)
    __convert_to_mono__(wav_files_3)


def __convert_to_mono__(files: list):
    for wav_f in files:
        sound = AudioSegment.from_file(wav_f, format='wav')
        sound = sound.set_frame_rate(16000)
        sound = sound.set_channels(1)
        sound.export(wav_f, format='wav')






