import os.path
from io import TextIOWrapper

import pandas as pd
from pandas import DataFrame

from paths_for_running_in_colab import PathsForRunningInColabSingleton


def create_subset_files(master_csvfile: str, metadata_folder: str, clip_wav_path: str):
    """
    Create subset-files (train, test, valid) based on master-csv file.
    Only files, which actually exist at clip_wav_path are added to subset-files.
    Names / Paths of subset-files to create have to be specified in paths_for_running_in_colab.
    :param master_csvfile: Path to master-csv-file.
    :param metadata_folder:  Path to folder with metadata.
    :param clip_wav_path: Path to folder with clips (.wav).
    :return:
    """

    paths_for_running_in_colab = PathsForRunningInColabSingleton.get_instance()

    info: str = "Creating subset-files (training, validation, testing)"
    print(info)
    event_df = pd.read_csv(master_csvfile)

    train_file: TextIOWrapper
    validation_file: TextIOWrapper
    testing_file: TextIOWrapper
    extra_file: TextIOWrapper

    d = os.path.dirname(paths_for_running_in_colab.train_data_file)
    if not os.path.exists(d):
        os.mkdir(d)

    try:
        with open(paths_for_running_in_colab.train_data_file, "w") as train_file:
            with open(paths_for_running_in_colab.valid_data_file, "w") as validation_file:
                with open(paths_for_running_in_colab.test_data_file, "w") as testing_file:

                    for i, event in event_df.iterrows():

                        if __exists_wav_file__(event, clip_wav_path):

                            subset = __is_label_certain_enough_else_extra_(event)

                            # label/filename
                            label = event["label_consolidated_vocab"]
                            filename = event["clip_name"]

                            line = "{}/{}\n".format(label, filename)

                            if subset == "train":
                                train_file.write(line)
                            elif subset == "validation":
                                validation_file.write(line)
                            elif subset == "test":
                                testing_file.write(line)
    except:
        print("-ERROR")

    try:
        train_file.close()
        validation_file.close()
        testing_file.close()
    except:
        print("-ERROR")

    print("-DONE: {}".format(info))


def __exists_wav_file__(event: DataFrame, clip_wav_path: str) -> bool:

    paths_for_running_in_colab = PathsForRunningInColabSingleton.get_instance()

    clip_split_subset = event["clip_split_subset"]
    clip_name = event["clip_name"]

    path = os.path.join(clip_wav_path, clip_split_subset,
                        str(int(os.path.splitext(clip_name)[0]) % paths_for_running_in_colab.n_subfolders), clip_name)
    if os.path.exists(path):
        return True

    return False


def __is_label_certain_enough_else_extra_(event: DataFrame) -> str:
    """

    :param event:
    :return: Label of event, if confidence is high enough. If confidence is not high enough: "extra".
    """
    if event["confidence"] >= 0.6 and event["label_consolidated_vocab"] != "None":
        return event["clip_split_subset"]

    return "extra"
