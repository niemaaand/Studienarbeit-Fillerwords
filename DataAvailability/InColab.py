import glob
import os
import shutil

from paths_for_running_in_colab import PathsForRunningInColabSingleton


def ensure_clips_availability_colab(dest_path: str):
    """
        Copy (train, test, valid) data from google-colab (path in paths_for_running_in_colab) to dest_path.
        Since zip-extracting is faster, upload a zip-file with the data to colab and use extract_zip().
    :return:
    """

    paths_for_running_in_colab = PathsForRunningInColabSingleton.get_instance()

    paths_for_running_in_colab.dataset_path = paths_for_running_in_colab.dataset_path_colab
    paths_for_running_in_colab.reload_all_paths()
    __ensure_clips_availability__(dest_path)
    paths_for_running_in_colab.dataset_path = dest_path
    paths_for_running_in_colab.reload_all_paths()


def __ensure_clips_availability__(dest_path: str):
    paths_for_running_in_colab = PathsForRunningInColabSingleton.get_instance()

    if os.path.exists(dest_path):
        raise FileExistsError

    info = "Copying clips"
    print(info)

    os.mkdir(dest_path)
    os.mkdir(os.path.join(dest_path, "metadata"))

    # copy relevant metadata
    # master-csv
    print("-Copy metadata")
    shutil.copy(paths_for_running_in_colab.master_csvfile,
                os.path.join(dest_path, "metadata", os.path.basename(paths_for_running_in_colab.master_csvfile)))

    # subsets
    subset_files = [paths_for_running_in_colab.test_data_file,
                    paths_for_running_in_colab.valid_data_file,
                    paths_for_running_in_colab.train_data_file
                    # ,paths_for_running_in_colab.extra_data_file
                    ]

    duration_of_clip_folder = os.path.join(dest_path, "metadata", paths_for_running_in_colab.subset_files_folder)
    os.mkdir(duration_of_clip_folder)

    for d_file in subset_files:
        tar_path = os.path.join(duration_of_clip_folder, os.path.basename(d_file))
        if not os.path.exists(tar_path):
            shutil.copy(paths_for_running_in_colab.test_data_file, tar_path)

    # copy audio clips
    # copy train
    print("-Copy train data")
    __copy_audios__(paths_for_running_in_colab.train_clips_folder, dest_path)

    # copy valid
    print("-Copy valid data")
    __copy_audios__(paths_for_running_in_colab.valid_clips_folder, dest_path)

    # copy test
    print("-Copy test data")
    __copy_audios__(paths_for_running_in_colab.test_clips_folder, dest_path)

    # copy extra
    # print("-Copy extra data")
    # __copy_audios__(paths_for_running_in_colab.extra_clips_folder, dest_path)

    print("-DONE: {}".format(info))


def __copy_audios__(subset_folder_name: str, dest_path: str):
    paths_for_running_in_colab = PathsForRunningInColabSingleton.get_instance()
    dest_clip_path = os.path.join(dest_path, "audio", paths_for_running_in_colab.clip_folder_name,
                                  subset_folder_name)
    src = os.path.join(paths_for_running_in_colab.clip_folder_path, subset_folder_name)

    for i in range(0, paths_for_running_in_colab.n_subfolders):
        if i % 10 == 0:
            print("- {} folders copied".format(i))

        current_path = os.path.join(src, str(i))
        dest_sub_clip_path = os.path.join(dest_clip_path, str(i))
        if not os.path.exists(dest_sub_clip_path):
            if os.path.exists(current_path):
                shutil.copytree(current_path, dest_sub_clip_path)


def copy_and_extract_zip(dest_path: str, master_csv_in_zip_path: str = None):
    """
    Methode to extract zip-file, which is supposed to contain (train, valid, test) data.
    Path of zip-file has to be specified in paths_for_running_in_colab.
    :param dest_path: Path to extract to.
    :return:
    """

    paths_for_running_in_colab = PathsForRunningInColabSingleton.get_instance()

    info: str = "Extracting zip"
    print(info)
    # extract zip
    from zipfile import ZipFile
    src_path = paths_for_running_in_colab.colab_zip_path

    paths_for_running_in_colab.dataset_path = paths_for_running_in_colab.dataset_path_colab
    paths_for_running_in_colab.reload_all_paths()

    target = dest_path
    if not os.path.exists(target):
        os.mkdir(target)
    else:
        raise FileExistsError

    audio = os.path.join(target, "audio")
    metadata = os.path.join(target, "metadata")

    os.mkdir(audio)
    os.mkdir(metadata)

    with ZipFile(src_path, "r") as zip_ref:
        zip_ref.extractall(audio)

    # master-csv
    print("-Copy metadata")
    if master_csv_in_zip_path:
        shutil.copy(os.path.join(dest_path, master_csv_in_zip_path),
                    os.path.join(metadata, os.path.basename(paths_for_running_in_colab.master_csvfile)))
    else:
        shutil.copy(paths_for_running_in_colab.master_csvfile,
                    os.path.join(metadata, os.path.basename(paths_for_running_in_colab.master_csvfile)))

    # update paths / variables
    paths_for_running_in_colab.dataset_path = target
    paths_for_running_in_colab.reload_all_paths()

    print("-DONE: {}".format(info))


def move_subsets_to_smaller_folders(n_folders: int):
    """
    Move subsets (train, test, valid) into n_folders subfolders.
    Purpose: easier copying of this data in google-colab.
    :param n_folders:
    :return:
    """

    paths_for_running_in_colab = PathsForRunningInColabSingleton.get_instance()

    files = glob.glob(os.path.join(paths_for_running_in_colab.clip_folder_path, "**/*.wav"))

    for f in files:
        sub_dir = str(int(os.path.splitext(os.path.basename(f))[0]) % n_folders)

        sub_dir_path = os.path.join(os.path.dirname(f), sub_dir)

        if not os.path.exists(sub_dir_path):
            os.mkdir(sub_dir_path)

        shutil.move(f, os.path.join(sub_dir_path, os.path.basename(f)))
