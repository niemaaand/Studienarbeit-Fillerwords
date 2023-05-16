import json
import os
import time
from multiprocessing import Process

from paths_for_running_in_colab import PathsForRunningInColabSingleton
from datapreprocessing.podcast_fillers_reparation import remove_clips_according_to_json, repair, get_mp3_from_corrupted
from DataAvailability import InColab
from Temporal_Convolution_Resnet.options import OptionsInformation, read_options_from_json
from datapreprocessing import podcast_fillers_reparation
from datapreprocessing.create_set_files import create_subset_files
from Temporal_Convolution_Resnet import model_train_fillers
from datapreprocessing.preprocessing import cut_clips
from evaluation.evaluation_on_clips import evaluate_model, DatasetSubset


def manual_checking(clip_wav_path, master_csvfile, n, target_path):
    """
    Copy some random clips to target_path to check labels.
    :param clip_wav_path:
    :param master_csvfile:
    :param n: Number of samples.
    :param target_path:
    :return:
    """
    _, file_name = podcast_fillers_reparation.check_labels(clip_wav_path, master_csvfile, n,
                                                           target_path)

    podcast_fillers_reparation.prepare_files_for_manual_check(file_name,
                                                              os.path.join(target_path,
                                                                           "files_{}".format(os.path.splitext(
                                                                               os.path.basename(file_name))[0])))


def run_in_colab(dest_path="/content/extracted/"):
    paths_for_running_in_colab = PathsForRunningInColabSingleton.get_instance()
    paths_for_running_in_colab.path_to_saved_models: str = paths_for_running_in_colab.path_saved_models_colab

    if not os.path.exists(dest_path):
        InColab.copy_and_extract_zip(dest_path, "audio/clip_wav_regenerate_2s/NewMasterCSV.csv")
        create_subset_files(paths_for_running_in_colab.master_csvfile, paths_for_running_in_colab.metadata_folder,
                            paths_for_running_in_colab.clip_folder_path)
    else:
        # update paths / variables
        paths_for_running_in_colab.dataset_path_colab: str = dest_path
        paths_for_running_in_colab.dataset_path = dest_path
        paths_for_running_in_colab.reload_all_paths()


def retrain_on_corrected_labels():
    paths_for_running_in_colab = PathsForRunningInColabSingleton.get_instance()
    # create new master csv and adapt paths
    # paths_for_running_in_colab.master_csvfile = \
    #    new_master_csv_from_manually_checked.update_master_csv_from_manual_check(
    #    paths_for_running_in_colab.master_csvfile,
    #    os.path.join(paths_for_running_in_colab.dataset_path, "ManualChecking", "Completed_zip"),
    #    datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + "_NewMasterCSV.csv"
    # )
    # paths_for_running_in_colab.master_csv_name = os.path.basename(paths_for_running_in_colab.master_csvfile)
    # paths_for_running_in_colab.reload_paths()

    # create_subsetfiles(paths_for_running_in_colab.master_csvfile, paths_for_running_in_colab.metadata_folder,
    #                  paths_for_running_in_colab.clip_folder_path)

    # manually set paths (new master csv created previously)
    # paths_for_running_in_colab.master_csvfile = "C:/Studienarbeit_Daten_Sicherungen/metadata/20230412-151616_NewMasterCSV.csv"
    # paths_for_running_in_colab.master_csv_name = os.path.basename(paths_for_running_in_colab.master_csvfile)
    # paths_for_running_in_colab.subset_files_folder = "2s_checked"
    # paths_for_running_in_colab.reload_paths()

    # build options and model
    p = "pretrained_models/chapter_6_2/tcresnet_variation/8_1_DS"
    options_path = os.path.join(p, "options.json")
    opt1 = read_options_from_json(options_path)
    opt1.reload_model = os.path.join(p, "best_dsmix_tcresnet_pool_drop11_tensor(0.9052, device='cuda_0').pt")

    evaluate_model(None, opt1, os.path.join(paths_for_running_in_colab.dataset_path, "audio/"), paths_for_running_in_colab.clip_folder_name, DatasetSubset.TEST)

    paths_for_running_in_colab.master_csvfile = "pretrained_models/chapter_6_3/checked_metadata/MasterCsvWithManuallyCheckedAnnotations.csv"
    paths_for_running_in_colab.master_csv_name = os.path.basename(paths_for_running_in_colab.master_csvfile)
    paths_for_running_in_colab.subset_files_folder = "subsets"
    paths_for_running_in_colab.metadata_folder = "pretrained_models/chapter_6_3/checked_metadata"
    paths_for_running_in_colab.reload_metadata_paths()

    opt1.epoch = 120
    opt1.lr = opt1.lr / 10
    opt1.retraining = True
    opt1.save = "RetrainedWithCorrectedLabels_{}_NEW".format(opt1.save)

    args_with_opt = (os.path.join(paths_for_running_in_colab.dataset_path, "audio/"),
                     paths_for_running_in_colab.clip_folder_name,
                     os.path.join(paths_for_running_in_colab.path_to_saved_models, "model_save_2s"),
                     opt1)

    start_threads(1, [args_with_opt, args_with_opt, args_with_opt])

    pass


def preprocess():
    """
    Complete preprocessing. Including repairing of master-csv, cutting clips, creating subset files.
    :return:
    """
    paths_for_running_in_colab = PathsForRunningInColabSingleton.get_instance()
    repair(paths_for_running_in_colab.dataset_path_local)
    cut_clips(paths_for_running_in_colab.dataset_path)
    broken = podcast_fillers_reparation.remove_clips_with_no_length(
        paths_for_running_in_colab.clip_folder_path, expected_duration=2.0, instant_delete=False)
    corrupted_mp3s = get_mp3_from_corrupted(paths_for_running_in_colab.dataset_path, paths_for_running_in_colab.clip_folder_path, paths_for_running_in_colab.master_csvfile,
                                            os.path.join(paths_for_running_in_colab.clip_folder_path, "faults.json"))

    # only necessary, if podcast_fillers_reparation.remove_clips_with_no_length(..., instant_delete=False)
    remove_clips_according_to_json(os.path.join(paths_for_running_in_colab.clip_folder_path, "faults.json"))

    create_subset_files(paths_for_running_in_colab.master_csvfile, paths_for_running_in_colab.metadata_folder,
                        paths_for_running_in_colab.clip_folder_path)


def start_threads(n_threads: int, args: list):
    processes = list()
    for i in range(n_threads):
        p = Process(target=begin_of_new_thread, args=[args[i], PathsForRunningInColabSingleton.get_instance()])
        p.start()
        time.sleep(2)
        processes.append(p)

    for t in processes:
        t.join()


def begin_of_new_thread(args, paths):
    PathsForRunningInColabSingleton.set_instance(paths)  # keep values in new thread/process
    model_train_fillers.run_training(*args)


def evaluate_multiple_models():

    paths_for_running_in_colab = PathsForRunningInColabSingleton.get_instance()

    save_path = "pretrained_models/chapter_6_4"

    reload_path = os.path.join(save_path, "1/options.json")
    o13 = read_options_from_json(reload_path)
    o13.reload_model = os.path.join(save_path, "1/best_dsmix_tcresnet_pool_drop13_tensor(0.8903, device='cuda_0').pt")

    reload_path = os.path.join(save_path, "2/options.json")
    o14 = read_options_from_json(reload_path)
    o14.reload_model = os.path.join(save_path, "2/best_dsmix_tcresnet_pool_drop42_tensor(0.9086, device='cuda_0').pt")

    reload_path = os.path.join(save_path, "3/options.json")
    o15 = read_options_from_json(reload_path)
    o15.reload_model = os.path.join(save_path, "3/model_dsmix_tcresnet_pool_drop15_tensor(0.9079, device='cuda_0').pt")

    options = [o13, o14, o15]

    evaluate_model(None, options, os.path.join(paths_for_running_in_colab.dataset_path, "audio/"), paths_for_running_in_colab.clip_folder_name, DatasetSubset.VALID)

    pass


if __name__ == '__main__':

    paths_for_running_in_colab = PathsForRunningInColabSingleton.get_instance()

    preprocess()

    #evaluate_multiple_models()
    #retrain_on_corrected_labels()
    #performance_comparison_pydub_librosa(os.path.join(paths_for_running_in_colab.dataset_path, "audio", "episode_wav_regenerate"))
    #calc_mean_event_time(paths_for_running_in_colab.master_csvfile, paths_for_running_in_colab.clip_folder_path)

    #run_in_colab()
    #get_mp3_from_corrupted(paths_for_running_in_colab.dataset_path, paths_for_running_in_colab.clip_folder_path,
    #                     paths_for_running_in_colab.master_csvfile,
    #                     "C:/Studienarbeit_Daten_Sicherungen/audio/clip_wav_regenerate_2s/faults.json")

    # manually check correct labels in data set
    #manual_checking("C:/Studienarbeit_Daten_Sicherungen/audio/clip_wav_regenerate_2s",
    #               "C:/Studienarbeit_Daten_Sicherungen/audio/clip_wav_regenerate_2s/NewMasterCSV.csv", 1000,
    #              "C:\Studienarbeit_Daten_Sicherungen\ManualChecking")

    o15 = OptionsInformation()
    o15.model = "dsmix_tcresnet_pool_drop"
    o15.last_res_pool = 2
    o15.p_dropout = 0.2
    o15.p_dropout_in_res = 0.2
    o15.trainable_feature_extraction = False
    o15.feature_extraction = "stft"
    o15.channel_scale = 1
    o15.hop_length = 128
    o15.optim = "sgd"
    o15.n_classifiers = 1
    o15.n_model = 1
    o15.k_fold = 0
    o15.train_data_size = 0
    o15.leaky_relu = True
    o15.set_everything()

    args_with_opt = (os.path.join(paths_for_running_in_colab.dataset_path, "audio/"),
                     paths_for_running_in_colab.clip_folder_name,
                     os.path.join(paths_for_running_in_colab.path_to_saved_models, "model_save_2s"),
                     o15)

    start_threads(1, [args_with_opt, args_with_opt])

    print("DONE")
