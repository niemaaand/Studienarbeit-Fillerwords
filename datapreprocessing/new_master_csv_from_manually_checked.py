import datetime
import os, glob
import pandas as pd


from PhoneCallsDataset.create_master_csv import ClipInformationHolder, __get_clip_information_from_filename__, __write_csv__


def update_master_csv_from_manual_check(master_csv_path: str, manually_checked_zips_directory: str, csv_target_name: str) -> str:
    """
    Create new master-csv from manually checked labels.
    :param master_csv_path:
    :param manually_checked_zips_directory:
    :param csv_target_name:
    :return: path of new master-csv
    """
    from zipfile import ZipFile

    clips: list(ClipInformationHolder) = list()

    unpacked_path = os.path.join(manually_checked_zips_directory, "{}_unpacked".format(datetime.datetime.now().strftime("%Y%m%d-%H%M%S")))
    if os.path.exists(unpacked_path):
        raise FileExistsError

    os.mkdir(unpacked_path)

    for src_path in os.listdir(manually_checked_zips_directory):
        if os.path.splitext(src_path)[1] == ".zip":
            with ZipFile(os.path.join(manually_checked_zips_directory, src_path), "r") as zip_ref:
                zip_ref.extractall(unpacked_path)

    event_df = pd.read_csv(master_csv_path)
    nok_files__actually_filler = list()
    nok_files__actually_non_filler = list()
    ok_files = list()
    nok_path_extension__actually_filler = os.path.join("files", "_Moved", "___NotOk", "filler")
    nok_path_extension__actually_non_filler = os.path.join("files", "_Moved", "___NotOk", "non_filler")
    ok_path_extension = os.path.join("files", "_Moved", "OK")
    for directory in os.listdir(unpacked_path):

        nok_files__actually_filler += glob.glob(os.path.join(unpacked_path, directory, nok_path_extension__actually_filler, "*.wav"))
        nok_files__actually_non_filler += glob.glob(os.path.join(unpacked_path, directory, nok_path_extension__actually_non_filler, "*.wav"))
        ok_files += glob.glob(os.path.join(unpacked_path, directory, ok_path_extension, "*.wav"))

    for nok_f in nok_files__actually_filler:
        clips.append(__get_clip_information_from_filename__(nok_f, event_df, not_ok=True))

    for nok_f in nok_files__actually_non_filler:
        clips.append(__get_clip_information_from_filename__(nok_f, event_df, not_ok=True))

    for ok_f in ok_files:
        clips.append(__get_clip_information_from_filename__(ok_f, event_df, not_ok=False))

    new_master_csv = os.path.join(os.path.dirname(master_csv_path), csv_target_name)
    __write_csv__(clips, new_master_csv)

    return new_master_csv


