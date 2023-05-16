import os.path

import pandas as pd

from Temporal_Convolution_Resnet.model_dataloader import filler_categories
from filler_detection_software import visualiziation
from utils import filler_occurence
from utils.filler_occurence import FillerOccurrence


def visualize_filler_occurrences_according_to_master_csv(master_csvfile: str, podcast_path: str):
    """
    Visualize the annotated fillers (according to master_csvfile) in .wav-file (podcast_path).
    This function only works if monitor is available, it does not run in Google-Colaboratory.
    :param master_csvfile:
    :param podcast_path:
    :return:
    """
    podcast_name = os.path.splitext(os.path.basename(podcast_path))[0]
    event_df = pd.read_csv(master_csvfile)
    filler_occurrences = []
    for i, event in event_df.iterrows():
        if event["podcast_filename"] == podcast_name:
            if event["label_consolidated_vocab"] in filler_categories:
                filler_occurrences.append(
                    FillerOccurrence(event["event_start_inepisode"], event["event_end_inepisode"], 1.0))

    filler_occurrences = filler_occurence.sort_fillers(filler_occurrences)
    v = visualiziation.Visualizer(filler_occurrences, podcast_path)
    v.visualize_filler_occurrences()


def get_corrected_labels_for_episode(master_csvfile: str, podcast_path: str, corrected_labels_path: str) -> list[
    FillerOccurrence]:
    """

    :param master_csvfile:
    :param podcast_path:
    :param corrected_labels_path: None or Pathlike. If None, then no corrections of labels are applied.
    :return:
    """
    fillers_labels = []
    podcast_name = os.path.splitext(os.path.basename(podcast_path))[0]
    event_df = pd.read_csv(master_csvfile)
    for i, event in event_df.iterrows():
        if event["podcast_filename"] == podcast_name:
            if event["label_consolidated_vocab"] in filler_categories:
                fillers_labels.append(
                    FillerOccurrence(event["event_start_inepisode"], event["event_end_inepisode"], 1.0))

    def split_line(l: str) -> (float, float):
        t_start, t_end = l.split(";")
        return float(t_start), float(t_end)

    if corrected_labels_path and os.path.exists(corrected_labels_path):
        current_file_reading_state = 0
        with open(corrected_labels_path, "r") as corrected_labels_file:
            for line in corrected_labels_file.readlines():
                match current_file_reading_state:
                    case 0:
                        if podcast_name in line:
                            current_file_reading_state = 1
                    case 1:
                        if "missing nonfillers" in line.lower():
                            current_file_reading_state = 2
                    case 2:
                        if "missing fillers" in line.lower():
                            current_file_reading_state = 3
                        if line and ";" in line:
                            t_start, t_end = split_line(line)

                            removes = []

                            for filler in fillers_labels:
                                if filler.start_time > t_start and filler.end_time < t_end:
                                    removes.append(filler)

                            if len(removes) != 1:
                                raise NotImplementedError()

                            for r in removes:
                                fillers_labels.remove(r)
                    case 3:
                        if line and ";" in line:
                            fillers_labels.append(
                                FillerOccurrence(*split_line(line), 1.0))
                        else:
                            pass

            if current_file_reading_state != 3:
                # file did not match expected format
                raise NotImplementedError()
    else:
        print("No correction file given.")

    fillers_labels = filler_occurence.sort_fillers(fillers_labels)
    return fillers_labels


#visualize_filler_occurrences_according_to_master_csv(paths_for_running_in_colab.master_csvfile,
#                                                     "../example_data/CEO Interview 2017_Pat Finn.wav")

#fillers = get_corrected_labels_for_episode(paths_for_running_in_colab.master_csvfile,
#                                           "../example_data/CEO Interview 2017_Pat Finn.wav",
#                                           "../example_data/CEO Interview 2017_Pat Finn.txt")
