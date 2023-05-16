import os
import torch.nn.functional as F
import torchaudio
from torch.utils.data import Dataset

from paths_for_running_in_colab import PathsForRunningInColabSingleton

__classes_precise__ = ["Uh", "Um", "Laughter", "Breath", "Words", "Music"]
__classes__ = ["filler", "non_filler"]

filler_categories = ["Uh", "Um"]
non_filler_categories = ["Laughter", "Breath", "Words", "Music"]


class FillersDataset(Dataset):
    """
    Base class for filler datasets.
    """
    def __init__(self, datapath, filename, is_training, subfoldername="", sampling_rate=16000):
        super(FillersDataset).__init__()

        paths_for_running_in_colab = PathsForRunningInColabSingleton.get_instance()

        self.sampling_rate = sampling_rate
        self.sample_length = self.sampling_rate * paths_for_running_in_colab.duration  # length of clip
        self.datapath = datapath
        self.filename = filename
        self.is_training = is_training
        self.subfoldername = subfoldername
        self.n_folders = paths_for_running_in_colab.n_subfolders

        self.class_encoding = {category: index for index, category in enumerate(__classes__)}

        # Bundle the file path of clip with labels -> list of [path, label]
        self.speech_dataset = self.paths_with_labels()

    def paths_with_labels(self):
        dataset_list = []
        for path in self.filename:
            category, wave_name = path.split("/")  # label of clip and Names of clip

            if category not in __classes__:
                if category in __classes_precise__:
                    if category in filler_categories:
                        category = "filler"
                    elif category in non_filler_categories:
                        category = "non_filler"
                    else:
                        raise NameError

            if category in __classes__:
                path = os.path.join(self.datapath, self.subfoldername,
                                    str(int(os.path.splitext(wave_name)[0]) % self.n_folders), wave_name)
                dataset_list.append([path, category])  # Add a list as a pair by grouping clip routes and categories
            else:
                raise NameError

        return dataset_list

    def load_audio(self, speech_path):
        waveform, sr = torchaudio.load(speech_path)

        if waveform.shape[1] < self.sample_length:
            waveform = F.pad(waveform, [0, self.sample_length - waveform.shape[1]])
        else:
            pass

        return waveform

    def one_hot(self, speech_category):  # A function that encodes a category into a numeric value
        encoding = self.class_encoding[speech_category]
        return encoding

    def __len__(self):
        return len(self.speech_dataset)

    def __getitem__(self, index):
        speech_path = self.speech_dataset[index][0]  # path
        speech_category = self.speech_dataset[index][1]  # category
        label = self.one_hot(speech_category)

        waveform = self.load_audio(speech_path)

        return waveform, label


class PodcastFillersDataset(FillersDataset):
    def __init__(self, datapath, filename, is_training: bool, subfoldername="", sampling_rate=16000):
        super(PodcastFillersDataset, self).__init__(datapath, filename, is_training, subfoldername, sampling_rate)


class CallHomeFillersDataset(FillersDataset):
    def __init__(self, datapath, filename, is_training: bool, subfoldername="", sampling_rate=16000):
        super(CallHomeFillersDataset, self).__init__(datapath, filename, is_training, subfoldername, sampling_rate)
