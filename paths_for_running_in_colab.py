import os


class PathsForRunningInColabSingleton:
    """
    This class is singleton!!!
    """
    __instance__ = None

    @staticmethod
    def get_instance():
        if not PathsForRunningInColabSingleton.__instance__:
            PathsForRunningInColabSingleton.__instance__ = PathsForRunningInColabSingleton()

        return PathsForRunningInColabSingleton.__instance__

    @staticmethod
    def set_instance(instance):
        if PathsForRunningInColabSingleton.__instance__ or not isinstance(instance, PathsForRunningInColabSingleton):
            raise ValueError

        PathsForRunningInColabSingleton.__instance__ = instance

    def __init__(self):
        """
        This class is singleton. Do not instantiate. Use get_instance().
        """
        self.dataset_path_colab: str = "/content/drive/MyDrive/DatenStudienarbeit/Daten/PodcastFillers/"
        self.colab_zip_path: str = "/content/drive/MyDrive/20230327_clip_wav_regenerate_2s_RandomDocumented.zip"
        # self.dataset_path_local: str = "Daten/PodcastFillers/"
        #self.dataset_path_local: str = "C:/Studienarbeit_Daten_Sicherungen/"
        self.dataset_path_local: str = "C:/DatenStudienarbeitNeu/PodcastFillers"
        # self.dataset_path_local: str = "C:/CallHomeDataset/"
        # self.dataset_path_local = "C:/CallHomeDataset/"
        # self.dataset_path_local: str = "G:/Meine Ablage/DatenStudienarbeit/Daten/PodcastFillers/"

        self.path_saved_models_colab: str = "/content/drive/MyDrive/DatenStudienarbeit/Daten/SavedModels"
        self.path_saved_models_local: str = "G:/Meine Ablage/DatenStudienarbeit/Daten/SavedModels"

        # self.master_csv_name: str = "CallHomeDataset_Original.csv"
        # self.master_csv_name: str = "CallHomeDataset.csv"
        # self.master_csv_name: str = "PodcastFillers.csv"
        #self.master_csv_name: str = "NewMasterCSV.csv"
        # self.master_csv_name: str = '20230412-151616_NewMasterCSV.csv'
        self.master_csv_name: str = "PodcastFillers.csv"

        self.duration: int = 2
        self.n_subfolders: int = 100
        self.dataset_path = self.dataset_path_local
        self.path_to_saved_models: str = self.path_saved_models_local

        self.clip_folder_name: str = "clip_wav_regenerate_{}s".format(self.duration)
        self.subset_files_folder: str = "{}s".format(self.duration)
        # self.subset_files_folder: str = "2s_checked"

        self.clip_folder_path: str = ""
        self.metadata_folder: str = ""
        self.master_csvfile: str = ""

        self.train_clips_folder: str = ""
        self.valid_clips_folder: str = ""
        self.test_clips_folder: str = ""
        self.extra_clips_folder: str = ""

        self.train_data_file: str = ""
        self.test_data_file: str = ""
        self.valid_data_file: str = ""
        self.extra_data_file: str = ""

        self.reload_all_paths()

    def reload_data_paths(self):
        self.clip_folder_path = os.path.join(self.dataset_path, "audio", self.clip_folder_name)
        self.train_clips_folder = "train"
        self.valid_clips_folder = "validation"
        self.test_clips_folder = "test"
        self.extra_clips_folder = "extra"

    def reload_metadata_paths(self):
        self.master_csvfile = os.path.join(self.metadata_folder, self.master_csv_name)

        self.train_data_file = os.path.join(self.metadata_folder, self.subset_files_folder, "training.txt")
        self.test_data_file = os.path.join(self.metadata_folder, self.subset_files_folder, "testing.txt")
        self.valid_data_file = os.path.join(self.metadata_folder, self.subset_files_folder, "validation.txt")
        self.extra_data_file = os.path.join(self.metadata_folder, self.subset_files_folder, "extra.txt")

    def reload_all_paths(self):
        self.metadata_folder = os.path.join(self.dataset_path, "metadata/")

        self.reload_data_paths()
        self.reload_metadata_paths()
