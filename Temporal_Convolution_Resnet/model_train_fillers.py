import datetime
from datetime import time
import math
import multiprocessing
import os
import random
from typing import Sequence

import torch
import torch.nn as nn
import torchmetrics
from sklearn.model_selection import KFold

from torch.utils.data import Sampler
from torch.utils.data import DataLoader

from paths_for_running_in_colab import PathsForRunningInColabSingleton
from Inspection.ModelInspection import get_model_size
from Temporal_Convolution_Resnet.model_dataloader import PodcastFillersDataset, __classes__
from Temporal_Convolution_Resnet.model_layer import ModelBuilder
from Temporal_Convolution_Resnet.model_utils import *
from Temporal_Convolution_Resnet.options import OptionsInformation
from evaluation.evaluation_on_clips import evaluate, print_info
from utils.loss_info import LossInfos

save_directories_lock = multiprocessing.Lock()


class SubsetKFoldSampler(Sampler[int]):
    r"""Samples elements randomly from a given list of indices, without replacement.

    Args:
        indices (sequence): a sequence of indices
        generator (Generator): Generator used in sampling.
    """
    indices: Sequence[int]

    def __init__(self, indices: Sequence[int], generator=None) -> None:
        self.indices = indices
        self.generator = generator

    def __iter__(self):
        return (self.indices[i] for i in torch.randperm(len(self.indices), generator=self.generator))

    def __len__(self):
        return len(self.indices)


class SubsetRandomSampler(Sampler[int]):
    r"""Samples elements randomly from a given list of indices, without replacement.

       Args:
           indices (sequence): a sequence of indices
           generator (Generator): Generator used in sampling.
       """
    indices: Sequence[int]

    def __init__(self, length: int, indices: Sequence[int], generator=None) -> None:
        indices_temp = set()
        r = random.Random()
        while len(indices_temp) < length:
            x = r.randint(0, len(indices) - 1)
            indices_temp.add(indices[x])

        self.indices = list(indices_temp)
        self.generator = generator

    def __iter__(self):
        return (self.indices[i] for i in torch.randperm(len(self.indices), generator=self.generator))

    def __len__(self):
        return len(self.indices)


def __build_model_etc__(opt: OptionsInformation, device) -> (nn.Module, torch.optim.Optimizer, torch.optim.lr_scheduler):
    if opt.use_opt:
        if opt.reload_model:
            model = ModelBuilder.load_model(opt)
        else:
            model = ModelBuilder.build_model(opt).to(device)
    else:
        raise NotImplementedError

    print(">>>   Num of model parameters")
    model_size = get_model_size(model)
    print("Params: {} Size: {} MB".format(model_size[0], model_size[1]))

    params = model.parameters_own()

    if opt.optim == "adam":
        optimizer = torch.optim.Adam(params, lr=opt.lr)
    elif opt.optim == "sgd":
        optimizer = torch.optim.SGD(params, lr=opt.lr, momentum=opt.lr_momentum)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=opt.step, gamma=opt.scheduler_lr_gamma, last_epoch=-1)

    return model, optimizer, scheduler


class TRAINER(object):
    def __init__(self, opt: OptionsInformation, data_path, clip_folder_name, model_save_path: str,
                 device=None):
        self.epo = 0
        self.opt: OptionsInformation = opt
        self.epoch = opt.epoch
        self.lr = opt.lr
        self.batch = opt.batch
        self.step = opt.step
        self.model_save_path = os.path.join(model_save_path, "{}_{}".format(self.opt.save, self.opt.optim))

        global save_directories_lock
        save_directories_lock.acquire()
        paths_for_running_in_colab = PathsForRunningInColabSingleton.get_instance()

        _now = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        self.save_directory = os.path.join(self.model_save_path, _now)

        while os.path.isdir(self.save_directory):
            time.sleep(0.01)
            _now = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            self.save_directory = os.path.join(self.model_save_path, _now)

        os.makedirs(self.save_directory)
        save_directories_lock.release()

        self.results_file_path = os.path.join(self.save_directory,
                                              "{}_{}_TrainingResults.txt".format(_now, self.opt.save))

        # write options
        self.opt.write_options_to_json(os.path.join(self.save_directory, "options.json"))

        # train_filename   = readlines("./splits/{}.txt".format("train"))
        train_filename = read_lines(paths_for_running_in_colab.train_data_file)
        # train_dataset    = SpeechCommandDataset("./dataset", train_filename, True)
        self.train_dataset = PodcastFillersDataset(os.path.join(data_path, clip_folder_name), train_filename, True,
                                                   # train_dataset = CallHomeFillersDataset(os.path.join(data_path,
                                                   # clip_folder_name), train_filename, True,
                                                   paths_for_running_in_colab.train_clips_folder)

        valid_filename = read_lines(paths_for_running_in_colab.valid_data_file)
        self.valid_dataset = PodcastFillersDataset(os.path.join(data_path, clip_folder_name), valid_filename, False,
                                                   # valid_dataset = CallHomeFillersDataset(os.path.join(data_path,
                                                   # clip_folder_name), valid_filename, False,
                                                   paths_for_running_in_colab.valid_clips_folder)
        self.loss_info = {
            "train": LossInfos("train"),
            "valid": LossInfos("valid_regular")
        }

        if opt.retraining:

            test_filename_corrected = read_lines(paths_for_running_in_colab.test_data_file)
            self.test_dataset_corrected = PodcastFillersDataset(os.path.join(data_path, clip_folder_name), test_filename_corrected, False,
                                                                paths_for_running_in_colab.test_clips_folder)
            try:
                test_filename_all = read_lines(os.path.join(paths_for_running_in_colab.metadata_folder, "2s", "testing.txt"))
                self.test_dataset_original = PodcastFillersDataset(os.path.join(data_path, clip_folder_name), test_filename_all,
                                                                   False,
                                                                   paths_for_running_in_colab.test_clips_folder)

                valid_filename_all = read_lines(os.path.join(paths_for_running_in_colab.metadata_folder, "2s", "validation.txt"))
                self.valid_dataset_original = PodcastFillersDataset(os.path.join(data_path, clip_folder_name), valid_filename_all, False,
                                                                    paths_for_running_in_colab.valid_clips_folder)

                self.loss_info["test_original"] = LossInfos("test_original")
                self.loss_info["valid_original"] = LossInfos("valid_original")

            except:
                print("No original data given. Only using retraining-data at location: {}".format(
                    os.path.dirname(paths_for_running_in_colab.test_data_file)))

            self.loss_info["test_corrected"] = LossInfos("test_corrected")

        self.classes = __classes__

        self.info_templet = "type: {} loss {:0.3f}  Acc {:0.2f} AccSep {} precision {} recall {} f1 {}"

        if not device:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.model: nn.Module
        self.optimizer: torch.optim.optimizer
        self.criterion = nn.CrossEntropyLoss().to(self.device)

    def model_train(self):

        valid_dataloader = DataLoader(
            self.valid_dataset, batch_size=self.batch, shuffle=True, drop_last=False)
        valid_length = len(valid_dataloader)

        if self.opt.retraining:
            test_corrected_dataloader = DataLoader(
                self.test_dataset_corrected, batch_size=self.batch, shuffle=True, drop_last=False)
            self.loss_info["test_corrected"].length = len(test_corrected_dataloader)

            if self.original_data_available():
                test_original_dataloader = DataLoader(
                    self.test_dataset_original, batch_size=self.batch, shuffle=True, drop_last=False)
                self.loss_info["test_original"].length = len(test_original_dataloader)

                valid_original_dataloader = DataLoader(
                    self.valid_dataset_original, batch_size=self.batch, shuffle=True, drop_last=False)
                self.loss_info["valid_original"].length = len(valid_original_dataloader)

        train_dataloaders = list()
        if self.opt.k_fold > 1:
            fold = KFold(self.opt.k_fold, shuffle=True, random_state=4792)
            for fold, (tr_idx, val_idx) in enumerate(fold.split(self.train_dataset)):
                train_dataloaders.append(DataLoader(
                    self.train_dataset, batch_size=self.batch, drop_last=True,
                    sampler=SubsetKFoldSampler(tr_idx)))
        elif self.opt.train_data_size and self.opt.train_data_size > 0:
            for j in range(math.ceil(len(self.train_dataset) / self.opt.train_data_size * 1.5)):
                train_dataloaders.append(DataLoader(
                    self.train_dataset, batch_size=self.batch, drop_last=True,
                    sampler=SubsetRandomSampler(self.opt.train_data_size, [i for i in range(len(self.train_dataset))])))
        else:
            train_dataloaders.append(DataLoader(self.train_dataset, shuffle=True,
                                                batch_size=self.batch, drop_last=True))

        for i in range(len(train_dataloaders)):

            (self.model, self.optimizer, self.scheduler) = __build_model_etc__(self.opt, self.device)

            train_dataloader = train_dataloaders[i]
            if os.path.basename(self.save_directory) == str(i - 1):
                self.save_directory = os.path.dirname(self.save_directory)

            self.save_directory = os.path.join(self.save_directory, str(i))
            os.mkdir(self.save_directory)

            train_length = len(train_dataloader)
            print(">>>   Train length: {}   Valid length: {}".format(train_length, valid_length))

            precision = torchmetrics.classification.MulticlassPrecision(len(self.classes), average=None,
                                                                        validate_args=False).to(self.device)
            recall = torchmetrics.classification.MulticlassRecall(len(self.classes), average=None,
                                                                  validate_args=False).to(self.device)
            f1 = torchmetrics.classification.MulticlassF1Score(len(self.classes), average=None,
                                                               validate_args=False).to(self.device)
            accuracy = torchmetrics.classification.MulticlassAccuracy(len(self.classes), average="micro",
                                                                      validate_args=False).to(self.device)
            accuracy_seperated = torchmetrics.classification.MulticlassAccuracy(len(self.classes), average=None,
                                                                                validate_args=False).to(self.device)

            if self.opt.retraining:
                self.loss_info["test_corrected"] = evaluate(test_corrected_dataloader, self.model, self.classes, self.loss_info["test_corrected"], self.criterion, self.device)
                if self.original_data_available():
                    self.loss_info["test_original"] = evaluate(test_original_dataloader, self.model, self.classes, self.loss_info["test_original"], self.criterion, self.device)
                    self.loss_info["valid_original"] = evaluate(valid_original_dataloader, self.model, self.classes, self.loss_info["valid_original"], self.criterion, self.device)
                self.epo = -1
                print_info(self.loss_info, self.info_templet, self.results_file_path, self.epo)

            for self.epo in range(self.epoch):
                for k in self.loss_info:
                    self.loss_info[k].set_to_zeros()

                self.model.train()

                predict_total = torch.Tensor().to(self.device)
                labels_total = torch.Tensor().to(self.device)
                t0 = datetime.datetime.now()

                for batch_idx, (waveform, labels) in enumerate(train_dataloader):
                    loss_info_name = "train"

                    waveform, labels = waveform.to(self.device), labels.to(self.device)
                    logits = self.model(waveform)

                    self.optimizer.zero_grad()
                    loss = self.criterion(logits, labels)
                    loss.backward()
                    self.optimizer.step()

                    self.loss_info[loss_info_name].loss += loss.item() / train_length
                    _, predict = torch.max(logits.data, 1)

                    predict_total = torch.cat((predict_total, predict))
                    labels_total = torch.cat((labels_total, labels))

                self.loss_info[loss_info_name].total += labels_total.size(0)
                self.loss_info[loss_info_name].correct += (predict_total == labels_total).sum().item()
                self.loss_info[loss_info_name].accuracy = accuracy(predict_total, labels_total)
                self.loss_info[loss_info_name].accuracy_seperated = accuracy_seperated(predict_total, labels_total)
                self.loss_info[loss_info_name].precision = precision(predict_total, labels_total)
                self.loss_info[loss_info_name].recall = recall(predict_total, labels_total)
                self.loss_info[loss_info_name].f1 = f1(predict_total, labels_total)

                if self.opt.retraining:
                    self.loss_info["test_corrected"] = evaluate(test_corrected_dataloader, self.model, self.classes, self.loss_info["test_corrected"], self.criterion, self.device)
                    if self.original_data_available():
                        self.loss_info["test_original"] = evaluate(test_original_dataloader, self.model, self.classes, self.loss_info["test_original"], self.criterion, self.device)
                        self.loss_info["valid_original"] = evaluate(valid_original_dataloader, self.model, self.classes, self.loss_info["valid_original"], self.criterion, self.device)

                self.loss_info["valid"] = evaluate(valid_dataloader, self.model, self.classes, self.loss_info["valid"], self.criterion, self.device)

                self.scheduler.step()
                print_info(self.loss_info, self.info_templet, self.results_file_path, self.epo)
                if self.opt.retraining and self.original_data_available():
                    model_save(self.loss_info["valid_original"], self.epo, self.opt.epoch, self.opt.save_freq, self.save_directory, self.model, self.opt.model)
                else:
                    model_save(self.loss_info["valid"], self.epo, self.opt.epoch, self.opt.save_freq, self.save_directory, self.model, self.opt.model)

                print("Duration: {}".format(datetime.datetime.now() - t0))

    def original_data_available(self):
        return "test_original" in self.loss_info and "valid_original" in self.loss_inf


def model_save(loss_info: LossInfos, epo, num_epoch, save_freq, save_directory, model, model_name):

    prefix: str = None

    if loss_info.accuracy >= 0.86:
        prefix = "best"
    if (epo + 1) % save_freq == 0:
        prefix = "model"
    if (epo + 1) == num_epoch:
        prefix = "last"

    if prefix:
        s = "{}_{}".format(prefix, __calc_save_model_name__(epo, loss_info, model_name))
        torch.save(model.state_dict(), os.path.join(save_directory, s))


def __calc_save_model_name__(epo, loss_info: LossInfos, model_name: str) -> str:
    s = model_name + str(epo + 1) + "_" + str(loss_info.accuracy) + ".pt"
    s = s.replace(":", "_")
    return s


def run_training(data_path: str, clip_folder_name: str, model_save_path: str, opt: OptionsInformation = None):
    info: str = "Training model"
    print(info)

    TRAINER(opt, data_path, clip_folder_name, model_save_path).model_train()

    print("-DONE: {}".format(info))
