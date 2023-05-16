import datetime
import os
from enum import Enum

import torch
import torchmetrics
from torch import nn as nn
from torch.utils.data import DataLoader

from paths_for_running_in_colab import PathsForRunningInColabSingleton
from Inspection.ModelInspection import get_model_size
from Temporal_Convolution_Resnet.model_dataloader import PodcastFillersDataset, __classes__
from Temporal_Convolution_Resnet.model_layer import ModelBuilder
from Temporal_Convolution_Resnet.model_utils import read_lines
from Temporal_Convolution_Resnet.options import OptionsInformation
from utils.loss_info import LossInfos


class DatasetSubset(Enum):
    TEST = 0
    VALID = 1


def evaluate_model(model: nn.Module, opt: OptionsInformation | list[OptionsInformation], data_path: str, clip_folder_name: str, subset: DatasetSubset=DatasetSubset.TEST):

    paths_for_running_in_colab = PathsForRunningInColabSingleton.get_instance()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if not model:
        use_opt: bool = True

        if isinstance(opt, list):
            for o in opt:
                if not o.use_opt or not o.reload_model:
                    use_opt = False
                    break
            batch = opt[0].batch
        else:
            use_opt = opt.use_opt and opt.reload_model
            batch = opt.batch

        if use_opt:
            model = ModelBuilder.load_model(opt)
        else:
            raise NotImplementedError

    if subset == DatasetSubset.TEST:
        data_file = paths_for_running_in_colab.test_data_file
        clips_folder = paths_for_running_in_colab.test_clips_folder
    elif subset == DatasetSubset.VALID:
        data_file = paths_for_running_in_colab.valid_data_file
        clips_folder = paths_for_running_in_colab.valid_clips_folder
    else:
        raise NotImplementedError

    test_filename = read_lines(data_file)
    test_dataset = PodcastFillersDataset(os.path.join(data_path, clip_folder_name), test_filename, False,
                                         clips_folder)

    classes = __classes__

    templet = "type: {} loss {:0.3f}  Acc {:0.2f} AccSep {} precision {} recall {} f1 {}"

    test_dataloader = DataLoader(
        test_dataset, batch_size=batch, shuffle=True, drop_last=False)

    print(">>>   Test length: {}  ".format(len(test_dataloader)))

    print(">>>   Num of model parameters")
    model_size = get_model_size(model)
    print("Params: {} Size: {} MB".format(model_size[0], model_size[1]))

    criterion = nn.CrossEntropyLoss().to(device)
    loss_info = {
        "test": LossInfos("test")
    }

    t0 = datetime.datetime.now()

    loss_info["test"] = evaluate(test_dataloader, model, classes, loss_info["test"], criterion, device)
    print_info(loss_info, templet)

    print("Duration: {}".format(datetime.datetime.now() - t0))


def evaluate(dataloader, model, classes, loss_info: LossInfos, criterion, device) -> LossInfos:

    if len(dataloader) > 0:
        precision = torchmetrics.classification.MulticlassPrecision(len(classes), average=None,
                                                                    validate_args=False).to(device)
        recall = torchmetrics.classification.MulticlassRecall(len(classes), average=None,
                                                              validate_args=False).to(device)
        f1 = torchmetrics.classification.MulticlassF1Score(len(classes), average=None,
                                                           validate_args=False).to(device)
        accuracy = torchmetrics.classification.MulticlassAccuracy(len(classes), average="micro",
                                                                  validate_args=False).to(device)
        accuracy_seperated = torchmetrics.classification.MulticlassAccuracy(len(classes), average=None,
                                                                            validate_args=False).to(device)

        model.eval()
        predict_total = torch.Tensor().to(device)
        labels_total = torch.Tensor().to(device)

        for batch_idx, (waveform, labels) in enumerate(dataloader):
            with torch.no_grad():
                waveform, labels = waveform.to(device), labels.to(device)
                logits = model(waveform)
                loss = criterion(logits, labels)

                loss_info.loss += loss.item() / len(dataloader)
                _, predict = torch.max(logits.data, 1)

                predict_total = torch.cat((predict_total, predict))
                labels_total = torch.cat((labels_total, labels))

        loss_info.total += labels_total.size(0)
        loss_info.correct += (predict_total == labels_total).sum().item()
        loss_info.accuracy = accuracy(predict_total, labels_total)
        loss_info.accuracy_seperated = accuracy_seperated(predict_total, labels_total)
        loss_info.precision = precision(predict_total, labels_total)
        loss_info.recall = recall(predict_total, labels_total)
        loss_info.f1 = f1(predict_total, labels_total)

        return loss_info


def print_info(loss_info: dict[LossInfos], info_templet: str, results_file_path: str="", epo: int=0):
    info_strs = []
    for lf in loss_info:
        li: LossInfos = loss_info[lf]
        info_strs.append(info_templet.format(li.name, li.loss, li.accuracy, li.accuracy_seperated, li.precision, li.recall, li.f1))

    info: str = "Epoch: {} ".format(epo + 1)
    for i in info_strs:
        info += "\n{}".format(i)

    print(info)
    if results_file_path:
        with open(results_file_path, "a") as res_file:
            res_file.write(info + "\n")
