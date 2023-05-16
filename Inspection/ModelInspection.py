import torch.nn as nn


def get_model_size(model: nn.Module) -> (int, float):
    """

    :param model:
    :return: size of model in MB
    """
    param_size = 0
    n_params = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
        n_params += param.nelement()

    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    size_all_mb = (param_size + buffer_size) / 1024 ** 2
    return n_params, size_all_mb


def are_models_equal(model1: nn.Module, model2: nn.Module) -> bool:
    if str(model1.state_dict()) != str(model2.state_dict()):
        return False

    for p1, p2 in zip(model1.parameters(), model2.parameters()):
        if p1.data.ne(p2.data).sum() > 0:
            return False

    return True

