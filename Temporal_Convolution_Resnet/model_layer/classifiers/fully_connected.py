import torch.nn as nn


class FNN(nn.Module):
    def __init__(self, layer_sizes, activation, residual: [bool] = None, device=None):
        super(FNN, self).__init__()

        if not residual:
            residual = []
            for i in range(len(activation)):
                residual.append(False)

        layers = []
        for in_size, out_size, activation_fnc, res in zip(layer_sizes[0:-1], layer_sizes[1:], activation, residual):

            if res:
                layers.append(ResFNN(in_size, out_size, device))
            else:
                layers.append(nn.Linear(in_size, out_size, device))

            if activation_fnc == "relu":
                layers.append(nn.ReLU())
            elif activation_fnc == "leaky_relu":
                layers.append(nn.LeakyReLU(0.5))
            elif activation_fnc == "tanh":
                layers.append(nn.Tanh())
            elif activation_fnc == "none":
                layers.append(nn.Sequential())
            else:
                layers.append(nn.Sigmoid())

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        out = self.layers(x)
        return out


class ResFNN(nn.Module):
    def __init__(self, in_features, out_features, device=None):
        super(ResFNN, self).__init__()

        self.res = nn.Sequential()
        if not in_features == out_features:
            self.res.append(nn.Linear(in_features, out_features, device))
            self.res.append(nn.LeakyReLU(0.8))

        self.fc1 = nn.Linear(in_features, in_features, device)
        self.act1 = nn.LeakyReLU(0.5)
        self.fc2 = nn.Linear(in_features, out_features, device)
        self.act2 = nn.Tanh()

    def forward(self, x):
        res = self.res(x)

        out = self.fc1(x)
        out = self.act1(out)
        out = self.fc2(out)

        out = self.act2(res + out)
        return out



