import torch.nn as nn


class RegimeMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dims=None, num_classes: int = 3):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [64, 32]

        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.2))
            prev = h

        layers.append(nn.Linear(prev, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)
