# traffic_signal_nn/agents/models.py

import torch.nn as nn

def build_mlp(input_dim, output_dim, hidden_units):
    """
    Construct a plain MLP: FC → ReLU → … → FC.
    """
    layers = []
    prev = input_dim
    for h in hidden_units:
        layers.append(nn.Linear(prev, h))
        layers.append(nn.ReLU())
        prev = h
    layers.append(nn.Linear(prev, output_dim))
    return nn.Sequential(*layers)
