# model.py

import torch
import torch.nn as nn

class CityDQN(nn.Module):
    def __init__(self, state_dim, actions_per_int, num_intersections):
        super(CityDQN, self).__init__()
        self.state_dim = state_dim
        self.actions_per_int = actions_per_int
        self.num_intersections = num_intersections
        # The total output is factorized: each intersection gets its own set of Q-values.
        self.action_dim = actions_per_int * num_intersections

        # A simple network structure – adjust number of neurons and layers as needed.
        self.fc1 = nn.Linear(state_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.out = nn.Linear(256, self.action_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.out(x)  # flat vector; later reshaped to (num_intersections, actions_per_int)
