# agent.py

import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from model import CityDQN
from replay_buffer import ReplayBuffer
from config import (STATE_DIM, ACTIONS_PER_INT, NUM_INTERSECTIONS, 
                    GAMMA, LEARNING_RATE, TARGET_UPDATE_FREQ, 
                    REPLAY_BUFFER_CAPACITY, BATCH_SIZE, EPS_START, EPS_END, EPS_DECAY)

class DQNAgent:
    def __init__(self, device):
        self.device = device
        self.state_dim = STATE_DIM
        self.actions_per_int = ACTIONS_PER_INT
        self.num_intersections = NUM_INTERSECTIONS

        self.policy_net = CityDQN(self.state_dim, self.actions_per_int, self.num_intersections).to(self.device)
        self.target_net = CityDQN(self.state_dim, self.actions_per_int, self.num_intersections).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=LEARNING_RATE)
        self.replay_buffer = ReplayBuffer(REPLAY_BUFFER_CAPACITY)
        self.steps_done = 0

    def select_action(self, state):
        """
        Select a joint action (a list with one action per intersection) using epsilon-greedy strategy.
        """
        sample = np.random.rand()
        eps_threshold = EPS_END + (EPS_START - EPS_END) * np.exp(-1. * self.steps_done / EPS_DECAY)
        self.steps_done += 1
        if sample < eps_threshold:
            # Random action for each intersection
            return [np.random.randint(self.actions_per_int) for _ in range(self.num_intersections)]
        else:
            # Use policy_net
            with torch.no_grad():
                state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
                q_values = self.policy_net(state_tensor)
                # Reshape to (num_intersections, actions_per_int)
                q_values = q_values.view(self.num_intersections, self.actions_per_int)
                # Choose the best action per intersection
                actions = q_values.argmax(dim=1)
                return actions.cpu().numpy().tolist()

    def train_step(self):
        if len(self.replay_buffer) < BATCH_SIZE:
            return

        batch = self.replay_buffer.sample(BATCH_SIZE)
        states, actions, rewards, next_states, dones = map(np.array, zip(*batch))

        states = torch.tensor(states, dtype=torch.float32).to(self.device)
        next_states = torch.tensor(next_states, dtype=torch.float32).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        dones = torch.tensor(dones, dtype=torch.float32).to(self.device)
        actions = torch.tensor(actions, dtype=torch.long).to(self.device)  # shape: (batch, num_intersections)

        # Compute current Q values:
        q_values = self.policy_net(states)
        # Reshape into (batch, num_intersections, actions_per_int)
        q_values = q_values.view(BATCH_SIZE, self.num_intersections, self.actions_per_int)

        # Gather Q-values for the taken actions:
        q_pred = torch.zeros(BATCH_SIZE).to(self.device)
        for i in range(self.num_intersections):
            q_pred += q_values[:, i, :].gather(1, actions[:, i].unsqueeze(1)).squeeze(1)

        # Compute next state Q values:
        with torch.no_grad():
            q_next = self.target_net(next_states)
            q_next = q_next.view(BATCH_SIZE, self.num_intersections, self.actions_per_int)
            max_q_next = torch.zeros(BATCH_SIZE).to(self.device)
            for i in range(self.num_intersections):
                # Select maximum Q-value for each intersection for each sample
                max_q_next += q_next[:, i, :].max(dim=1)[0] * (1 - dones)
        
        # Compute Bellman target:
        target = rewards + GAMMA * max_q_next

        # Compute loss (mean squared error)
        loss = F.mse_loss(q_pred, target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def update_target(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())
