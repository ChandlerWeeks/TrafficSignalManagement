# traffic_signal_nn/agents/policies.py

import random
import torch
import torch.optim as optim
import torch.nn.functional as F
from .models import build_mlp
from .replay_buffer import ReplayBuffer

def get_agent_class(name):
    name = name.lower()
    if name == "dqn":
        return DQNAgent
    raise ValueError(f"Unknown agent '{name}'")

class DQNAgent:
    """
    Standard (single‑agent) DQN.
    """
    def __init__(self, state_dim, action_dim, cfg):
        self.state_dim  = state_dim
        self.action_dim = action_dim
        self.lr         = cfg["LR"]
        self.gamma      = cfg["GAMMA"]
        self.epsilon    = cfg["EPS_START"]
        self.eps_end    = cfg["EPS_END"]
        self.eps_decay  = cfg["EPS_DECAY"]

        hidden = list(map(int, cfg["HIDDEN"].split(",")))
        self.policy_net = build_mlp(state_dim, action_dim, hidden)
        self.target_net = build_mlp(state_dim, action_dim, hidden)
        self.target_net.load_state_dict(self.policy_net.state_dict())

        self.optimizer   = optim.Adam(self.policy_net.parameters(), lr=self.lr)
        self.replay_buf  = ReplayBuffer(int(cfg["BUFFER_SIZE"]))
        self.batch_size  = cfg["BATCH_SIZE"]
        self.update_freq = cfg.get("TARGET_UPDATE_FREQ", 10)
        self.step_count  = 0
        self.loss        = 0.0

    def select_action(self, state):
        if random.random() < self.epsilon:
            return random.randrange(self.action_dim)
        with torch.no_grad():
            qvals = self.policy_net(torch.tensor(state, dtype=torch.float32))
            return int(qvals.argmax().item())

    def remember(self, s, a, r, ns, done):
        self.replay_buf.add(s, a, r, ns, done)

    def learn(self):
        if len(self.replay_buf) < self.batch_size:
            return

        states, actions, rewards, next_states, dones = self.replay_buf.sample(self.batch_size)
        s  = torch.tensor(states, dtype=torch.float32)
        a  = torch.tensor(actions, dtype=torch.long).unsqueeze(1)
        r  = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1)
        ns = torch.tensor(next_states, dtype=torch.float32)
        d  = torch.tensor(dones, dtype=torch.float32).unsqueeze(1)

        q_pred = self.policy_net(s).gather(1, a)
        with torch.no_grad():
            q_next  = self.target_net(ns).max(1)[0].unsqueeze(1)
            q_target= r + (1 - d) * self.gamma * q_next

        self.loss = F.mse_loss(q_pred, q_target)
        self.optimizer.zero_grad()
        self.loss.backward()
        self.optimizer.step()

        self.step_count += 1
        self.epsilon = max(self.eps_end, self.epsilon * self.eps_decay)
        if self.step_count % self.update_freq == 0:
            self.update_target_network()

    def update_target_network(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def save(self, path):
        torch.save(self.policy_net.state_dict(), path)

    def load(self, path):
        self.policy_net.load_state_dict(torch.load(path))
        self.target_net.load_state_dict(self.policy_net.state_dict())
