# traffic_signal_nn/agents/policies.py
import random
import torch
import torch.nn.functional as F
import torch.optim as optim

from agents.models        import build_mlp
from agents.replay_buffer import ReplayBuffer


# ──────────────────────────────────────────────────────────────────────
class DQNAgent:
    """One‑intersection DQN."""
    def __init__(self, state_dim, action_dim, cfg):
        hidden = list(map(int, cfg.get('HIDDEN', '64,64').split(',')))
        self.net    = build_mlp(state_dim, action_dim, hidden)
        self.target = build_mlp(state_dim, action_dim, hidden)
        self.target.load_state_dict(self.net.state_dict())

        self.opt       = optim.Adam(self.net.parameters(),
                                    lr=cfg.get('LR', 1e-3))
        self.gamma     = cfg.get('GAMMA', 0.99)
        self.eps       = cfg.get('EPS_START', 1.0)
        self.eps_end   = cfg.get('EPS_END', 0.05)
        self.eps_decay = cfg.get('EPS_DECAY', 0.995)

        self.buf       = ReplayBuffer(cfg.get('BUFFER_SIZE', 10000))
        self.batch     = cfg.get('BATCH_SIZE', 32)
        self.update_fr = cfg.get('TARGET_UPDATE_FREQ', 10)
        self.step_n    = 0

    # --------------- public ----------------
    def select_action(self, state, greedy=False):
        if (not greedy) and random.random() < self.eps:
            return random.randrange(self.net[-1].out_features)
        with torch.no_grad():
            q = self.net(torch.tensor(state, dtype=torch.float32))
            return int(q.argmax())

    def remember(self, s, a, r, ns, done):
        self.buf.add(s, a, r, ns, done)

    def learn(self):
        if len(self.buf) < self.batch:
            return
        s, a, r, ns, d = self.buf.sample(self.batch)
        s  = torch.tensor(s,  dtype=torch.float32)
        a  = torch.tensor(a,  dtype=torch.long).unsqueeze(1)
        r  = torch.tensor(r,  dtype=torch.float32).unsqueeze(1)
        ns = torch.tensor(ns, dtype=torch.float32)
        d  = torch.tensor(d,  dtype=torch.float32).unsqueeze(1)

        q_pred = self.net(s).gather(1, a)
        with torch.no_grad():
            q_next = self.target(ns).max(1)[0].unsqueeze(1)
            q_tgt  = r + (1 - d) * self.gamma * q_next

        loss = F.mse_loss(q_pred, q_tgt)
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()

        # housekeeping
        self.eps = max(self.eps_end, self.eps * self.eps_decay)
        self.step_n += 1
        if self.step_n % self.update_fr == 0:
            self.target.load_state_dict(self.net.state_dict())

    # --------------- I/O -------------------
    def save(self, path):
        torch.save(self.net.state_dict(), path)

    def load(self, path):
        self.net.load_state_dict(torch.load(path))
        self.target.load_state_dict(self.net.state_dict())


# ──────────────────────────────────────────────────────────────────────
class MultiDQNAgent:
    """
    Lightweight wrapper: one DQNAgent per TLS.
    state_dims / action_dims are *lists* with length N_tls.
    """
    def __init__(self, state_dims, action_dims, cfg):
        assert len(state_dims) == len(action_dims)
        self.agents = [DQNAgent(sd, ad, cfg)
                       for sd, ad in zip(state_dims, action_dims)]

    # ----- RL façade -----
    def select_action(self, states, evaluate=False):
        return [ag.select_action(s, greedy=evaluate)
                for ag, s in zip(self.agents, states)]

    def remember(self, ss, aa, rr, nss, dones):
        for ag, s, a, r, ns, d in zip(self.agents, ss, aa, rr, nss, dones):
            ag.remember(s, a, r, ns, d)

    def learn(self):
        for ag in self.agents:
            ag.learn()

    # ----- persistence -----
    def save(self, prefix):
        for i, ag in enumerate(self.agents):
            ag.save(f"{prefix}_tls{i}.pth")

    def load(self, prefix):
        for i, ag in enumerate(self.agents):
            ag.load(f"{prefix}_tls{i}.pth")


# Allow:  from agents.policies import Agent
Agent = MultiDQNAgent
