import os
from torch.utils.tensorboard import SummaryWriter

class Logger:
    def __init__(self, log_dir):
        os.makedirs(log_dir, exist_ok=True)
        self.writer  = SummaryWriter(log_dir)
        self.log_dir = log_dir

    def log_episode(self, ep: int, reward: float, agent=None):
        """
        Log one episode.
        If a third argument (`agent`) is supplied we can also log per‑agent
        statistics, otherwise we ignore it.
        """
        self.writer.add_scalar('Reward/Episode', reward, ep)

        # optional per‑TLS loss/epsilon etc.
        if agent is not None and hasattr(agent, "agents"):
            for i, ag in enumerate(agent.agents):
                if hasattr(ag, "eps"):
                    self.writer.add_scalar(f'TLS{i}/Epsilon', ag.eps, ep)
                if hasattr(ag, "step_n"):
                    self.writer.add_scalar(f'TLS{i}/Steps', ag.step_n, ep)

    def close(self):
        self.writer.close()

