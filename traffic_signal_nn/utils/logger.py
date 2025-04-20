# traffic_signal_nn/utils/logger.py

import os
from torch.utils.tensorboard import SummaryWriter

class Logger:
    """
    Wrap TF‑Board and CSV logging.
    """
    def __init__(self, log_dir):
        os.makedirs(log_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir)
        self.log_dir = log_dir

    def log_episode(self, episode, reward, agent):
        """
        Log scalar metrics at end of each episode.
        """
        self.writer.add_scalar("Reward/episode", reward, episode)
        # example: if agent tracks loss
        if hasattr(agent, "loss"):
            self.writer.add_scalar("Loss/td_loss", agent.loss, episode)

    def close(self):
        self.writer.close()
