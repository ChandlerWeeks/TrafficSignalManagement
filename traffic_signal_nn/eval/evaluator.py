import os
import traci
import sumolib
import statistics
import sys

# Add the root directory (TrafficSignalManagement) to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from traffic_signal_nn.agents.policies import Agent
from traffic_signal_nn.env.multi_env import CityEnv
from traffic_signal_nn.utils.config_parser import load_config


class Evaluator:
    def __init__(self, cfg_path):
        self.cfg = load_config(cfg_path)
        self.env_cfg = self.cfg["ENV"]
        self.eval_cfg = self.cfg["EVAL"]
        self.agent_cfg = self.cfg["AGENT"]
        self.model_dir = self.cfg["LOG"]["LOG_DIR"]

        self.env = CityEnv(self.env_cfg)
        self.agent = Agent(
            self.env.observation_space,
            self.env.action_space,
            self.agent_cfg
        )
        self.load_models()

    def load_models(self):
        print("Loading trained models...")
        for i, ag in enumerate(self.agent.agents):
            path = os.path.join(self.model_dir, f"multi_dqn_tls{i}.pth")
            if os.path.exists(path):
                ag.load(path)
            else:
                print(f"⚠️ Missing model: {path}")

    def evaluate(self):
        print("Running evaluation...")
        state, done = self.env.reset(), False

        max_steps = 2400 # 1 hour of simulation time
        steps = 0

        while not done and steps < max_steps:
            actions = self.agent.select_action(state, evaluate=True)
            state, _, done, _ = self.env.step(actions)
            steps += 2

        # --- Extract SUMO Metrics ---
        tripinfos = traci.simulation.getArrivedIDList()
        durations = []
        delays = []
        waiting_times = []

        for vid in tripinfos:
            try:
                durations.append(traci.vehicle.getAccumulatedWaitingTime(vid))
                delays.append(traci.vehicle.getTimeLoss(vid))
                waiting_times.append(traci.vehicle.getWaitingTime(vid))
            except traci.exceptions.TraCIException:
                continue

        queue_lengths = [
            traci.lane.getLastStepHaltingNumber(lane)
            for lane in traci.lane.getIDList()
        ]

        print("\n📊 Evaluation Results:")
        print(f"➤ Average Queue Length: {statistics.mean(queue_lengths):.2f}")
        print(f"➤ Max Queue Length:     {max(queue_lengths)}")
        print(f"➤ Avg Waiting Time:     {statistics.mean(waiting_times) if waiting_times else 0:.2f}s")
        print(f"➤ Avg Delay:            {statistics.mean(delays) if delays else 0:.2f}s")
        print(f"➤ Avg Trip Time:        {statistics.mean(durations) if durations else 0:.2f}s")

        self.env.close()


# ✨ Entry point
if __name__ == "__main__":
    cfg_path = os.path.join("traffic_signal_nn", "config", "config_dqn_city.ini")
    evaluator = Evaluator(cfg_path)
    evaluator.evaluate()
