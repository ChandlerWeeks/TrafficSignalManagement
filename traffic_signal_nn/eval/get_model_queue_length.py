#!/usr/bin/env python3
"""
get_model_queue_length.py

Runs a trained multi-DQN traffic signal controller in SUMO, collects the queue length
(number of stopped vehicles at controlled lanes) at each control step, and writes the
results to a CSV file.
"""
import os
import sys
import csv
import argparse
import statistics
import traci

# add project root so traffic_signal_nn package can be imported
sys.path.insert(0, os.path.abspath(os.path.join(__file__, '..', '..', '..')))

from traffic_signal_nn.env.multi_env      import CityEnv
from traffic_signal_nn.utils.config_parser import load_config
from traffic_signal_nn.agents.policies     import Agent


def main():
    p = argparse.ArgumentParser(description="Run model and log queue lengths per control step.")
    p.add_argument('--config', required=True,
                   help="path to traffic_signal_nn config .ini file")
    p.add_argument('--output', default="./data/output/queue_length.csv",
                   help="CSV file to write step,queue_length")
    p.add_argument('--steps', type=int, default=None,
                   help="maximum number of control steps to run")
    args = p.parse_args()

    # load config
    cfg = load_config(args.config)
    env_cfg = cfg['ENV']
    agent_cfg = cfg.get('AGENT', {})
    model_dir = cfg['LOG']['LOG_DIR']

    # create environment and agent
    env = CityEnv(env_cfg)
    agent = Agent(env.observation_space, env.action_space, agent_cfg)

    # load weights
    print("Loading trained models …")
    for i, ag in enumerate(agent.agents):
        path = os.path.join(model_dir, f"multi_dqn_tls{i}.pth")
        if os.path.exists(path):
            ag.load(path)
        else:
            print(f"⚠️ Missing model: {path}")

    # run simulation
    print("Starting evaluation simulation …")
    state, done = env.reset(), False
    step = 0

    with open(args.output, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["step", "queue_length"])

        while not done and (args.steps is None or step < args.steps):
            actions = agent.select_action(state, evaluate=True)
            state, _, done, _ = env.step(actions)

            # Compute queue-length across all controlled lanes:
            # OPTION 2: use env.get_state()
            obs = env.get_state()
            queue_len = sum(sum(group) for group in obs)

            writer.writerow([step, queue_len])
            step += 1

    env.close()
    print(f"✅ Queue length data written to {args.output}")



if __name__ == '__main__':
    main()
