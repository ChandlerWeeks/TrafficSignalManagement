#!/usr/bin/env python3
import os
import sys
import argparse
import traci

from traffic_signal_nn.env.multi_env      import CityEnv
from traffic_signal_nn.utils.config_parser import load_config
from traffic_signal_nn.agents.policies     import Agent

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Run trained traffic signal control model")
    parser.add_argument(
        '--config', '-c',
        required=True,
        help='Path to the config file (e.g. config_dqn_city.ini)'
    )
    parser.add_argument(
        '--max-steps', '-m',
        type=int,
        default=None,
        help='Maximum number of control steps (default: run until simulation ends)'
    )
    args = parser.parse_args()

    # Load configuration
    cfg = load_config(args.config)
    env_cfg   = cfg['ENV']
    agent_cfg = cfg['AGENT']
    model_dir = cfg['LOG']['LOG_DIR']

    # Initialize SUMO environment
    env = CityEnv(env_cfg)

    # Initialize multi-agent
    agent = Agent(env.observation_space, env.action_space, agent_cfg)

    # Load trained models
    print("Loading trained models...")
    for idx, ag in enumerate(agent.agents):
        model_path = os.path.join(model_dir, f"multi_dqn_tls{idx}.pth")
        if os.path.exists(model_path):
            ag.load(model_path)
            print(f"  ✓ Loaded model for TLS #{idx}: {model_path}")
        else:
            print(f"  ⚠️  Missing model for TLS #{idx}: {model_path}")

    # Run simulation loop
    print("Starting simulation...")
    state, done = env.reset(), False
    step = 0
    max_steps = args.max_steps

    try:
        while not done and (max_steps is None or step < max_steps):
            actions = agent.select_action(state, evaluate=True)
            state, _, done, _ = env.step(actions)
            step += 1
        print(f"Simulation finished after {step} control steps.")
    finally:
        env.close()
        print("SUMO environment closed.")


if __name__ == '__main__':
    main()
