# main.py

import time
import torch
import numpy as np
from env import SUMOEnv
from agent import DQNAgent
from config import NUM_EPISODES, TARGET_UPDATE_FREQ

def main():
    # Use GPU if available; otherwise CPU.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Initialize environment and agent.
    env = SUMOEnv()
    env.start()
    agent = DQNAgent(device)

    global_step = 0

    for episode in range(NUM_EPISODES):
        state = env.reset()
        episode_reward = 0
        done = False

        while not done:
            # Select joint action using epsilon-greedy.
            action = agent.select_action(state)
            # Step the simulation for CONTROL_INTERVAL steps and accumulate reward.
            next_state, reward, done = env.step(action)
            episode_reward += reward

            # Add transition to replay buffer.
            agent.replay_buffer.add(state, action, reward, next_state, done)
            state = next_state

            # Train the agent at every step.
            loss = agent.train_step()
            global_step += 1

            # Update target network periodically.
            if global_step % TARGET_UPDATE_FREQ == 0:
                agent.update_target()
                print(f"Updated target network at step {global_step}")

        print(f"Episode {episode+1}: Total Reward = {episode_reward}")
    
    env.close()

if __name__ == '__main__':
    main()
