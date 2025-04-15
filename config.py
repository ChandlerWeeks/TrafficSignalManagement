# config.py

# SUMO simulation settings


# Full command for starting the simulation (executable required)
SUMO_START_CMD = [
    "C:/Program Files (x86)/Eclipse/Sumo/bin/sumo.exe",
    "-c", "data/simulation.sumocfg",
    "--step-length", "1"
]

# Command for reloading the simulation (executable is NOT allowed here)
SUMO_LOAD_CMD = [
    "-c", "data/simulation.sumocfg",
    "--step-length", "1"
]

# Other configuration parameters...

# Training settings
NUM_EPISODES = 1000        # number of training episodes
MAX_STEPS = 3600           # maximum simulation steps per episode (e.g., 1 hour)
CONTROL_INTERVAL = 5       # use the same action for 5 simulation steps to avoid rapid switching

#TODO: Fine tune these @Andrei
# DQN hyperparameters 
STATE_DIM = 1000           # total dimension of state vector (for example, 100 intersections * 10 features each)
ACTIONS_PER_INT = 4        # number of phases per intersection (assuming all intersections have the same count)
NUM_INTERSECTIONS = 100    # total intersections controlled
BATCH_SIZE = 64
GAMMA = 0.99
LEARNING_RATE = 1e-4
TARGET_UPDATE_FREQ = 1000  # steps
REPLAY_BUFFER_CAPACITY = 100000

# Epsilon parameters for epsilon-greedy exploration
EPS_START = 1.0
EPS_END = 0.05
EPS_DECAY = 5000  # decay steps
