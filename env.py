# env.py

import traci
import random
import numpy as np
from config import SUMO_START_CMD, SUMO_LOAD_CMD, CONTROL_INTERVAL, MAX_STEPS, FEATURES_PER_INT

class SUMOEnv:
    def __init__(self):
        # Use the start command for starting SUMO (which includes the full executable path)
        # and the load command for reloading.
        self.start_cmd = SUMO_START_CMD
        self.load_cmd = SUMO_LOAD_CMD
        self.sim_step = 0
        self.tls_ids = None  # Will be populated when starting SUMO.
        self.monitored_edges = []  # Populate with your edge IDs or leave empty for now.
    
    def start(self):
        # Start the SUMO simulation.
        traci.start(self.start_cmd)
        self.tls_ids = traci.trafficlight.getIDList()
    
    def reset(self):
        # Reload the simulation using parameters allowed in traci.load()
        # Option 1: Use traci.load() if you want a "soft reset":
        traci.load(self.load_cmd)
        self.sim_step = 0
        return self.get_state()
        
        # Option 2 (Alternate): Restart the simulation completely:
        # traci.close()
        # traci.start(self.start_cmd)
        # self.sim_step = 0
        # return self.get_state()
    
    def step(self, joint_action):
        """
        Apply a joint action, which is a list of actions for each traffic light.
        The method simulates CONTROL_INTERVAL steps with that action.
        Returns: next_state, cumulative_reward, done flag.
        """
        cumulative_reward = 0.0
        
        # Execute the same action for CONTROL_INTERVAL simulation steps.
        for _ in range(CONTROL_INTERVAL):
            for idx, tls in enumerate(self.tls_ids):
                # Apply the action for each traffic light using the given phase index.
                # Adjust this call as needed (e.g., if you use setRedYellowGreenState instead)
                traci.trafficlight.setPhase(tls, int(joint_action[idx]))
            traci.simulationStep()
            self.sim_step += 1
            cumulative_reward += self.compute_reward()
            
            # Check termination condition (e.g., max simulation steps reached)
            if self.sim_step >= MAX_STEPS:
                next_state = self.get_state()
                return next_state, cumulative_reward, True
                
        next_state = self.get_state()
        done = False
        return next_state, cumulative_reward, done
    
    def get_state(self):
        state = []
        for tls in self.tls_ids:
            # 1) Current phase index
            phase = traci.trafficlight.getPhase(tls)

            # 2) Queue length on inbound edge
            inflows = traci.trafficlight.getControlledLinks(tls)
            # flatten link list to edge IDs
            edges = [link[0] for group in inflows for link in group]
            queue_lens = [traci.edge.getLastStepHaltingNumber(e) for e in edges]

            # 3) Total waiting time on those edges
            wait_times = []
            for e in edges:
                for vid in traci.edge.getLastStepVehicleIDs(e):
                    wait_times.append(traci.vehicle.getWaitingTime(vid))

            # 4) Instantaneous delay estimate (timeLoss is cumulative, so use speed)
            speeds = [traci.edge.getLastStepMeanSpeed(e) for e in edges]
            freespeed = [traci.edge.getMaxSpeed(e) for e in edges]
            # normalized delay per edge 1 - (speed/maxSpeed)
            delays = [1 - s/ms for s,ms in zip(speeds, freespeed)]

            features = [
                phase / len(self.tls_ids),  # normalized phase
                sum(queue_lens),            # total queue
                sum(wait_times),            # total wait
                np.mean(delays),            # average delay
                len(edges)                  # number of inbound links
            ]
            assert len(features) == FEATURES_PER_INT

            state.extend(features)
            
        return np.array(state, dtype=np.float32)
    
    def compute_reward(self):
        """
        Compute reward based on your simulation metrics.
        Here we use a dummy reward, but you can replace this with your own logic.
        """
        total_wait = 0.0
        max_q = 0
        
        # If you have defined monitored_edges, compute waiting times and queue lengths.
        for edge in self.monitored_edges:
            q = traci.edge.getLastStepHaltingNumber(edge)
            max_q = max(max_q, q)
            veh_ids = traci.edge.getLastStepVehicleIDs(edge)
            for veh in veh_ids:
                total_wait += traci.vehicle.getWaitingTime(veh)
        reward = - (total_wait + 5 * max_q)
        return reward
    
    def close(self):
        traci.close()
