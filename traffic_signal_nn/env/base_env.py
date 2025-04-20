# traffic_signal_nn/env/base_env.py

import traci

class BaseEnv:
    """
    Wrap a single‑junction SUMO sim as an OpenAI‑style env.
    """
    def __init__(self, cfg):
        # cfg should contain SUMO_BINARY, SUMO_CFG, STEP_LEN
        self.sumo_binary = cfg.get("SUMO_BINARY", "sumo")
        self.sumocfg     = cfg["SUMO_CFG"]
        self.step_len    = cfg.get("STEP_LEN", 1)
        self.sumo_cmd    = [self.sumo_binary, "-c", self.sumocfg]
        self._start_sumo()
        self.lanes       = traci.lane.getIDList()

    def _start_sumo(self):
        traci.start(self.sumo_cmd)

    def reset(self):
        traci.load(self.sumo_cmd)
        self._start_sumo()
        return self.get_state()

    def step(self, action):
        self._apply_action(action)
        traci.simulationStep()
        next_state = self.get_state()
        reward     = self._compute_reward()
        done       = traci.simulation.getMinExpectedNumber() == 0
        return next_state, reward, done, {}

    def _apply_action(self, action):
        """
        Must be overridden: map `action` to traffic‑light phase changes.
        """
        raise NotImplementedError

    def _compute_reward(self):
        """
        Default reward = –total queue.
        """
        return -sum(traci.lane.getLastStepHaltingNumber(l)
                    for l in self.lanes)

    def get_state(self):
        """
        Default state = vector of per‑lane queue lengths.
        """
        return [traci.lane.getLastStepHaltingNumber(lane)
                for lane in self.lanes]

    def close(self):
        traci.close()
