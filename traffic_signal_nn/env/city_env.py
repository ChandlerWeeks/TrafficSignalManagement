# traffic_signal_nn/env/city_env.py

from .base_env import BaseEnv
import traci

class CityEnv(BaseEnv):
    """
    A single‑junction env for your target city.
    """
    def __init__(self, cfg):
        """
        cfg must include:
          - SUMO_CFG: path to simulation.sumocfg
          - TLS_ID:  traffic‑light system ID in the network
          - N_PHASES: number of signal phases
        """
        super().__init__(cfg)
        self.tls_id   = cfg["TLS_ID"]
        self.n_phases = cfg["N_PHASES"]

    def _apply_action(self, action):
        # assume `action` is int phase index
        traci.trafficlight.setPhase(self.tls_id, int(action))

    @property
    def observation_space(self):
        # simple vector length = #lanes
        return len(self.lanes)

    @property
    def action_space(self):
        return self.n_phases
