import traci
from .base_env import BaseEnv


class CityEnv(BaseEnv):
    """
    Control *all* traffic lights (“fixed‑routes” scenario).
    """

    def __init__(self, cfg):
        super().__init__(cfg)

        raw = cfg.get('TLS_IDS', 'all').strip().lower()
        self.tls_ids = (traci.trafficlight.getIDList()
                        if raw == 'all'
                        else [t.strip() for t in raw.split(',')])

        self.n_phases = [len(traci.trafficlight
                             .getAllProgramLogics(t)[0].phases)
                         for t in self.tls_ids]

    # ---- overrides ----
    def _apply_action(self, actions):
        if not isinstance(actions, (list, tuple)):
            actions = [actions] * len(self.tls_ids)
        for tid, act in zip(self.tls_ids, actions):
            traci.trafficlight.setPhase(tid, int(act))

    def get_state(self):
        out = []
        for tid in self.tls_ids:
            lanes = traci.trafficlight.getControlledLanes(tid)
            out.append([traci.lane.getLastStepHaltingNumber(l) for l in lanes])
        return out

    # ---- Gym spaces (simple) ----
    @property
    def observation_space(self):
        return [len(x) for x in self.get_state()]

    @property
    def action_space(self):
        return self.n_phases
