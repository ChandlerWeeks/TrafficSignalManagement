import traci


class BaseEnv:
    """
    Thin SUMO ↔ Gym‑like wrapper (fixed routes; one step = 1 s).
    """

    def __init__(self, cfg: dict):
        self.sumo_bin = cfg.get('SUMO_BINARY', 'sumo')
        self.sumocfg  = cfg['SUMO_CFG']

        self._start_cmd = [
            self.sumo_bin, '-Q', '--no-step-log', '--no-warnings',
            '--step-length', '1',              # ← here
            '-c', self.sumocfg
        ]
        self._load_cmd  = ['--step-length', '1', '-c', self.sumocfg]

        traci.start(self._start_cmd)
        self.lanes = traci.lane.getIDList()

    # --------------- mandatory API ---------------
    def reset(self):
        traci.load(self._load_cmd)
        traci.simulationStep()
        return self.get_state()

    def step(self, action):
        self._apply_action(action)
        traci.simulationStep()
        ns   = self.get_state()
        rew  = self._compute_reward()
        done = traci.simulation.getMinExpectedNumber() == 0
        return ns, rew, done, {}

    # ---------- helpers overridable by subclass ----------
    def _apply_action(self, action):
        raise NotImplementedError

    def _compute_reward(self):
        # negative total queue
        return -sum(traci.lane.getLastStepHaltingNumber(l) for l in self.lanes)

    def get_state(self):
        return [traci.lane.getLastStepHaltingNumber(l) for l in self.lanes]

    def close(self):
        traci.close()
