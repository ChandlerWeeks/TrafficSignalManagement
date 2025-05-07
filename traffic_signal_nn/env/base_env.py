import traci

class BaseEnv:
    def __init__(self, cfg: dict, sumo_dt: float = 0.1, ctrl_dt: float = 5.0):
        self.sumo_bin = cfg.get("SUMO_BINARY", "sumo")
        self.sumocfg  = cfg["SUMO_CFG"]
        self.sumo_dt  = float(sumo_dt)
        self.ctrl_dt  = float(ctrl_dt)

        COMMON = [
            "--no-step-log",
            "--no-warnings",
            "--emergencydecel.warning-threshold", "9.99",
            "--step-length",  str(self.sumo_dt),
            "-c",             self.sumocfg,
        ]
        traci.start([self.sumo_bin, "-Q", *COMMON])
        self._load_cmd = COMMON[:]                         # used by traci.load()

        # -------------- lane list PER traffic light -----------------
        tls_ids = cfg.get("TLS_IDS", None)
        if tls_ids and isinstance(tls_ids, str):
            tls_ids = [tid.strip() for tid in tls_ids.split(",") if tid.strip()]
        self.tls_ids  = tls_ids or [ ]                     # []  ⇒ single global agent
        self.tls_lanes = (self._collect_tls_lanes()
                          if self.tls_ids else [traci.lane.getIDList()])

    # ----------------------------------------------------------------
    def _collect_tls_lanes(self):
        """Return a list[ list[str] ]: lanes controlled by each TLS."""
        all_lanes = traci.lane.getIDList()
        res = []
        for tls in self.tls_ids:
            lanes = set()
            for group in traci.trafficlight.getControlledLinks(tls):
                for edge_in, _, _ in group:
                    prefix = edge_in + "_"
                    lanes.update(l for l in all_lanes if l.startswith(prefix))
            res.append(sorted(lanes))
        return res

    # =============  Gym‑style API  ==================================
    def reset(self):
        traci.load(self._load_cmd)
        traci.simulationStep()                 # 1 internal step
        return self.get_state()

    def step(self, actions):
        self._apply_action(actions)

        now = traci.simulation.getTime()
        traci.simulationStep(now + self.ctrl_dt)

        obs  = self.get_state()                # list[list[int]]
        rews = [-sum(q) for q in obs]          # list[float]
        done = traci.simulation.getMinExpectedNumber() == 0
        return obs, rews, done, {}

    # ----------------------------------------------------------------
    def get_state(self):
        """Return queue‑length vector per TLS (list of lists)."""
        return [
            [traci.lane.getLastStepHaltingNumber(l) for l in lanes]
            for lanes in self.tls_lanes
        ]

    # --------- hooks for subclass (traffic‑light actions) -----------
    def _apply_action(self, actions):
        raise NotImplementedError

    def close(self):
        if traci.isLoaded():
            traci.close(False)
