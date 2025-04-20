# traffic_signal_nn/utils/metrics.py

def compute_total_queue(env):
    """
    Sum of halting vehicles across all lanes.
    """
    return sum(env.traci.lane.getLastStepHaltingNumber(lane)
               for lane in env.lanes)

def compute_average_wait(env):
    """
    Average waiting time per vehicle.
    """
    vids = env.traci.vehicle.getIDList()
    if not vids:
        return 0.0
    total = sum(env.traci.vehicle.getWaitingTime(v) for v in vids)
    return total / len(vids)
