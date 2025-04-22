def compute_total_queue(env):
    return sum(env.traci.lane.getLastStepHaltingNumber(l) for l in env.lanes)

def compute_average_wait(env):
    vids = env.traci.vehicle.getIDList()
    if not vids:
        return 0.0
    return sum(env.traci.vehicle.getWaitingTime(v) for v in vids) / len(vids)
