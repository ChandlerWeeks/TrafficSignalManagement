#!/usr/bin/env python3
"""
get_queue_length.py

Collects the queue length (number of stopped vehicles) at traffic lights for each simulation step
and writes the results to a CSV file.
"""

import os
import sys
import csv
import traci
import subprocess

# Set SUMO_HOME and append tools path
SUMO_HOME = "/Library/Frameworks/EclipseSUMO.framework/Versions/1.22.0/EclipseSUMO"
if not os.path.exists(SUMO_HOME):
    raise EnvironmentError("SUMO_HOME path does not exist. Please check the installation path.")

os.environ["SUMO_HOME"] = SUMO_HOME
tools_path = os.path.join(SUMO_HOME, "tools")
if tools_path not in sys.path:
    sys.path.append(tools_path)

# Optional: log where we're calling sumo from
def check_sumo():
    try:
        result = subprocess.run(["which", "sumo"], capture_output=True, text=True)
        print("Using SUMO at:", result.stdout.strip())
        result = subprocess.run(["sumo", "-v"], capture_output=True, text=True)
        print("SUMO version:", result.stdout.strip())
    except Exception as e:
        print("Error running SUMO:", e)

check_sumo()

def get_queue_length(sim_cfg, output_csv="./data/output/queue_length_data.csv", max_steps=None):
    print(f"Starting SUMO simulation with config: {sim_cfg}")
    traci.start(["sumo", "-c", sim_cfg])

    step = 0
    queue_data = []

    while True:
        if max_steps is not None and step > max_steps:
            break
        if traci.simulation.getMinExpectedNumber() == 0:
            break

        traci.simulationStep()

        total_queue_length = 0
        for tl_id in traci.trafficlight.getIDList():
            for lane_id in traci.trafficlight.getControlledLanes(tl_id):
                total_queue_length += traci.lane.getLastStepHaltingNumber(lane_id)

        queue_data.append((step, total_queue_length))
        step += 1

    traci.close()

    # Write to CSV
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    with open(output_csv, mode="w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["step", "queue_length"])
        writer.writerows(queue_data)

    print(f"✅ Queue length data written to {output_csv}")
    print("✅ Simulation complete.")

if __name__ == "__main__":
    SUMO_CONFIG = "./data/simulation.sumocfg"
    MAX_STEPS = 7200  # 2 hours
    get_queue_length(sim_cfg=SUMO_CONFIG, max_steps=MAX_STEPS)
