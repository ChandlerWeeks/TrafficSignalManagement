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
import argparse

# -- figure out where SUMO is installed --
sumo_home = os.environ.get("SUMO_HOME")
if sumo_home:
    tools_path = os.path.join(sumo_home, "tools")
    if os.path.isdir(tools_path):
        sys.path.insert(0, tools_path)
    else:
        print(f"⚠️  SUMO_HOME is set to '{sumo_home}', but no 'tools' dir found there.")
        print("   Continuing anyway—assuming 'sumo' and 'sumo-gui' are on your PATH.")
else:
    print("⚠️  SUMO_HOME not set—assuming 'sumo' and 'sumo-gui' are on your PATH.")

def check_sumo():
    try:
        which = subprocess.run(["where" if os.name == "nt" else "which", "sumo"],
                               capture_output=True, text=True).stdout.strip()
        print("Using SUMO at:", which or "<not found>")
        version = subprocess.run(["sumo", "-v"], capture_output=True, text=True).stdout.strip()
        print("SUMO version:", version or "<unknown>")
    except Exception as e:
        print("Error running SUMO:", e)

check_sumo()

def get_queue_length(sim_cfg, output_csv="./data/output/queue_length_data.csv", max_steps=None):
    print(f"\n▶ Starting SUMO simulation with config: {sim_cfg}")
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

    print(f"\n✅ Queue length data written to {output_csv}")
    print("✅ Simulation complete.")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument('--config', '-c', default="./data/simulation.sumocfg",
                   help="path to your SUMO .sumocfg file")
    p.add_argument('--output', '-o', default="./data/output/queue_length_data.csv",
                   help="where to write the CSV")
    p.add_argument('--steps', '-s', type=int, default=None,
                   help="max number of simulation steps (default=run until done)")
    args = p.parse_args()

    get_queue_length(sim_cfg=args.config,
                     output_csv=args.output,
                     max_steps=args.steps)
