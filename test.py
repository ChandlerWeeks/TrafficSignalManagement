import traci
from config import SUMO_CMD

traci.start(SUMO_CMD)
print("Simulation started successfully.")
tls_ids = traci.trafficlight.getIDList()
print("Traffic lights:", tls_ids)
traci.close()
