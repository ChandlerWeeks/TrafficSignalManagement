import xml.etree.ElementTree as ET
import pandas as pd

# ----------------------------
# Parsing trip information data
# ----------------------------
# Replace 'tripinfo.xml' with your actual SUMO trip output file name
tripinfo_file = './data/output/tripinfo.xml'
try:
    tree = ET.parse(tripinfo_file)
    root = tree.getroot()
except Exception as e:
    raise Exception(f"Error parsing {tripinfo_file}: {e}")

# List to accumulate trip records
trip_records = []
# Loop through each tripinfo element in the XML file
for trip in root.findall('tripinfo'):
    # Extract trip attributes; if an attribute is missing, a default value of 0 is used
    trip_id = trip.attrib.get('id')
    duration = float(trip.attrib.get('duration', 0))
    waiting_time = float(trip.attrib.get('waitingTime', 0))
    delay = float(trip.attrib.get('timeLoss', 0))
    
    trip_records.append({
        'id': trip_id,
        'duration': duration,
        'waiting_time': waiting_time,
        'delay': delay
    })

# Create a DataFrame from the collected trip information
df_trips = pd.DataFrame(trip_records)

# Calculate average values from the trip data
# Use a conditional check to avoid errors if the DataFrame is empty
avg_trip_time = df_trips['duration'].mean() if not df_trips.empty else 0
avg_wait_time = df_trips['waiting_time'].mean() if not df_trips.empty else 0
avg_delay = df_trips['delay'].mean() if not df_trips.empty else 0

print()
print("=== Trip Information Averages ===")
print("Average Trip Time:", avg_trip_time)
print("Average Waiting Time:", avg_wait_time)
print("Average Delay:", avg_delay)

# ----------------------------
# Parsing queue information data
# ----------------------------
# Replace 'queueinfo.xml' with your actual queue output file name
queueinfo_file = './data/output/queue_length_data.csv'
df_queue = pd.read_csv(queueinfo_file)
# drop 0 values at the end of the simulation
for i in range(len(df_queue)-1, 0, -1):
    if df_queue['queue_length'][i] == 0:
        df_queue = df_queue.drop(i)
    else:
        break

max_queue_length = df_queue['queue_length'].max()
avg_queue_length = df_queue['queue_length'].mean()

print()
print("=== Queue Information Averages ===")
print("Average Queue Length:", avg_queue_length.round(4))
print("Maximum Queue Length:", max_queue_length)
print()