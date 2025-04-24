# Traffic Signal Management

This Project uses SUMO and MARL Deep Q-learning to predict optimal timings for traffic signals. It's goal is the prove that the city of starkville could benefit from a smart traffic signal control system, compared to the threshold systems currently in place. This could be used to help manage traffic on busy days such as football game days, during busy hours where people are commuting to and from class, or even light hours such as summer when the student body is much smaller. 

## Run instructions

### Install sumo here: https://sumo.dlr.de/docs/Installing/index.html

If on mac XQuarts is required as a GUI system for SUMO: https://www.xquartz.org
Linux systems have not been tested for running SUMO

run SUMO through SUMO GUI application, and varry the traffic for various environments.

## Files

Benchmark.py - Benchmarks the implemented traffic signal controllers based on the behavior of the cars in the simulation. 
get_queue_length.py - Get the benchmark of average queue length of a light at any point in time. 

python -m traffic_signal_nn.eval.evaluator --config traffic_signal_nn/config/config_dqn_city.ini
python traffic_signal_nn/main.py --config traffic_signal_nn/config/config_dqn_city.ini train
