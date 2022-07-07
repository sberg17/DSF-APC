# Data-driven Simulation Framework for Adaptive Process Control
This repository contains the Python scripts, which comprise The Data-driven Simulation Framework (DSF), which can be used for performing adaptive process control.

The framework is developed with a simulation of a screwing process, which takes real-world process data as input and uses it to emulate friction that occurs during the process. An RL agent can then be trained on the framework in order to learn adaptive process control.

The main.py script is used to start the training of the RL agent on the simulation framework. From the script, it is possible to define, which RL algorithm to use, number of training timesteps, and the number of episodes to evaluate the RL policy on.
