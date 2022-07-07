from environment import *
from agent import Agent
from stable_baselines3 import PPO
from getpass import getuser

# Path to csv-file containing process data
process_data_path = r"C:\Users\{}\Documents\VT4_Velux\data\process_data.csv".format(getuser())

# Number of training timesteps
num_train_timesteps = 1000000

# Number of testing episodes
num_test_eps = 3

# Number of timesteps between every model evaluation
eval_freq = 10000

# Name of folder used for saving files
save_name = "results_for_paper_new_time_limit"

for _ in range(3):

    # Specify environment (screw length is added to observation space)
    env = ScrewEnv(process_data=process_data_path, goal_in_obs=True, acc_cost_weight=75)

    # Define agent object
    agent = Agent(algorithm=PPO, env=env, save_name=save_name, eval_freq=eval_freq)
    agent.set_hyperparams()

    # Begin training of agent
    agent.train(num_train_timesteps=num_train_timesteps)

    # Begin testing of agent
    agent.test(num_test_episodes=num_test_eps)
