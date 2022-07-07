from stable_baselines3 import PPO
from datetime import datetime
from modAL.models import BayesianOptimizer
from sklearn.base import BaseEstimator
import os
from environment import *
from agent import Agent
import numpy as np

# Specify path for tensorboard log files
log_dir = r"C:\Users\{}\Documents\Tensorboard".format(getuser())

# Name of folder used for saving files
save_folder = os.path.join("Bayesian_tuning", "tuning_BO_{}".format(datetime.now().strftime("%d-%m-%y_%H-%M")))

class BayesHpOptimizer:
    def __init__(self, estimator: BaseEstimator, query_strategy=None, storage_dir=None, num_train_timesteps=1000000):
        storage_dir = storage_dir or os.getcwd()
        self._results_folder = os.path.join(storage_dir, "results_BO")

        self._estimator = estimator
        self._query_strategy = query_strategy
        self._num_train_timesteps = num_train_timesteps

        self._sampled_points = []
        self._performances = []

    def _evaluate(self, parameters, sample, step, n_steps, filename):
        print("Performing step {:d} out of {:d}".format(step, n_steps))
        print("Evaluating with parameters:")
        for key, val in zip(parameters, sample):
            print("\t{}:\t{}".format(key, val))

        # Define agent
        agent = Agent(algorithm=PPO, env=ScrewEnv(), log_dir=log_dir, save_name=save_folder, eval_freq=50000)

        # Add sample to list of sampled points
        self._sampled_points.append(sample)

        # Set sample as model HPs
        params = dict(zip(parameters, sample))
        agent.set_hyperparams(params)

        # Begin agent training
        agent.train(num_train_timesteps=self._num_train_timesteps)

        # Add last final mean reward to list of performances
        self._performances.append(agent.callbacks[2].last_mean_reward)

        np.savez(filename, parameter_names=parameters, performance=np.array(self._performances),
                 sampled_points=np.array(self._sampled_points))

        print("Performed step {:d} out of {:d}:".format(step, n_steps))
        print("Mean reward:\t {:.4f}".format(self._performances[-1]))
        print("")

        return self._performances[-1]

    def optimize(self, parameters, parameter_grid, steps, initial_training_samples=5):

        # Create storage directory
        os.makedirs(self._results_folder, exist_ok=True)

        filename = os.path.join(self._results_folder, "BayesianOptimization_{}.npz"
                                .format(datetime.now().strftime("%d-%m_%H-%M")))

        # Train on initial training samples
        if len(self._sampled_points) < initial_training_samples:
            print("Evaluating initial parameters")
            print("")
            start_indexes = np.random.choice(len(parameter_grid), initial_training_samples - len(self._sampled_points))

            for i, index in enumerate(start_indexes):
                self._evaluate(parameters, parameter_grid[index], len(self._sampled_points)+1,
                               initial_training_samples, filename)

        print("Done with initial parameter evaluation!")
        print()

        # Perform Bayesian optimization
        x_train, y_train = np.array(self._sampled_points), np.array(self._performances).reshape(-1, 1)

        optimizer = BayesianOptimizer(estimator=self._estimator, query_strategy=self._query_strategy,
                                      X_training=x_train, y_training=y_train)

        optimum_points = [optimizer.get_max()[0]]

        predictions, std_devs = optimizer.predict(parameter_grid, return_std=True)
        predictions, std_devs = [predictions], [std_devs]

        best_performance = -np.inf

        while len(predictions) <= steps:
            np.savez(filename, parameter_names=parameters, parameter_grid=parameter_grid,
                     predictions=np.array(predictions), optimum_points=np.array(optimum_points),
                     std_devs=np.array(std_devs), sampled_points=np.array(self._sampled_points),
                     performance=np.array(self._performances))


            # For every 5th query use a random sample (exploration)
            if len(predictions) % 10 == 0:
                print("Using random sample for exploration!")
                query_idx = [np.random.choice(len(parameter_grid))]
            else:
                query_idx, query_inst = optimizer.query(parameter_grid)


            performance = self._evaluate(parameters, parameter_grid[query_idx][0],
                                         len(predictions), steps, filename)

            if performance > best_performance:
                best_performance = performance
                best_step = len(predictions)
                print("New best performance found!")
                print("")

            optimizer.teach(parameter_grid[query_idx].reshape(1, -1), np.array([performance]).reshape(1, -1))
            predictions.append(optimizer.predict(parameter_grid))
            std_devs.append(optimizer.predict(parameter_grid, return_std=True)[1])
            optimum_points.append(optimizer.get_max()[0])

        np.savez(filename, parameter_names=parameters, parameter_grid=parameter_grid,
                 predictions=np.array(predictions), optimum_points=np.array(optimum_points),
                 std_devs=np.array(std_devs), sampled_points=np.array(self._sampled_points),
                 performance=np.array(self._performances))

        print("Hyperparameter tuning done!")
        print("Best mean reward found to {} after {} queries".format(best_performance, best_step))
        print("With best parameters:".format(optimum_points[-1]))
        for key, val in zip(parameters, optimum_points[-1]):
            print("\t{}:\t{}".format(key, val))
