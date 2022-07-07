import numpy as np
import os
import re
import string
from getpass import getuser
from utils.plot import Plotter
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from callbacks import TensorboardCallback, CheckpointCallback, EvalCallback

# Specify path for tensorboard log files
log_dir = r"C:\Users\{}\Documents\Tensorboard".format(getuser())

class Agent:
    def __init__(self, algorithm, env, log_dir=log_dir, save_name=None, eval_freq=10000, policy="MlpPolicy", verbose=0):
        self.algorithm = algorithm
        self.env = DummyVecEnv([lambda: Monitor(env)])
        self.log_dir = log_dir
        self.save_name = save_name
        self.eval_freq = eval_freq

        self.episode_length = env.episode_length
        self.kwargs = {"policy": policy, "env": VecNormalize(venv=self.env, norm_obs=True, norm_reward=True)}
        self.verbose = verbose
        self.model = None

        if self.log_dir is not None:
            if self.save_name is not None:
                self.log_dir = os.path.join(self.log_dir, env.__class__.__name__, self.save_name)
            else:
                self.log_dir = os.path.join(self.log_dir, env.__class__.__name__)
            # Create log directory
            if not os.path.isdir(self.log_dir):
                os.makedirs(self.log_dir)

            # Define name of agent
            self.agent_name = self.algorithm.__name__

            # Get path for saving agent data
            self.save_name = self._get_save_name()

    def _get_save_name(self):
        num = [int(re.findall(r'\d+', folder)[-1]) for folder in os.listdir(self.log_dir)
               if folder.rstrip(string.digits) == self.agent_name + "_"]
        if len(num) > 0:
            n = np.max(num) + 1
        else:
            n = 1
        save_name = os.path.join(self.log_dir, self.agent_name + "_" + str(n))

        return save_name

    def _initiate_callbacks(self):
        # Initiate list of callbacks
        self.callbacks = [TensorboardCallback(),
                          CheckpointCallback(save_path=self.save_name),
                          EvalCallback(eval_env=VecNormalize(self.env),
                                       best_model_save_path=self.save_name,
                                       eval_freq=self.eval_freq,
                                       n_eval_episodes=10)]

    def set_hyperparams(self, hyperparams={}):

        self.agent_name = self.algorithm.__name__

        if "learning_rate" in hyperparams and self.log_dir is not None:
            self.agent_name = self.agent_name + "_lr{}".format(hyperparams["learning_rate"])

        if "gamma" in hyperparams and self.log_dir is not None:
            self.agent_name = self.agent_name + "_gamma{}".format(hyperparams["gamma"])

        if "gae_lambda" in hyperparams and self.log_dir is not None:
            self.agent_name = self.agent_name + "_gae_lambda{}".format(hyperparams["gae_lambda"])

        if "target_kl" in hyperparams and self.log_dir is not None:
            self.agent_name = self.agent_name + "_target_kl{}".format(hyperparams["target_kl"])

        if "clip_range" in hyperparams and self.log_dir is not None:
            self.agent_name = self.agent_name + "_clip_range{}".format(hyperparams["clip_range"])

        if "n_epochs" in hyperparams and self.log_dir is not None:
            hyperparams["n_epochs"] = int(hyperparams["n_epochs"])
            self.agent_name = self.agent_name + "_n_epochs{}".format(hyperparams["n_epochs"])

        if "vf_coef" in hyperparams and self.log_dir is not None:
            self.agent_name = self.agent_name + "_vf_coef{}".format(hyperparams["vf_coef"])

        if "ent_coef" in hyperparams and self.log_dir is not None:
            self.agent_name = self.agent_name + "_ent_coef{}".format(hyperparams["ent_coef"])

        if "max_grad_norm" in hyperparams and self.log_dir is not None:
            self.agent_name = self.agent_name + "_mgn{}".format(hyperparams["max_grad_norm"])

        if "batch_size" in hyperparams and self.log_dir is not None:
            hyperparams["batch_size"] = int(hyperparams["batch_size"])
            if hyperparams["batch_size"] > hyperparams["n_steps"]:
                hyperparams["n_steps"] = hyperparams["batch_size"]
            self.agent_name = self.agent_name + "_bs{}".format(hyperparams["batch_size"])

        if "n_steps" in hyperparams and self.log_dir is not None:
            hyperparams["n_steps"] = int(hyperparams["n_steps"])
            self.agent_name = self.agent_name + "_n_steps{}".format(hyperparams["n_steps"])

        if "net_arch" in hyperparams:
            hyperparams.update({"policy_kwargs": {"net_arch": hyperparams["net_arch"]}})
            hyperparams.pop("net_arch")
            if self.log_dir is not None:
                self.agent_name = self.agent_name + "_hl{}_neu{}".format(len(hyperparams["policy_kwargs"]["net_arch"]),
                                                                       hyperparams["policy_kwargs"]["net_arch"][0])

        if "activation_fn" in hyperparams:
            hyperparams.update({"policy_kwargs": {"activation_fn": hyperparams["activation_fn"]}})
            hyperparams.pop("activation_fn")
            if self.log_dir is not None:
                self.agent_name = self.agent_name + "_actfn{}".format(hyperparams["policy_kwargs"]["activation_fn"])

        if "hidden_layers" and "neurons" in hyperparams:
            hyperparams.update({"policy_kwargs":
                                    {"net_arch": [int(hyperparams["neurons"])]*int(hyperparams["hidden_layers"])}})
            hyperparams.pop("hidden_layers")
            hyperparams.pop("neurons")
            if self.log_dir is not None:
                self.agent_name = self.agent_name + "_hl{}_neu{}".format(len(hyperparams["policy_kwargs"]["net_arch"]),
                                                                       hyperparams["policy_kwargs"]["net_arch"][0])

        self.kwargs.update(hyperparams)

        if self.log_dir is not None:
            self.save_name = self._get_save_name()

        self.model = self.algorithm(**self.kwargs, tensorboard_log=self.log_dir, verbose=self.verbose)
        self._initiate_callbacks()

    def train(self, num_train_timesteps):
        if self.model is None:
            self.set_hyperparams({"net_arch": [64], "learning_rate": 0.001, "n_steps": 2048,
                                  "batch_size": 512, "n_epochs": 1, "gamma": 0.995, "gae_lambda": 0.92,
                                  "vf_coef": 0.5})
            self._initiate_callbacks()

        print("")
        print("Performing training for {} timesteps".format(num_train_timesteps))
        print("")
        if self.log_dir is not None:
            self.model.learn(total_timesteps=num_train_timesteps, callback=self.callbacks, tb_log_name=self.agent_name)
        else:
            self.model.learn(total_timesteps=num_train_timesteps, callback=self.callbacks)

        print("")
        print("¤¤¤¤¤¤¤ Training finished ¤¤¤¤¤¤¤")
        print("")

        if self.save_name is not None:
            path = os.path.join(self.save_name, "final_model")
            os.makedirs(path, exist_ok=True)
            self.model.save(os.path.join(path, "model"))
            self.model.get_vec_normalize_env().save(os.path.join(path, "vec_normalize.pkl"))
            if self.verbose > 1:
                print("Saving final model to {}".format(os.path.join(path, "model")))

    def continue_training(self, num_train_timesteps=1000, model_path=None):
        if self.model is None:
            env = VecNormalize.load(os.path.join(model_path, "final_model", "vec_normalize.pkl"), venv=self.env)
            env.reset()
            self.model = self.algorithm.load(path=os.path.join(model_path, "final_model", "model"),
                                             tensorboard_log=model_path, env=env)

            self.save_name = model_path
            self._initiate_callbacks()

        print("")
        print("Continuing training for {} timesteps".format(num_train_timesteps))
        print("")

        self.model.learn(total_timesteps=num_train_timesteps, tb_log_name="continued training",
                         callback=self.callbacks, reset_num_timesteps=False)

        print("")
        print("¤¤¤¤¤¤¤ Training finished ¤¤¤¤¤¤¤")
        print("")

        path = os.path.join(model_path, "final_model")
        self.model.save(os.path.join(path, "model"))
        self.model.get_vec_normalize_env().save(os.path.join(path, "vec_normalize.pkl"))
        if self.verbose > 1:
            print("Saving new model to {}".format(os.path.join(path, "model")))


    def test(self, model_path=None, save_dir=None, num_test_episodes=1, render_test=False, use_best_model=False,
             show_plot=False, show_vel_acc=False):

        if save_dir is not None:
            self.save_name = save_dir

        if num_test_episodes <= 0:
            print("Agent will not be tested - exiting")
            exit()

        if self.model is None:
            if use_best_model:
                self.env = VecNormalize.load(os.path.join(model_path, "best_model", "vec_normalize.pkl"), venv=self.env)
                self.env.training = False
                self.env.norm_reward = False
                self.model = self.algorithm.load(os.path.join(model_path, "best_model", "model"), env=self.env)
            else:
                self.env = VecNormalize.load(os.path.join(model_path, "final_model", "vec_normalize.pkl"), venv=self.env)
                self.env.training = False
                self.env.norm_reward = False
                self.model = self.algorithm.load(os.path.join(model_path, "final_model", "model"), env=self.env)
        else:
            if use_best_model:
                self.env = VecNormalize.load(os.path.join(model_path, "best_model", "vec_normalize.pkl"), venv=self.env)
                self.env.training = False
                self.env.norm_reward = False
                self.model = self.algorithm.load(os.path.join(self.save_name, "best_model", "model"))
            else:
                self.env.norm_reward = False
                self.model.env.norm_reward = False

        print("")
        print("Running test for {} episodes".format(num_test_episodes))
        print("")

        if self.model.env.venv.envs[0].env.__class__.__name__ == "ScrewEnv":

            torques = []
            rewards = []
            durations = []
            frictions = []
            depths = []

            for episode in range(num_test_episodes):

                episode_depths = []
                episode_velocities = []
                episode_accelerations = []
                episode_torques = []
                episode_frictions = []

                obs = self.model.env.reset()
                episode_reward = 0
                done = False

                while not done:
                    action = self.model.predict(obs, deterministic=True)[0]
                    episode_frictions.append(self.model.env.venv.envs[0].env._set_friction())
                    obs, reward, done, _ = self.model.env.step(action)
                    episode_reward += reward[0]
                    if render_test:
                        self.model.env.venv.envs[0].env.render()
                    if done:
                        episode_depths.append(self.model.env.venv.buf_infos[0]["terminal_observation"][0])
                        episode_velocities.append(-1*self.model.env.venv.buf_infos[0]["terminal_observation"][1])
                        episode_accelerations.append(-1*self.model.env.venv.buf_infos[0]["terminal_observation"][2])
                        episode_torques.append(self.model.env.venv.buf_infos[0]["terminal_observation"][3])
                    else:
                        episode_depths.append(self.model.env.get_original_obs()[0][0])
                        episode_velocities.append(-1*self.model.env.get_original_obs()[0][1])
                        episode_accelerations.append(-1*self.model.env.get_original_obs()[0][2])
                        episode_torques.append(self.model.env.get_original_obs()[0][3])

                episode_duration = len(episode_depths)*0.01

                torques.append(episode_torques)
                rewards.append(episode_reward)
                durations.append(episode_duration)
                frictions.append(episode_frictions)
                depths.append(episode_depths)

                print('Episode: {} - Total reward: {}'.format(episode+1, episode_reward))
                print("Process duration: {:.2f} seconds".format(episode_duration))
                print("")

                Plotter().agent_test(n_episode=episode+1,
                                     depths=episode_depths,
                                     torques=episode_torques,
                                     target=episode_frictions,
                                     velocities=episode_velocities,
                                     accelerations=episode_accelerations,
                                     xlabel="Insertion depth [m]",
                                     ylabel="Torque [Nm]",
                                     save_dir=self.save_name,
                                     show_plot=show_plot,
                                     show_vel_acc=show_vel_acc)

            if self.log_dir is not None:
                np.savez(file=os.path.join(self.save_name, "test"), depths=np.array(depths, dtype=object),
                         torques=np.array(torques, dtype=object), rewards=np.array(rewards, dtype=object),
                         frictions=np.array(frictions, dtype=object), durations=np.array(durations, dtype=object))
            elif model_path is not None:
                np.savez(file=os.path.join(model_path, "test"), depths=np.array(depths, dtype=object),
                         torques=np.array(torques, dtype=object), rewards=np.array(rewards, dtype=object),
                         frictions=np.array(frictions, dtype=object), durations=np.array(durations, dtype=object))

            print("Average reward after testing for {} episodes: {}".format(num_test_episodes, np.mean(rewards)))
            print("Average process duration: {:.2f} seconds".format(np.mean(durations)))
