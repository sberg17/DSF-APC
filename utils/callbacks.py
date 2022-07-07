from stable_baselines3.common.callbacks import BaseCallback, EventCallback
from stable_baselines3.common.vec_env import VecEnv, DummyVecEnv, sync_envs_normalization
from stable_baselines3.common.evaluation import evaluate_policy
from typing import Any, Dict, Union, Optional
import warnings
import gym
import os
import numpy as np

class TensorboardCallback(BaseCallback):
    def __init__(self, verbose=1):
        super(TensorboardCallback, self).__init__(verbose)

    def _on_step(self) -> bool:
        observations = self.model.env.get_original_obs()[0]

        self.logger.record("observations/position", observations[0])
        self.logger.record("observations/velocity", observations[1])
        self.logger.record("observations/acceleration", observations[2])
        self.logger.record("observations/applied torque", observations[3])

        return True

class CheckpointCallback(BaseCallback):
    def __init__(self, save_path, save_freq: int = 100000, verbose: int = 0):
        super(CheckpointCallback, self).__init__(verbose)
        self.save_freq = save_freq
        self.save_path = os.path.join(save_path, "checkpoints")

    def _init_callback(self) -> None:
        # Create folder if needed
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.num_timesteps % self.save_freq == 0:
            path = os.path.join(self.save_path, "{}_steps".format(self.num_timesteps))
            os.makedirs(path, exist_ok=True)

            self.model.save(os.path.join(path, "model"))
            self.model.get_vec_normalize_env().save(os.path.join(path, "vec_normalize.pkl"))

            if self.verbose > 1:
                print("Saving model checkpoint to {}".format(os.path.join(path, "model")))

            return True

class EvalCallback(EventCallback):

    def __init__(self,
                 eval_env: Union[gym.Env, VecEnv],
                 callback_on_new_best: Optional[BaseCallback] = None,
                 callback_after_eval: Optional[BaseCallback] = None,
                 n_eval_episodes: int = 5,
                 eval_freq: int = 10000,
                 best_model_save_path: Optional[str] = None,
                 deterministic: bool = True,
                 render: bool = False,
                 verbose: int = 1,
                 warn: bool = True):

        super(EvalCallback, self).__init__(callback_after_eval, verbose=verbose)

        self.callback_on_new_best = callback_on_new_best
        if self.callback_on_new_best is not None:
            # Give access to the parent
            self.callback_on_new_best.parent = self

        self.n_eval_episodes = n_eval_episodes
        self.eval_freq = eval_freq
        self.best_mean_reward = -np.inf
        self.last_mean_reward = -np.inf
        self.deterministic = deterministic
        self.render = render
        self.warn = warn

        # Convert to VecEnv for consistency
        if not isinstance(eval_env, VecEnv):
            eval_env = DummyVecEnv([lambda: eval_env])

        self.eval_env = eval_env
        self.best_model_save_path = os.path.join(best_model_save_path, 'best_model')
        # For computing success rate
        self._is_success_buffer = []
        self.evaluations_successes = []

    def _init_callback(self) -> None:
        # Does not work in some corner cases, where the wrapper is not the same
        if not isinstance(self.training_env, type(self.eval_env)):
            warnings.warn("Training and eval env are not of the same type" f"{self.training_env} != {self.eval_env}")

        # Create folder if needed
        if self.best_model_save_path is not None:
            os.makedirs(self.best_model_save_path, exist_ok=True)

        # Init callback called on new best model
        if self.callback_on_new_best is not None:
            self.callback_on_new_best.init_callback(self.model)

    def _on_step(self) -> bool:

        continue_training = True

        if self.eval_freq > 0 and self.num_timesteps % self.eval_freq == 0:

            # Sync training and eval env if there is VecNormalize
            if self.model.get_vec_normalize_env() is not None:
                try:
                    sync_envs_normalization(self.training_env, self.eval_env)
                except AttributeError:
                    raise AssertionError(
                        "Training and eval env are not wrapped the same way, "
                        "see https://stable-baselines3.readthedocs.io/en/master/guide/callbacks.html#evalcallback "
                        "and warning above."
                    )

            episode_rewards, episode_lengths = evaluate_policy(model=self.model,
                                                               env=self.eval_env,
                                                               n_eval_episodes=self.n_eval_episodes,
                                                               render=self.render,
                                                               return_episode_rewards=True,
                                                               deterministic=self.deterministic,
                                                               warn=self.warn)

            mean_reward, std_reward = np.mean(episode_rewards), np.std(episode_rewards)
            mean_ep_length, std_ep_length = np.mean(episode_lengths), np.std(episode_lengths)
            self.last_mean_reward = mean_reward

            if self.verbose > 0:
                print(f"Eval num_timesteps = {self.num_timesteps}, " f"episode_reward = {mean_reward:.2f} +/- {std_reward:.2f}")
                print(f"Episode length: {mean_ep_length:.2f} +/- {std_ep_length:.2f}")
            # Add to current Logger
            self.logger.record("eval/mean_reward", float(mean_reward))
            self.logger.record("eval/mean_ep_length", mean_ep_length)

            # Dump log so the evaluation results are printed with the correct timestep
            self.logger.record("time/total_timesteps", self.num_timesteps, exclude="tensorboard")
            self.logger.dump(self.num_timesteps)

            if mean_reward > self.best_mean_reward:
                if self.verbose > 0:
                    print("New best mean reward!")
                if self.best_model_save_path is not None:
                    self.model.save(os.path.join(self.best_model_save_path, "model"))
                    self.model.get_vec_normalize_env().save(os.path.join(self.best_model_save_path, "vec_normalize.pkl"))
                self.best_mean_reward = mean_reward

            # Trigger callback after every evaluation, if needed
            if self.callback is not None:
                continue_training = continue_training and self._on_event()

        return continue_training

    def update_child_locals(self, locals_: Dict[str, Any]) -> None:
        """
        Update the references to the local variables.

        :param locals_: the local variables during rollout collection
        """
        if self.callback:
            self.callback.update_locals(locals_)
