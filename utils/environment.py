from gym import Env
from gym.spaces import Box
import numpy as np
from dynamics_estimation import DynamicsEstimator
from dm_control import mujoco
import cv2
from getpass import getuser

class ScrewEnv(Env):
    def __init__(self,
                 xml_file=r"C:\Users\{}\Documents\VT4_Velux\utils\resources\mujoco\screw.xml".format(getuser()),
                 process_data=None,
                 goal_in_obs=True,
                 trq_cost_weight=0.00002,
                 acc_cost_weight=75,
                 frame_skip=5,
                 random_init_state=False):

        self.action_space = Box(low=np.array([-1], dtype=np.float32),
                                high=np.array([1], dtype=np.float32))

        self._trq_cost_weight = trq_cost_weight
        self._acc_cost_weight = acc_cost_weight
        self.frame_skip = frame_skip
        self.random_init_state = random_init_state
        self.physics = mujoco.Physics.from_xml_path(xml_file)
        self.model = self.physics.model
        self.data = self.physics.data

        if process_data is not None:
            self.dynamics = DynamicsEstimator(data_path=process_data)
        else:
            self.dynamics = DynamicsEstimator()

        self.friction_data = self.dynamics.get_random_sample()
        self.rand_var = np.clip(np.random.normal(1, 1), a_min=0.5, a_max=None)

        self.pitch = 0.001
        self.goal = self.friction_data.screw_length[0]
        self.goal_in_obs = goal_in_obs
        self.episode_duration = 20
        self.episode_length = self.episode_duration/(self.physics.timestep()*self.frame_skip)

        self.torque = 0

        action = self.action_space.sample()
        observation, _reward, done, _info = self.step(action)
        assert not done

        self._set_observation_space(observation)

    def _set_observation_space(self, observation):
        if isinstance(observation, np.ndarray):
            low = np.full(observation.shape, -float("inf"), dtype=np.float32)
            high = np.full(observation.shape, float("inf"), dtype=np.float32)
            space = Box(low, high, dtype=observation.dtype)
        else:
            raise NotImplementedError(type(observation), observation)

        self.observation_space = space

        return self.observation_space

    def step(self, action):
        xpos_before = self._get_obs()[0]
        self._set_friction()
        self.torque += action[0]/10
        self.do_simulation(np.array(self.torque), self.frame_skip)
        self.physics.data.qpos[0] = -self.pitch * self.physics.data.qpos[1]
        xpos_after = self._get_obs()[0]

        reward = (xpos_after - xpos_before) * 10

        #trq_cost = np.square(self.torque)*self._trq_cost_weight
        trq_cost = 0
        acc_cost = np.square(self._get_obs()[2])*self._acc_cost_weight
        ctrl_cost = trq_cost + acc_cost

        reward -= ctrl_cost

        if xpos_after >= self.goal:
            done = True
            reward = 1
        elif self.physics.data.time >= self.episode_duration:
            done = True
            reward = -1
        else:
            done = False

        obs = self._get_obs()

        return obs, reward, done, dict()

    def _get_obs(self):
        position = np.array([self.physics.data.qpos[0]])
        velocity = np.array([self.physics.data.qvel[1]])*self.pitch
        acceleration = np.array([self.physics.data.qacc[1]])*self.pitch
        torque = np.array([self.torque])
        goal = np.array([self.goal])

        if self.goal_in_obs:
            observations = np.concatenate((position, velocity, acceleration, torque, goal))
        else:
            observations = np.concatenate((position, velocity, acceleration, torque))

        return observations

    def _set_friction(self):
        position = self._get_obs()[0]
        self.model.dof_frictionloss[1] = self.dynamics.determine_friction(position, self.friction_data)*self.rand_var
        return self.model.dof_frictionloss[1]

    def reset(self):
        self.physics.reset()
        self.torque = 0
        self.friction_data = self.dynamics.get_random_sample()
        self.rand_var = np.clip(np.random.normal(1, 1), a_min=0.5, a_max=None)
        self.goal = self.friction_data.screw_length[0]
        return self._get_obs()

    @property
    def dt(self):
        return self.model.opt.timestep * self.frame_skip

    def do_simulation(self, ctrl, n_frames):
        self.physics.data.ctrl[:] = -ctrl
        for _ in range(n_frames):
            self.physics.step()

    def render(self, show_render=True):
        frame = self.physics.render(720, 1280, camera_id=0)
        if show_render:
            cv2.imshow("ScrewEnv", frame)
            cv2.waitKey(1)
        return frame
