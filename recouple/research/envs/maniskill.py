import gym
import gymnasium
from mani_skill.utils.wrappers.gymnasium import CPUGymWrapper

import mani_skill.envs
import numpy as np

class ManiSkillRGBEnv(gym.Env):
    def __init__(
        self,
        env_name: str,
        obs_mode: str = "rgb",
        control_mode: str = "pd_ee_delta_pose",
    ):
        # Create the environment.
        self.env = gymnasium.make(
            env_name,
            num_envs=1,
            obs_mode=obs_mode, 
            control_mode=control_mode,
        )
        self.env = CPUGymWrapper(self.env)

        rgb_shape = self.env.observation_space["sensor_data"]["base_camera"]["rgb"].shape

        # Get the observation space
        self.observation_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=(rgb_shape[-1], rgb_shape[-3], rgb_shape[-2]),
            dtype=np.uint8,
        )

        # Get the action space
        self.action_space = gym.spaces.Box(
            self.env.action_space.low, 
            self.env.action_space.high, 
            self.env.action_space.shape, 
            dtype=self.env.action_space.dtype
        )

    def _format_obs(self, obs):
        _obs = obs["sensor_data"]["base_camera"]["rgb"]
        _obs = np.transpose(_obs , (2, 0, 1))
        return _obs
    
    def step(self, action: np.ndarray):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return self._format_obs(obs), reward, terminated or truncated, info

    def reset(self):
        return self._format_obs(self.env.reset()[0])


# class ManiSkillStateEnv(gym.Env):
#     def __init__(
#         self,
#         env_name: str,
#         obs_mode: str = "state",
#         control_mode: str = "pd_ee_delta_pos",
#     ):
#         # Create the environment.
#         self.env = gymnasium.make(
#             env_name,
#             num_envs=1,
#             obs_mode=obs_mode, 
#             control_mode=control_mode,
#         )

#         self.env = CPUGymWrapper(self.env)

#         # Get the observation space
#         self.observation_space = gym.spaces.Box(
#             self.env.observation_space.low, 
#             self.env.observation_space.high, 
#             self.env.observation_space.shape, 
#             dtype=self.env.observation_space.dtype
#         )

#         # Get the action space
#         self.action_space = gym.spaces.Box(
#             self.env.action_space.low, 
#             self.env.action_space.high, 
#             self.env.action_space.shape, 
#             dtype=self.env.action_space.dtype
#         )
    
#     def step(self, action: np.ndarray):
#         obs, reward, terminated, truncated, info = self.env.step(action)
#         return obs, reward, terminated or truncated, info

#     def reset(self):
#         return self.env.reset()[0]


class ManiSkillStateEnv(gym.Env):
    def __init__(
        self,
        env_name: str,
        obs_mode: str = "state",
        control_mode: str = "pd_ee_delta_pose",
    ):
        # Create the environment.
        self.env = gymnasium.make(
            env_name,
            num_envs=1,
            obs_mode=obs_mode, 
            control_mode=control_mode,
            sim_backend="gpu",
        )

        # Get the observation space
        self.observation_space = gym.spaces.Box(
            self.env.observation_space.low[0, ...], 
            self.env.observation_space.high[0, ...], 
            self.env.observation_space.shape[1:], 
            dtype=self.env.observation_space.dtype
        )

        # Get the action space
        self.action_space = gym.spaces.Box(
            self.env.action_space.low, 
            self.env.action_space.high, 
            self.env.action_space.shape, 
            dtype=self.env.action_space.dtype
        )
        
        # breakpoint()

    def _format_obs(self, obs):
        _obs = obs[0, ...]
        return _obs
    
    def step(self, action: np.ndarray):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return self._format_obs(obs), reward, terminated or truncated, info

    def reset(self):
        return self._format_obs(self.env.reset()[0])