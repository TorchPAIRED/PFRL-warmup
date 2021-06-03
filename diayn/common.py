import gym
import numpy as np

def augment_obs(observation, z):
    obs = np.append(observation, z)
    obs = obs.astype(np.float32)
    return obs

def resize_obs_space(original_obs_space):
    obs_space_shape = np.array(original_obs_space.shape).copy()
    obs_space_shape[-1] = obs_space_shape[-1] + 1

    new_observation_space = gym.spaces.Box(
        original_obs_space.low[0],
        original_obs_space.high[0],
        tuple(list(obs_space_shape)),
        dtype=original_obs_space.dtype
    )

    return new_observation_space