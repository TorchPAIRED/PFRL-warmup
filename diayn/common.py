import gym
import numpy as np

def augment_obs(observation, z, oh_len):
    if oh_len == 1:
        oh = z
    else:
        oh = np.zeros((oh_len,), dtype=np.float32)
        oh[z] = 1

    obs = np.append(observation, oh)
    obs = obs.astype(np.float32)
    return obs

def resize_obs_space(original_obs_space, oh_len):
    obs_space_shape = np.array(original_obs_space.shape).copy()
    obs_space_shape[-1] = obs_space_shape[-1] + oh_len

    new_observation_space = gym.spaces.Box(
        original_obs_space.low[0],
        original_obs_space.high[0],
        tuple(list(obs_space_shape)),
        dtype=original_obs_space.dtype
    )

    return new_observation_space