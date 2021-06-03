import gym
import numpy as np

class DIAYNAugmentationWrapper(gym.ObservationWrapper):
    def __init__(self, env, augment_with_z):
        super().__init__(env)
        obs_space = env.observation_space
        obs_space_shape = np.array(obs_space.shape).copy()
        obs_space_shape[-1] = obs_space_shape[-1] + 1

        self.observation_space = gym.spaces.Box(obs_space.low[0], obs_space.high[0], tuple(list(obs_space_shape)),
                                                dtype=obs_space.dtype)

        self.action_space = env.action_space
        self.augment_with_z = augment_with_z

    def observation(self, observation):
        obs = np.append(observation, self.augment_with_z)
        obs = obs.astype(np.float32)
        return obs