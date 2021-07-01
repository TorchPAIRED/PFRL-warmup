import gym


class DiscreteToBoxActionSpace(gym.ActionWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.did_conversion = False
        if isinstance(env.action_space, gym.spaces.Discrete):
            size = env.action_space.n
            self.did_conversion = True
            self.action_space = gym.spaces.Box(0, size, shape=(1,))
        assert isinstance(self.action_space, gym.spaces.Box)

    def action(self, action):
        if not self.did_conversion:
            return action

        from math import floor
        action = floor(action)
        return action