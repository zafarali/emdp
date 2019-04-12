"""Allows using emdp as a gym environment."""
import numpy as np
import gym
from gym import spaces

import emdp.utils as utils

def gymify(mdp, **kwargs):
    return GymToMDP(mdp, **kwargs)

class GymToMDP(gym.Env):

    def __init__(self, mdp, observation_one_hot=True):
        """
        :param mdp: The emdp.MDP object to wrap.
        :param observation_one_hot: Boolean indicating if the observation space
            should be one hot or an integer.
        """
        self.mdp = mdp
        if observation_one_hot:
            self.observation_space = spaces.Box(
                low=0, high=1, shape=(self.mdp.state_space, ), dtype=np.int32)
        else:
            self.observation_space = spaces.Discrete(self.mdp.state_space)

        self.action_space = spaces.Discrete(self.mdp.action_space)

        self._obs_one_hot = observation_one_hot

    def reset(self):
        return self.maybe_convert_state(self.mdp.reset())

    def step(self, action):
        state, reward, done, info = self.mdp.step(action)
        
        return (self.maybe_convert_state(state),
                reward, done, info)

    def seed(self, seed):
        self.mdp.set_seed(seed)

    # TODO:
    def render(self):
        pass

    def maybe_convert_state(self, state):
        if self._obs_one_hot:
            return state
        else:
            return utils.convert_onehot_to_int(state)


