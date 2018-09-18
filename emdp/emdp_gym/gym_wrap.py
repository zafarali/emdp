"""Allows using emdp as a gym environment."""
import gym
from gym import spaces

def gymify(mdp):
    return GymToMDP(mdp)

class GymToMDP(gym.Env):

    def __init__(self, mdp):
        self.mdp = mdp
        self.observation_space = spaces.Discrete(self.mdp.state_space)
        self.action_space = spaces.Discrete(self.mdp.action_space)

    def reset(self):
        return self.mdp.reset()

    def step(self, action):
        return self.mdp.step(action)

    def seed(self, seed):
        self.mdp.set_seed(seed)

    # TODO:
    def render(self):
        pass

