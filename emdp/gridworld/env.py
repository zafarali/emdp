"""
A simple grid world environment
"""
import numpy as np
import random
from ..common import MDP
from ..exceptions import EpisodeDoneError, InvalidActionError
from ..actions import LEFT, RIGHT, UP, DOWN

class GridWorldMDP(MDP):
    def __init__(self, P, R, gamma, p0, terminal_states, size, seed=1337, skip_check=False):
        """
        (!) if terminal_states is not empty then there will be an absorbing state. So
            the actual number of states will be size x size + 1
            if there is a terminal state, it should be the last one.
        :param P: Transition matrix |S| x |A| x |S|
        :param R: Transition matrix |S| x |A|
        :param gamma: discount factor
        :param p0: initial starting distribution
        :param terminal_states: the terminal states
        :param size: the size of the grid world (i.e there are size x size (+ 1)= |S| states)
        :param seed:
        :param skip_check:
        """
        super().__init__(P, R, gamma, p0, terminal_states, seed=seed, skip_check=skip_check)
        self.size =  size
        self.human_state = (None, None)
        self.reset()
        self.has_absorbing_state = len(terminal_states) > 0
        self.human_state = self.unflatten_state(self.current_state)

    def flatten_state(self, state):
        """Flatten state (x,y) into a one hot vector"""
        idx = self.size * state[0] + state[1]
        one_hot = np.zeros(self.state_space)
        one_hot[idx] = 1
        return one_hot

    def unflatten_state(self, onehot):
        """Unflatten a one hot vector into a (x,y) pair"""
        if self.has_absorbing_state:
            onehot = onehot[:-1]
        onehot = onehot.reshape(self.size, self.size)
        x = onehot.argmax(0).max()
        y = onehot.argmax(1).max()
        return (x, y)

    def step(self, action):
        state, reward, done, info = super().step(action)
        self.human_state = self.unflatten_state(self.current_state)
        return state, reward, done, info

    def set_current_state_to(self, tuple_state):
        return super().set_current_state_to(self.flatten_state(tuple_state).argmax())