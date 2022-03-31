"""
A simple grid world environment
"""
import numpy as np
import random
from ..common import MDP
from ..exceptions import EpisodeDoneError, InvalidActionError
from ..actions import LEFT, RIGHT, UP, DOWN
from .helper_utilities import flatten_state, unflatten_state
from typing import List, Tuple


class GridWorldMDP(MDP):
    """
    .. note:: 
        if ``terminal_states`` is not empty then there will be an absorbing state. So
        the actual number of states will be :math:`size^2 + 1`
        if there is a terminal state, it should be the last one.

    Args:
        P (np.ndarray): state transition matrix :math:`P: \mathcal{S}\\times\mathcal{A}\\times\mathcal{S}\mapsto\mathbb{R}`, 
            the shape is :math:`|S| \\times |A| \\times |S|`.
        R (np.ndarray): reward matrix :math:`r: \mathcal{S}\\times \mathcal{A}\mapsto \mathbb{R}`, 
            the shape is:math:`|S| \\times |A|`.
        gamma (float): discount factor :math:`\gamma`
        p0 (np.ndarray): initial starting distribution :math:`p_0`. The array shape is :math:`|\mathcal{S}|=size\\times size`.
        terminal_states (List[Tuple[int,int]]): Must be a list of (x,y) tuples.  
            use skip_terminal_state_conversion if giving ints
        size (int): the size of the grid world (i.e there are :math:`size \\times size + 1 = |\mathcal{S}|` states in total).
        seed (int, optional): the random seed for simulations. Defaults to 1337.
        skip_check (bool, optional): _description_. Defaults to False.
        convert_terminal_states_to_ints (bool, optional): _description_. Defaults to False.
    """

    def __init__(self, P, R, gamma, p0, terminal_states: List[Tuple[int, int]], size:int, 
                seed=1337, skip_check=False,
                convert_terminal_states_to_ints=False):
        if not convert_terminal_states_to_ints:
            terminal_states = list(map(lambda tupl: int(size * tupl[0] + tupl[1]), terminal_states))
        self.size = size
        self.human_state = (None, None)
        self.has_absorbing_state = len(terminal_states) > 0
        super().__init__(P, R, gamma, p0, terminal_states, seed=seed, skip_check=skip_check)

    def reset(self):
        super().reset()
        self.human_state = self.unflatten_state(self.current_state)
        return self.current_state

    def flatten_state(self, state):
        """Flatten state (x,y) into a one hot vector"""
        return flatten_state(state, self.size, self.state_space)

    def unflatten_state(self, onehot):
        """Unflatten a one hot vector into a (x,y) pair"""
        return unflatten_state(onehot, self.size, self.has_absorbing_state)

    def step(self, action):
        state, reward, done, info = super().step(action)
        self.human_state = self.unflatten_state(self.current_state)
        return state, reward, done, info

    def set_current_state_to(self, tuple_state):
        return super().set_current_state_to(self.flatten_state(tuple_state).argmax())
