"""
Utilities to help build more complex grid worlds.
"""
import numpy as np
from typing import List, Tuple, Dict
from . import GridWorldMDP
from .helper_utilities import (build_simple_grid,
                               flatten_state)


class TransitionMatrixBuilder(object):
    """
    Builder object to build a transition matrix for a grid world
    """

    def __init__(self, grid_size, action_space=4, has_terminal_state=True):
        self.has_terminal_state = has_terminal_state
        self.grid_size = grid_size
        self.action_space = action_space
        self.state_space = grid_size * grid_size + int(has_terminal_state)
        self._P = np.zeros((self.state_space, self.action_space, self.state_space))
        self.grid_added = False
        self.P_modified = False

    def add_grid(self, terminal_states: List[int] = None, p_success: float = 1):
        """Adds a grid so that you cant walk off the edges of the grid


        Args:
            terminal_states (List[int], optional): Terminal states. 
                Defaults to ``[]``.
            p_success (float, optional): Defaults to 1.

        Raises:
            ValueError
        """
        if terminal_states is None:
            terminal_states = []

        if self.has_terminal_state and len(terminal_states) == 0:
            raise ValueError('has_terminal_states is true, but no terminal states supplied.')

        if self.grid_added:
            raise ValueError('Grid has already been added')

        if self.P_modified:
            raise ValueError('transition matrix has already been modified. '
                             'Adding a grid now can lead to weird behaviour')

        self._P = build_simple_grid(size=self.grid_size, p_success=p_success, terminal_states=terminal_states)
        self.grid_added = True
        self.P_modified = True

    def add_wall_at(self, tuple_location):
        """
        Add a blockade at this position
        :param tuple_location: (x,y) location of the wall
        :return:
        """
        target_state = flatten_state(tuple_location, self.grid_size, self.state_space)
        target_state = target_state.argmax()
        # find all the ways to go to "target_state"
        # from_states contains states that can lead you to target_state by executing from_action
        from_states, from_actions = np.where(self._P[:, :, target_state] != 0)

        # get the transition probability distributions that go from s--> t via some action
        transition_probs_from = self._P[from_states, from_actions, :]
        # TODO: optimize this loop
        for i, from_state in enumerate(from_states):  # enumerate over states
            tmp = transition_probs_from[i, target_state]  # get the prob of transitioning
            transition_probs_from[i, target_state] = 0  # set it to zero
            transition_probs_from[i, from_state] += tmp  # add the transition prob to staying in the same place

        self._P[from_states, from_actions, :] = transition_probs_from

        # Get the probability of going to any state for all actions from target_state.
        transition_probs_from_wall = self._P[target_state, :, :]
        for i, probs_from_action in enumerate(transition_probs_from_wall):
            # Reset the probabilities.
            transition_probs_from_wall[i, :] = 0.0
            # Set the probability of going to the target state to be 1.0
            transition_probs_from_wall[i, target_state] = 1.0
        # Now set the probs of going to any state from target state as above (i.e only targets).
        self._P[target_state, :, :] = transition_probs_from_wall

        # renormalize and update transition matrix.
        normalization = self._P.sum(2)
        # normalization[normalization == 0] = 1
        normalization = 1 / normalization
        self._P = (self._P * np.repeat(normalization, self._P.shape[0]).reshape(*self._P.shape))

        assert np.allclose(self._P.sum(2), 1), 'Normalization did not occur correctly: {}'.format(self._P.sum(2))
        assert np.allclose(self._P[target_state, :, target_state], 1.0), 'All actions from wall should lead to wall!'
        self._P_modified = True

    @property
    def P(self, nocopy=False):
        """Returns a new array with the transition matrix built so far.

        Args:
            nocopy (bool, optional): Defaults to False.

        Returns:
            np.array: the transition model matrix
        """
        if nocopy:
            return self._P
        else:
            return self._P.copy()

    def add_wall_between(self, start: Tuple[int, int], end: Tuple[int, int]):
        """
        Adds a wall between the starting and ending location

        Args:
            start (Tuple[int,int]): tuple (x,y) representing the starting position of the wall
            end (Tuple[int,int]): tuple (x,y) representing the ending position of the wall

        Raises:
            ValueError
        """

        if not (start[0] == end[0] or start[1] == end[1]):
            raise ValueError('Walls can only be drawn in straight lines. '
                             'Therefore, at least one of the x or y between '
                             'the states should match.')

        if start[0] == end[0]:
            direction = 1
        else:
            direction = 0

        constant_idx = start[int(not direction)]
        start_idx = start[direction]
        end_idx = end[direction]

        if end_idx < start_idx:
            # flip start and end directions
            # to ensure we can still draw walls
            start_idx, end_idx = end_idx, start_idx

        for i in range(start_idx, end_idx + 1):
            my_location = [None, None]
            my_location[direction] = i
            my_location[int(not direction)] = constant_idx
            print(my_location)
            self.add_wall_at(tuple(my_location))


def create_reward_matrix(state_space, size, reward_spec: Dict[Tuple[int, int], float], action_space=4):
    """
    Abstraction to create reward matrices.

    Args:
        state_space(int): Size of the state space, :math:`|\mathcal{S}|`.
        size(int): size of the gird world (width or height).
        reward_spec(Dict[Tuple[int,int], float]): the reward specification.
        action_space(int): the size of the action space

    Returns:
        np.ndarray: the reward matrix.
    """

    R = np.zeros((state_space, action_space))
    for (reward_location, reward_value) in reward_spec.items():
        reward_location = flatten_state(reward_location, size, state_space).argmax()
        R[reward_location, :] = reward_value

    return R


"""
Simple builders for gridworlds
"""


def build_simple_grid_world_with_terminal_states(reward_spec,
                                                 size,
                                                 p_success=1,
                                                 gamma=0.99,
                                                 seed=2017,
                                                 start_state=0):
    """
    A simple size x size grid world where agents actions has a prob of p_success of executing correctly.
    rewards are given by a dict where the indices and the x,y positions and the value is the magnitude of the reward.
    Upon reaching a state with a reward, every action gives a reward. The episode then goes to an absorbing state and terminates.

    :param reward_spec: Reward specification
    :param size: Size of the gridworld (grid world will be size x size)
    :param p_success: The probability the action is successful.
    :param gamma: The discount factor.
    :param seed: Seed for the GridWorldMDP object.
    :param start_state: The index of the starding state.
    :return:
    """
    P = build_simple_grid(size=size, terminal_states=reward_spec.keys(), p_success=p_success)
    R = create_reward_matrix(P.shape[0], size, reward_spec, action_space=4)
    p0 = np.zeros(P.shape[0])
    p0[start_state] = 1

    return GridWorldMDP(P, R, gamma, p0, terminal_states=reward_spec.keys(), size=size, seed=seed)


def build_simple_grid_world_without_terminal_states(reward_spec,
                                                    size,
                                                    p_success=1,
                                                    gamma=0.99,
                                                    seed=2017,
                                                    start_state=0):
    """
    A simple size x size grid world where agents actions has a prob of p_success of executing correctly.
    rewards are given by a dict where the indices and the x,y positions and the value is the magnitude of the reward.
    Upon reaching a state with a reward, every action gives a reward. The episode does not terminate.
    
    :param reward_spec: Reward specification
    :param size: Size of the gridworld (grid world will be size x size)
    :param p_success: The probability the action is successful.
    :param gamma: The discount factor.
    :param seed: Seed for the GridWorldMDP object.
    :param start_state: The index of the starting state.
    :return:
    """
    P = build_simple_grid(size=size, terminal_states=[], p_success=p_success)
    R = create_reward_matrix(P.shape[0], size, reward_spec, action_space=4)
    p0 = np.zeros(P.shape[0])
    p0[start_state] = 1

    return GridWorldMDP(P, R, gamma, p0, terminal_states=[], size=size, seed=seed)
