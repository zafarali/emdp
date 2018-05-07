"""
Utilities to help build more complex grid worlds.
"""
import numpy as np
from .helper_utilities import (build_simple_grid,
                               get_possible_actions,
                               check_can_take_action,
                               get_state_after_executing_action,
                               flatten_state,
                               unflatten_state)
from . import GridWorldMDP

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

    def add_grid(self, terminal_states=[], p_success=1):
        """
        Adds a grid so that you cant walk off the edges of the grid
        :param terminal_states:
        :param p_success:
        :return:
        """
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
        for i, from_state in enumerate(from_states): # enumerate over states
            tmp = transition_probs_from[i, target_state] # get the prob of transitioning
            transition_probs_from[i, target_state] = 0 # set it to zero
            transition_probs_from[i, from_state] += tmp # add the transition prob to staying in the same place

        self._P[from_states, from_actions, :] = transition_probs_from
        # renormalize and update transition matrix.
        normalization = self._P.sum(2)
        # normalization[normalization == 0] = 1
        normalization = 1/normalization
        self._P = (self._P * np.repeat(normalization, self._P.shape[0]).reshape(*self._P.shape))

        assert np.all(np.equal(self._P.sum(2), 1)), 'Normalization did not occur correctly: {}'.format(self._P.sum(2))
        self._P_modified = True

    @property
    def P(self, nocopy=False):
        """
        Returns a new array with the transition matrix built so far.
        :param nocopy:
        :return:
        """
        if nocopy:
            return self._P
        else:
            return self._P.copy()


# class RewardMatrixBuilder()

def create_reward_matrix(state_space, size, reward_spec, action_space=4):
    """
    Abstraction to create reward matrices.
    :param state_space: Size of the state space
    :param size: Size of the gird world (width)
    :param reward_spec: The reward specification
    :param action_space: The size of the action space
    :return:
    """
    R = np.zeros((state_space, action_space))
    for (reward_location, reward_value) in reward_spec.items():
        reward_location = flatten_state(reward_location, size, state_space).argmax()
        R[reward_location, :] = reward_value

    return R

"""
Simple builders for gridworlds
"""
def build_simple_grid_world_with_terminal_states(reward_spec, size, p_success=1, gamma=0.99, seed=2017):
    """
    A simple size x size grid world where agents actions has a prob of p_success of executing correctly.
    rewards are given by a dict where the indices and the x,y positions and the value is the magnitude of the reward.
    Upon reaching a state with a reward, every action gives a reward. The episode then goes to an absorbing state and terminates.
    :param reward_spec: Reward specification
    :param size: Size of the gridworld (grid world will be size x size)
    :param p_success: The probability the action is successful.
    :param gamma: The discount factor.
    :param seed: Seed for the GridWorldMDP object.
    :return:
    """
    P = build_simple_grid(size=size, terminal_states=reward_spec.keys(), p_success=p_success)
    R = create_reward_matrix(P.shape[0], size, reward_spec, action_space=4)
    p0 = np.array([1] + [0]*(P.shape[0]-1)) #starting state distribution

    return GridWorldMDP(P, R, gamma, p0, terminal_states=reward_spec.keys(), size=size, seed=seed)


def build_simple_grid_world_without_terminal_states(reward_spec, size, p_success=1, gamma=0.99, seed=2017):
    """
    A simple size x size grid world where agents actions has a prob of p_success of executing correctly.
    rewards are given by a dict where the indices and the x,y positions and the value is the magnitude of the reward.
    Upon reaching a state with a reward, every action gives a reward. The episode does not terminate.
    :param reward_spec: Reward specification
    :param size: Size of the gridworld (grid world will be size x size)
    :param p_success: The probability the action is successful.
    :param gamma: The discount factor.
    :param seed: Seed for the GridWorldMDP object.
    :return:
    """
    P = build_simple_grid(size=size, terminal_states=[], p_success=p_success)
    R = create_reward_matrix(P.shape[0], size, reward_spec, action_space=4)
    p0 = np.array([1] + [0]*(P.shape[0]-1)) #starting state distribution

    return GridWorldMDP(P, R, gamma, p0, terminal_states=[], size=size, seed=seed)

