import numpy as np
from emdp import actions
from emdp.gridworld.helper_utilities import build_simple_grid, check_can_take_action
from emdp.gridworld.env import GridWorldMDP
from emdp.common import MDP

def build_SB_example35():
    """
    Example 3.5 from (Sutton and Barto, 2018) pg 60 (March 2018 version).
    A rectangular Gridworld representation of size 5 x 5.

    Quotation from book:
    At each state, four actions are possible: north, south, east, and west, which deterministically
    cause the agent to move one cell in the respective direction on the grid. Actions that
    would take the agent off the grid leave its location unchanged, but also result in a reward
    of −1. Other actions result in a reward of 0, except those that move the agent out of the
    special states A and B. From state A, all four actions yield a reward of +10 and take the
    agent to A'. From state B, all actions yield a reward of +5 and take the agent to B'
    """
    size = 5
    P = build_simple_grid(size=size, p_success=1)
    # modify P to match dynamics from book.

    P[1, :, :] = 0 # first set the probability of all actions from state 1 to zero
    P[1, :, 21] = 1 # now set the probability of going from 1 to 21 with prob 1 for all actions

    P[3, :, :] = 0  # first set the probability of all actions from state 3 to zero
    P[3, :, 13] = 1  # now set the probability of going from 3 to 13 with prob 1 for all actions

    # TODO: add rewards for walking off the grid


    R = np.zeros((P.shape[0], P.shape[1])) # initialize a matrix of size |S|x|A|

    for state in range(P.shape[0]):
        for action in [actions.UP, actions.LEFT, actions.RIGHT, actions.DOWN]:
            if not check_can_take_action(action, state, size):
                R[state, action] = -1

    R[1, :] = +10
    R[3, :] = +5

    p0 = np.ones(P.shape[0])/P.shape[0] # uniform starting probability (assumed)
    gamma = 0.9

    terminal_states = []
    return GridWorldMDP(P, R, gamma, p0, terminal_states, size)


def build_SB_example41():
    """
    Example 4.1 from (Sutton and Barto, 2018)  pg (Jan 2018 version).
    """
    pass


def build_twostate_MDP():
    """
    MDP with transition probabilities
    P(s_0 | s_0, a_0) = 0.5
    P(s_1 | s_0, a_0) = 0.5
    P(s_0 | s_0, a_1) = 0
    P(s_1 | s_0, a_1) = 1
    P(s_1 | s_0, a_2) = 0
    P(s_1 | s_1, a_2) = 1
    Rewards: r(s_0, a_0) = 5, r(s_0, a_1) = 10, r(s_1, a_2) = -1
    Discount factor : 0.95
    :return:
    """
    P = np.zeros((2, 3, 2))
    P[0, 0] = [0.5, 0.5]
    P[0, 1] = [0, 1]
    P[0, 2] = [1, 0]  # no op
    P[1, 2] = [0, 1]
    P[1, 1] = [0, 1]
    P[1, 0] = [0, 1]
    T = {0: {0: [0.5, 0.5], 1: [0, 1]}, 1: {2: [0, 1]}}
    gamma = 0.9
    R = np.zeros((2, 3))
    R[0, 0] = 5
    R[0, 1] = 10
    R[1, 2] = -1

    return MDP(P, R, gamma, p0=np.array([0.5, 0.5]), terminal_states=[])


