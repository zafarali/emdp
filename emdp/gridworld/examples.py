import numpy as np
from ..actions import LEFT, RIGHT, UP, DOWN
from .helper_utilities import build_simple_grid
from .env import GridWorldMDP

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

    R = np.zeros((P.shape[0], P.shape[1])) # initialize a matrix of size |S|x|A|
    R[1, :] = +10
    R[3, :] = +1

    p0 = np.ones(P.shape[0])/P.shape[0] # uniform starting probability (assumed)
    gamma = 0.9

    terminal_states = []
    return GridWorldMDP(P, R, gamma, p0, terminal_states, size)


def build_SB_example41():
    """
    Example 4.1 from (Sutton and Barto, 2018)  pg (Jan 2018 version).
    """
    pass

