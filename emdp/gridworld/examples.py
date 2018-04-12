import numpy as np
from ..actions import LEFT, RIGHT, UP, DOWN

def build_SB_example35():
    """
    Exmaple 3.5 from (Sutton and Barto, 2018) pg 60 (March 2018 version).
    A rectangular Gridworld representation of size 5 x 5.

    Quotation from book:
    At each state, four actions are possible: north, south, east, and west, which deterministically
    cause the agent to move one cell in the respective direction on the grid. Actions that
    would take the agent off the grid leave its location unchanged, but also result in a reward
    of −1. Other actions result in a reward of 0, except those that move the agent out of the
    special states A and B. From state A, all four actions yield a reward of +10 and take the
    agent to A'. From state B, all actions yield a reward of +5 and take the agent to B'
    """
    pass

def build_SB_example41():
    """
    Exmaple 4.1 from (Sutton and Barto, 2018) (March 2018 version).
    """
    pass

