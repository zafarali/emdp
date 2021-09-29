import numpy as np
from emdp import actions
from emdp.gridworld.helper_utilities import build_simple_grid, check_can_take_action
from emdp.gridworld.builder_tools import create_reward_matrix
from emdp.gridworld.txt_utilities import get_char_matrix, build_gridworld_from_char_matrix
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

    # add rewards for walking off the grid
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


def build_SB_example41(size=4):
    """
    There are four actions possible in each state, A = {up, down, right, left}, which deterministically cause the
    corresponding state transitions, except that actions that would take the agent off the grid in fact leave the
    state unchanged.
    This is an undiscounted, episodic task.
    The reward is -1 on all transitions until the terminal state is reached.
    The terminal state is shaded in the figure (although it is shown in two places, it is formally one state).

    note on reward_spec/terminal_states:
    ------------------------------------
    On entering any of these states, the mdp inadvertently makes a last transition to the absorbing state and stays
    there forever (Ref: SB Section 3.4).
    In this example, all transitions get -1 as reward, except those last ones from terminal states, which have
    0 reward. Also in absorbing state, all actions receive 0 reward by definition of an episodic task.
    """
    size = size
    gamma = 1  # undiscounted episodic task
    p_success = 1  # actions always successful

    reward_spec = {(0, 0): 1, (size - 1, size - 1): 1}

    P = build_simple_grid(size=size, terminal_states=reward_spec.keys(), p_success=p_success)
    R = create_reward_matrix(P.shape[0], size, reward_spec, action_space=4)
    # print(f"R.shape={R.shape}")
    R += -1  # makes rewards for all transitions -1, except from terminal states
    R[P.shape[0] - 1, :] = 0  # also set the constructed dummy absorbing state's reward to 0
    # print(R)

    num_terminal_states = len(reward_spec.keys()) + 1

    p0 = np.ones(P.shape[0])  # assumption: terminal/absorbing states can't be the starting state!
    for t in reward_spec:
        # print(f"t={t}")
        p0[size * t[0] + t[1]] = 0

    p0[size * size] = 0  # start prob. of dummy absorbing state is also zero
    p0 = p0 / (P.shape[0] - num_terminal_states)

    return GridWorldMDP(P, R, gamma, p0, terminal_states=reward_spec.keys(), size=size)


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


_EXAMPLE_FOUR_ROOMS_TXT = """#############
#s    #     #
#     #     #
#           #
#     #     #
#     #     #
### ##### ###
#     #     #
#     #     #
#           #
#     #    g#
#     #     #
#############""".split('\n')

def build_four_rooms_example(gamma=0.99, seed=2017):
    char_matrix = get_char_matrix(_EXAMPLE_FOUR_ROOMS_TXT)
    return build_gridworld_from_char_matrix(char_matrix, seed=seed, gamma=gamma)


