import numpy as np
from emdp.common import MDP

def build_two_circle_MDP(discount=0.6, good_reward=10., distractor_reward=5.):
    """MDP counter example given in Fig 1a of Zhang, et al.

    See "Generalized Off-Policy Actor-Critic" https://arxiv.org/pdf/1903.11329.pdf

    :param discount: The discount factor.
    :param good_reward: The good reward that the agent must find.
    :param distractor_reward: The disctraction reward.
    :returns: An emdp.common.MDP object.
    """
    ACTIONS = 2
    STATES = 11
    # Referrence of MDP states as in the paper.
    A = 0
    C = 1
    B = 5
    ACTUAL_REWARD_STATE = 3
    JOINER_STATE = 4

    # State 0 (A) is the starting state
    # States 1 - 3 are states in the first chain.
    FIRST_CHAIN = [C, 2, ACTUAL_REWARD_STATE, JOINER_STATE]
    # States 5 - 7 are states in the second chain.
    SECOND_CHAIN = [B, 6, 7, JOINER_STATE]
    # State 4 joins the two chains.
    # States 8 - 10 are states that lead back to A
    CONNECTION_CHAIN = [JOINER_STATE, 8, 9, 10, A]

    # DEFINING TRANSITION MATRIX.
    P = np.zeros((STATES, ACTIONS, STATES))

    # From the first state, the actions lead to different circumstances.
    P[A, 0, C] = 1.
    P[A, 1, B] = 1.

    # Within the chains, any action should lead to the next state in the chain.
    for chain in [FIRST_CHAIN, SECOND_CHAIN, CONNECTION_CHAIN]:
        for state_t, state_tp1 in zip(chain[:-1], chain[1:]):
            P[state_t, :, state_tp1] = 1.

    # DEFINING DISCOUNT FACTOR.
    gamma = discount

    # DEFINING REWARDS.
    # Both actions lead to the good reward.
    R = np.zeros((STATES, ACTIONS))
    R[ACTUAL_REWARD_STATE, :] = good_reward
    # Both actions lead to the distractor reward.
    R[B, :] = distractor_reward

    # DEFINING START STATES.
    p0 = np.zeros(STATES)
    p0[A] = 1.

    return MDP(P, R, gamma, p0=p0, terminal_states=[])
