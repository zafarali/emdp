import numpy as np
from emdp.common import MDP


def build_cake_world_mdp(epsilon, discount, cake_reward=1.0):
    r"""Cake world MDP from Action Gap Paper (Fig 1 of Bellemare et al. 2016).

    Increasing the Action Gap: New Operators for Reinforcement Learning.
    https://arxiv.org/pdf/1512.04860.pdf

    The action gap is modulated by epsilon since the difference between Q values
    for each action is given by `Q(x1, a2) - Q(x1, a2) = epsilon`.

    Args:
    :param epsilon: Float epsilon for the action gap.
    :param discount: Float discount factor.
    :param cake_reward: Float reward for eating cake.
    :returns: An emdp.common.MDP object.
    """
    STATES = 2
    ACTIONS = 2

    # Short hand to make following paper easy.
    x1, x2 = 0, 1
    a1, a2 = 0, 1

    P = np.zeros((STATES, ACTIONS, STATES))

    # Taking action a1 in state x1 takes you to x1 or x2 with equal likelihood.
    P[x1, a1, :] = .5

    # Taking the abstain action leads you back to x1.
    P[x1, a2, x1] = 1.

    # All actions from x2 should lead to x2 (Terminal state).
    P[x2, :, x2] = 1.

    # Found by solving for `r` in `V(x2) = r + discount * V(x2)`.
    # -2(1+e)/gamma = r + gamma * -2(1+e)/gamma.
    # Let r = rhat * 1/ gamma.
    # => -2 (1+e) = rhat + -2 * gamma * (1+e).
    # => -2 [ (1+e) - gamma * (1+e)] = rhat
    # => -2 [ (1+e)(1-gamma)] = rhat.
    # ==> r = -2 (1+e)(1-gamma)/gamma.
    forever_reward = -2.0 * (1 + epsilon) * (1 - discount) / discount

    R = np.zeros((STATES, ACTIONS))
    R[x2, :] = forever_reward  # Small negative forever reward.
    R[x1, a1] = cake_reward  # Cake!
    R[x1, a2] = 0.  # Abstain cake!

    p0 = np.array([1.0, 0.0])

    return MDP(P, R, discount, p0=p0, terminal_states=[x2])


