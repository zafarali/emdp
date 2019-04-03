import numpy as np
from emdp.common import MDP

def build_imani_counterexample():
    """
    MDP counter example given in Fig 1a of Imani, et al.
    "An Off-policy Policy Gradient Theorem Using Emphatic Weightings."
    Neurips 2018
    :return:
    """
    # |S| = 4, |A| = 2
    STATES = 4
    ACTIONS = 2
    P = np.zeros((STATES, ACTIONS, STATES))
    P[0, 0] = [0, 1, 0, 0]
    P[0, 1] = [0, 0, 1, 0]
    P[1, 0] = [0, 0, 0, 1]
    P[1, 1] = [0, 0, 0, 1]
    P[2, 0] = [0, 0, 0, 1]
    P[2, 1] = [0, 0, 0, 1]
    P[3, 0] = [0, 0, 0, 1]
    P[3, 1] = [0, 0, 0, 1]
    gamma = 0.99999
    R = np.zeros((STATES, ACTIONS))
    R[0, 0] = 0
    R[0, 1] = 1
    R[1, 0] = 2
    R[1, 1] = 0
    R[2, 0] = 0
    R[2, 1] = 1

    return MDP(P, R, gamma, p0=np.array([1, 0, 0, 0]), terminal_states=[3])
    pass

