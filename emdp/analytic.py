"""
Tools to get analytic solutions from MDPs
"""
import numpy as np


def calculate_P_pi(P, pi):
    r"""
    calculates P_pi
    P_pi(s,t) = \sum_a pi(s,a) p(s, a, t)
    :param P: transition matrix of size |S|x|A|x|S|
    :param pi: matrix of size |S| x |A| indicating the policy
    :return: a matrix of size |S| x |S|
    """
    return np.einsum('sat,sa->st', P, pi)

def calculate_R_pi(R, pi):
    r"""
    calculates R_pi
    R_pi(s) = \sum_a pi(s,a) r(s,a)
    :param R: reward matrix of size |S| x |A|
    :param pi: matrix of size |S| x |A| indicating the policy
    :return:
    """
    return np.einsum('sa,sa->s', R, pi)

def calculate_successor_representation(P_pi, gamma):
    """
    Calculates the successor representation
    (I- gamma*P_pi)^{-1}
    :param P_pi:
    :param gamma:
    :return:
    """
    return np.linalg.inv(np.eye(P_pi.shape[0]) - gamma * P_pi)


def calculate_V_pi_from_successor_representation(Phi, R_pi):
    return np.einsum('st,t->s', Phi, R_pi)

def calculate_V_pi(P, R, pi, gamma):
    r"""
    Calculates V_pi from the successor representation using the analytic form:
    (I- gamma*P_pi)^{-1} * R_pi
    where P_pi(s,t) = \sum_a pi(s,a) p(s, a, t)
    and R_pi(s) = \sum_a pi(s,a) r(s,a)
    :param P: Transition matrix
    :param R: Reward matrix
    :param pi: policy matrix
    :param gamma: discount factor
    :return:
    """
    P_pi = calculate_P_pi(P, pi)
    R_pi = calculate_R_pi(R, pi)
    Phi = calculate_successor_representation(P_pi, gamma)
    return calculate_V_pi_from_successor_representation(Phi, R_pi)
