"""
Tools to get analytic solutions from MDPs.

These functions are differentiable as they are written in torch.
"""
import numpy as np
try:
    import torch

    def _silent_convert(np_or_tensor):
        """Silently convert a numpy array into a tensor."""
        if isinstance(np_or_tensor, np.ndarray):
            return torch.from_numpy(np_or_tensor).float()
        return np_or_tensor

    def convert_arguments_to_torch(function):
        """A simple decorator to prevent type checking everywhere."""
        def wrapped_function(*args):
            converted_args = [_silent_convert(arg) for arg in args]
            return function(*converted_args)
        return wrapped_function

    @convert_arguments_to_torch
    def calculate_P_pi(P, pi):
        """
        calculates P_pi
        P_pi(s,t) = \sum_a pi(s,a) p(s, a, t)
        :param P: transition matrix of size |S|x|A|x|S|
        :param pi: matrix of size |S| x |A| indicating the policy
        :return: a matrix of size |S| x |S|
        """
        return torch.einsum('sat,sa->st', P, pi)

    @convert_arguments_to_torch
    def calculate_R_pi(R, pi):
        """
        calculates R_pi
        R_pi(s) = \sum_a pi(s,a) r(s,a)
        :param R: reward matrix of size |S| x |A|
        :param pi: matrix of size |S| x |A| indicating the policy
        :return:
        """
        return torch.einsum('sa,sa->s', R, pi)

    @convert_arguments_to_torch
    def calculate_successor_representation(P_pi, gamma):
        """
        Calculates the successor representation
        (I- gamma*P_pi)^{-1}
        :param P_pi:
        :param gamma:
        :return:
        """
        return torch.inverse(torch.eye(P_pi.shape[0]) - gamma * P_pi)

    @convert_arguments_to_torch
    def calculate_V_pi_from_successor_representation(Phi, R_pi):
        return torch.einsum('st,t->s', Phi, R_pi)

    @convert_arguments_to_torch
    def calculate_V_pi(P, R, pi, gamma):
        """
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

except ImportError:
    pass
