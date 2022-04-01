r"""
Tools to get analytic solutions from MDPs.

we can compute :math:`v_\pi(s)` recursively by solving the system of Bellman equations below [Bellman1957]_:

.. math::

    \begin{align}
        v_\pi(s) &= \sum_{a} \left[
            \pi(a|s) \left( r(s,a) + \gamma \sum_{s'} p(s'|s,a) v_\pi(s') \right)
        \right] \\
        &=\sum_a \pi(a|s)r(s,a) + \gamma \sum_{s'} \left[ \left(\sum_a \pi(a|s)p(s'|s,a)\right) v_\pi(s') \right] \\
        &=r_\pi(s) + \gamma \sum_{s'} p_\pi(s'|s) v_\pi(s')
    \end{align}

These equations can also be written in matrix form with
:math:`\mathbf{v}_\pi, \mathbf{r}_\pi \in \mathbb{R}^{|\mathcal{S}|}` and 
:math:`\mathbf{p}_\pi \in \mathbb{R}^{|S|\times|S|}`:

.. math::

    \begin{align}
        \mathbf{v}_\pi &= \mathbf{r}_\pi + \gamma \mathbf{p}_\pi \mathbf{v}_\pi \\
        &= (\mathbf{I} - \gamma \mathbf{p}_\pi)^{-1} \mathbf{r}_\pi \\
        &= \Phi \mathbf{r}_\pi    
    \end{align}
    
.. [Bellman1957] Bellman, Richard. 1957. “A Markovian Decision Process.” Journal of mathematics and mechanics: 679--684.
"""
import numpy as np


def calculate_P_pi(P, pi):
    """
    Calculates the transition matrix :math:`P` under policy :math:`pi`. 
    :math:`p_\pi:=Pr(s'|s,a\sim\pi))`, which is represented as a matrix of shape :math:`|\mathcal{S}|\\times|\mathcal{S}|`.

    .. math::

        p_\pi(s,s') = \sum_a \pi(a|s) p(s'|s, a)

    where :math:`s` and :math:`s'` are the states before and after taking action :math:`a`.

    Args:
        P(np.ndarray): transition matrix of size :math:`|\mathcal{S}|\\times|\mathcal{A}|\\times|\mathcal{S}|`
        pi(np.ndarray): matrix of size :math:`|\mathcal{S}|\\times|\mathcal{A}|` indicating the policy

    Returns:
        np.ndarray: a matrix of size :math:`|\mathcal{S}|\\times|\mathcal{S}|`
    """
    return np.einsum('sat,sa->st', P, pi)

def calculate_R_pi(R, pi):
    r"""
    Calculates the expected reward :math:`r_\pi` under policy :math:`\pi`, 
    which is represented as a matrix of shape :math:`|\mathcal{S}|`.
    
    .. math::

        r_\pi(s) = \sum_a \pi(a|s) r(s,a)

    Args:
        R(np.ndarray): reward matrix of size :math:`|\mathcal{S}|\times|\mathcal{A}|`
        pi(np.ndarray): matrix of size :math:`|\mathcal{S}|\times|\mathcal{A}|` indicating the policy
    
    Returns:
        np.ndarray: a matrix of size :math:`|\mathcal{S}|`
    """
    return np.einsum('sa,sa->s', R, pi)

def calculate_successor_representation(P_pi, gamma):
    r"""
    Calculates the successor representation :math:`\Phi`
    
    .. math::
    
        \Phi := (\mathbf{I} - \gamma \mathbf{p}_\pi)^{-1}

    see also: :func:`emdp.analytic.calculate_V_pi`

    :param P_pi:
    :param gamma:

    Returns:
        np.ndarray: successor representation
    """
    return np.linalg.inv(np.eye(P_pi.shape[0]) - gamma * P_pi)


def calculate_V_pi_from_successor_representation(Phi, R_pi):
    r"""
    Calculates the state-value vector :math:`\mathbf{v}_\pi` from the successor representation :math:`\Phi` and the expected reward :math:`\mathbf{r}_\pi`.

    see also: :func:`emdp.analytic.calculate_V_pi`

    Args:
        Phi(np.ndarray): successor representation of size :math:`|\mathcal{S}|\times|\mathcal{S}|`
        R_pi(np.ndarray): expected reward of size :math:`|\mathcal{S}|`
    
    Returns:
        np.ndarray: value function of size :math:`|\mathcal{S}|`
    """

    return np.einsum('st,t->s', Phi, R_pi)

def calculate_V_pi(P, R, pi, gamma):
    r"""
    Calculates the state-value :math:`v_\pi` from the successor representation using the analytic form:
    
    .. math::
        
        (\mathbf{I} - \gamma \mathbf{p}_\pi)^{-1} \mathbf{r}_\pi

    where :math:`p_\pi(s,t) = \sum_a \pi(a|s) p(t|s, a)`
    and :math:`r_\pi(s) = \sum_a \pi(a|s) r(s,a)`

    see also :func:`emdp.analytic.calculate_P_pi` and :func:`emdp.analytic.calculate_R_pi`.

    .. note::

        we can compute :math:`v_\pi(s)` recursively by solving the system of Bellman equations below [Bellman1957]_:

        .. math::

            \begin{align}
                v_\pi(s) &= \sum_{a} \left[
                    \pi(a|s) \left( r(s,a) + \gamma \sum_{s'} p(s'|s,a) v_\pi(s') \right)
                \right] \\
                &=\sum_a \pi(a|s)r(s,a) + \gamma \sum_{s'} \left[ \left(\sum_a \pi(a|s)p(s'|s,a)\right) v_\pi(s') \right] \\
                &=r_\pi(s) + \gamma \sum_{s'} p_\pi(s'|s) v_\pi(s')
            \end{align}

        These equations can also be written in matrix form with
        :math:`\mathbf{v}_\pi, \mathbf{r}_\pi \in \mathbb{R}^{|\mathcal{S}|}` and 
        :math:`\mathbf{p}_\pi \in \mathbb{R}^{|S|\times|S|}`:

        .. math::

            \begin{align}
                \mathbf{v}_\pi &= \mathbf{r}_\pi + \gamma \mathbf{p}_\pi \mathbf{v}_\pi \\
                &= (\mathbf{I} - \gamma \mathbf{p}_\pi)^{-1} \mathbf{r}_\pi \\
                &= \Phi \mathbf{r}_\pi    
            \end{align}

    Args:
        P(np.ndarray): Transition matrix
        R(np.ndarray): Reward matrix
        pi(np.ndarray): policy matrix
        gamma(float): discount factor

    Returns:
        np.ndarray: state-value vector under policy :math:`\pi`.
    """
    P_pi = calculate_P_pi(P, pi)
    R_pi = calculate_R_pi(R, pi)
    Phi = calculate_successor_representation(P_pi, gamma)
    return calculate_V_pi_from_successor_representation(Phi, R_pi)
