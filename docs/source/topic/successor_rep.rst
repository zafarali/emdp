Successor Representation
============================

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

see also: :mod:`emdp.analytic`