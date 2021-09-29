from emdp.analytic import calculate_V_pi
from emdp.examples import build_SB_example35, build_SB_example41
import numpy as np

def test_V_pi():
    mdp = build_SB_example35()

    print(mdp.R)
    # random policy:
    policy = np.ones((mdp.P.shape[0], mdp.P.shape[1]))/mdp.P.shape[1]

    V_pi = calculate_V_pi(mdp.P, mdp.R, policy, mdp.gamma)

    assert np.allclose(np.round(V_pi, 1), np.array([3.3, 8.8, 4.4, 5.3, 1.5,
                                       1.5, 3.0, 2.3, 1.9, 0.5,
                                       0.1, 0.7, 0.7, 0.4, -0.4,
                                       -1.0, -0.4, -0.4, -0.6, -1.2,
                                       -1.9, -1.3, -1.2, -1.4, -2.0]))

    # now test example 4.1
    mdp = build_SB_example41()
    # random policy:
    policy = np.ones((mdp.P.shape[0], mdp.P.shape[1])) / mdp.P.shape[1]

    # since this example had terminal state, an absorbing state was appended to the state space.
    # make sure to remove the added absorbing state i.e., take only the original (size x size) states
    V_pi = calculate_V_pi(mdp.P[:-1, ..., :-1], mdp.R[:-1, ...], policy[:-1, ...], mdp.gamma)
    book_sol = np.array([[  0., -14., -20., -22.],
                         [-14., -18., -20., -20.],
                         [-20., -20., -18., -14.],
                         [-22., -20., -14.,   0.]])

    assert np.allclose(np.round(V_pi.reshape(mdp.size, mdp.size), 1), book_sol)



