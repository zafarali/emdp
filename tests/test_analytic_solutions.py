from emdp.analytic import calculate_V_pi
from emdp.examples import build_SB_example35
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

