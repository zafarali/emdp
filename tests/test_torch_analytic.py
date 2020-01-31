import pytest
from emdp.examples import build_SB_example35
import numpy as np

try:
    import torch
    from emdp.torch_analytic import calculate_V_pi
    torch_imported = True
except ImportError:
    # Torch test will not be applied.
    torch_imported = False

@pytest.mark.skipif(not torch_imported, reason='Torch not imported')
def test_V_pi():
    """Check if computation works."""
    mdp = build_SB_example35()

    print(mdp.R)
    # random policy:
    policy = np.ones((mdp.P.shape[0], mdp.P.shape[1]))/mdp.P.shape[1]
    policy = torch.from_numpy(policy).float()
    V_pi = calculate_V_pi(mdp.P, mdp.R, policy, mdp.gamma).detach().numpy()

    assert np.allclose(np.round(V_pi, 1), np.array([3.3, 8.8, 4.4, 5.3, 1.5,
                                       1.5, 3.0, 2.3, 1.9, 0.5,
                                       0.1, 0.7, 0.7, 0.4, -0.4,
                                       -1.0, -0.4, -0.4, -0.6, -1.2,
                                       -1.9, -1.3, -1.2, -1.4, -2.0]))

@pytest.mark.skipif(not torch_imported, reason='Torch not imported')
def test_differentiable():
    mdp = build_SB_example35()

    print(mdp.R)
    # random policy:
    policy = np.ones((mdp.P.shape[0], mdp.P.shape[1]))/mdp.P.shape[1]
    policy = torch.tensor(policy, requires_grad=True).float()
    V_pi = calculate_V_pi(mdp.P, mdp.R, policy, mdp.gamma)
    grads = torch.autograd.grad(V_pi.mean(), [policy])
    assert grads is not None
    for grad in grads:
        assert torch.isfinite(grad).all()
        assert not torch.equal(grad, torch.tensor(0.0))

