from emdp.gridworld import  builder_tools
from emdp import actions
import numpy as np

TEST_CASE_SIZE = 3

def _create_tmb():
    tmb = builder_tools.TransitionMatrixBuilder(
        TEST_CASE_SIZE, has_terminal_state=False)
    tmb.add_grid(p_success=0.9)
    return tmb

def test_corner_wall():
    # Wall in top left corner.
    tmb = _create_tmb()
    tmb.add_wall_at((0, 0))

    assert np.allclose(tmb.P[0, :, 0], 1.0), 'All actions from wall state should lead to wall state'
    assert np.allclose(tmb.P[1, :, 0], 0.0), 'You cannot transition into the wall.'
    assert np.allclose(tmb.P[1, actions.RIGHT, 2], 0.9), 'Correct action with p_success.'

def test_middle_wall():
    # Wall in center of the world.
    tmb = _create_tmb()
    tmb.add_wall_at((1, 1))

    assert np.allclose(tmb.P[4, :, 4], 1.0), 'All actions from wall state should lead to wall state'
    assert np.isclose(tmb.P[3, actions.RIGHT, 4], 0.0), 'You cannot transition into the wall.'

def test_two_walls_sandwich():
    # Wall in center of the world.
    tmb = _create_tmb()
    tmb.add_wall_at((0, 0))
    tmb.add_wall_at((1, 1))

    assert np.isclose(tmb.P[1, actions.LEFT, 0], 0.0), 'Should not transition into first wall.'
    assert np.isclose(tmb.P[1, actions.DOWN, 4], 0.0), 'Should not transition into second wall'
    assert np.isclose(tmb.P[1, actions.RIGHT, 2], 0.9), 'Correct action with p_success.'
    assert np.isclose(tmb.P[1, actions.RIGHT, 1], 0.1), 'p_fail should put all mass on state since it is sandwiched.'
