import numpy as np
from emdp import actions
from emdp.gridworld.helper_utilities import (build_simple_grid,
                                             get_possible_actions,
                                             get_state_after_executing_action,
                                             check_can_take_action)

GRID_SIZE = 2 # check for a small grid of size 2x2
def test_check_can_take_action():
    assert not check_can_take_action(actions.LEFT, 0, GRID_SIZE)
    assert not check_can_take_action(actions.LEFT, 2, GRID_SIZE)
    assert not check_can_take_action(actions.RIGHT, 1, GRID_SIZE)
    assert not check_can_take_action(actions.RIGHT, 3, GRID_SIZE)
    assert not check_can_take_action(actions.DOWN, 2, GRID_SIZE)
    assert not check_can_take_action(actions.DOWN, 3, GRID_SIZE)
    assert not check_can_take_action(actions.UP, 0, GRID_SIZE)
    assert not check_can_take_action(actions.UP, 1, GRID_SIZE)

    assert check_can_take_action(actions.LEFT, 1, GRID_SIZE)
    assert check_can_take_action(actions.LEFT, 3, GRID_SIZE)
    assert check_can_take_action(actions.UP, 3, GRID_SIZE)
    assert check_can_take_action(actions.RIGHT, 0, GRID_SIZE)
    assert check_can_take_action(actions.RIGHT, 2, GRID_SIZE)
    assert check_can_take_action(actions.DOWN, 0, GRID_SIZE)
    assert check_can_take_action(actions.RIGHT, 2, GRID_SIZE)

def test_get_possible_actions():
    assert set([actions.LEFT, actions.DOWN]) == set(get_possible_actions(1, GRID_SIZE))
    assert set([actions.LEFT, actions.UP]) == set(get_possible_actions(3, GRID_SIZE))
    assert set([actions.RIGHT, actions.DOWN]) == set(get_possible_actions(0, GRID_SIZE))
    assert set([actions.UP, actions.RIGHT]) == set(get_possible_actions(2, GRID_SIZE))

def test_get_state_after_executing_action():
    assert get_state_after_executing_action(actions.RIGHT, 0, GRID_SIZE) == 1
    assert get_state_after_executing_action(actions.LEFT, 0, GRID_SIZE) == 0
    assert get_state_after_executing_action(actions.UP, 0, GRID_SIZE) == 0
    assert get_state_after_executing_action(actions.DOWN, 0, GRID_SIZE) == 2
    assert get_state_after_executing_action(actions.UP, 3, GRID_SIZE) == 1
    assert get_state_after_executing_action(actions.UP, 1, GRID_SIZE) == 1
    assert get_state_after_executing_action(actions.RIGHT, 1, GRID_SIZE) == 1
