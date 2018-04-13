from emdp.examples import build_SB_example35
from emdp import actions

def test_SB_example35():

    mdp = build_SB_example35()

    mdp.set_current_state_to((0, 0))
    state, reward, done, _ = mdp.step(actions.UP)
    assert not done
    assert reward == 0
    assert mdp.unflatten_state(state) == (0, 0)

    state, reward, done, _ = mdp.step(actions.RIGHT)
    assert not done
    assert reward == 0
    assert mdp.unflatten_state(state) == (0, 1)

    state, reward, done, _ = mdp.step(actions.RIGHT)
    assert not done
    assert reward == +10
    assert mdp.unflatten_state(state) == (4, 1)

