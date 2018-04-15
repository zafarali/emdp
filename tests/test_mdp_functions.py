from emdp.common import MDP, EpisodeDoneError
import numpy as np

def test_simple_reset_MDP():
    P = np.array([[[1, 0], [0, 1]], # LEFT action results in the same state, RIGHT next state.
                  [[0, 1], [0, 1]]]) # from terminal state, any action goes to the same state.
    p0 = np.array([1, 0])
    R = np.array([[0, 5], # RIGHT action from state 0 gives +5 reward.
                  [0, 0]])
    # 2 state MDP
    # where transitioning into state 1 will terminate and get +1 reward
    mdp = MDP(P,R, 0.9, p0, [1])

    # check if we are indeed in the starting state:
    assert np.all(np.equal(mdp.current_state, np.array([1, 0])))

    # simulate an episode
    state, reward, done, _ = mdp.step(0) # left step, no end
    assert np.all(np.equal(state, np.array([1, 0])))
    assert reward == 0
    assert not done

    # simulate another step
    state, reward, done, _ = mdp.step(1) # right step
    assert np.all(np.equal(state, np.array([0, 1])))
    assert reward == +5
    assert not done

    # simulate another step (should return done)
    state, reward, done, _ = mdp.step(1)  # right step
    assert np.all(np.equal(state, np.array([0, 1])))
    assert reward == 0
    assert done

    try:
        mdp.step(0)
        assert False, 'This should throw an EpisodeDoneError'
    except EpisodeDoneError:
        assert True
