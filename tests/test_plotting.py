"""Integration tests for plotting tools."""
from emdp import examples
from emdp.gridworld import GridWorldPlotter
from emdp import actions
import random
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def test_plotting_integration():
    mdp = examples.build_SB_example35()

    trajectories = []
    for _ in range(3):  # 3 trajectories
        trajectory = [mdp.reset()]
        for _ in range(10):  # 10 steps maximum
            state, reward, done, info = mdp.step(random.sample([actions.LEFT, actions.RIGHT,
                                                                actions.UP, actions.DOWN], 1)[0])
            trajectory.append(state)
        trajectories.append(trajectory)

    gwp = GridWorldPlotter(mdp.size,
                           mdp.has_absorbing_state)  # alternatively you can use GridWorldPlotter.from_mdp(mdp)
    fig = plt.figure(figsize=(10, 4))
    ax = fig.add_subplot(121)

    # trajectory
    gwp.plot_trajectories(ax, trajectories)
    gwp.plot_grid(ax)

    # heatmap
    ax = fig.add_subplot(122)
    gwp.plot_heatmap(ax, trajectories)
    gwp.plot_grid(ax)

