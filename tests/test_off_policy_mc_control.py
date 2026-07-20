import numpy as np
import pytest

from src.rl.algorithms.monte_carlo import off_policy_monte_carlo_control
from src.rl.environments.grid_world import DOWN, GridWorldEnv
from src.rl.environments.line_world import LineWorldEnv


def test_off_policy_mc_control_hand_verifiable_on_smallest_line_world():
    # Avec 3 cases, le seul état non terminal est 1 : aller à droite donne
    # directement +1, aller à gauche donne directement -1. La politique
    # cible apprise doit converger vers "toujours aller à droite".
    np.random.seed(0)
    env = LineWorldEnv(num_cells=3)
    pi, Q = off_policy_monte_carlo_control(env, iterations_count=2_000)

    assert pi[1, 1] == 1.0
    assert pi[1, 0] == 0.0
    assert Q[1, 1] == pytest.approx(1.0, abs=0.05)


def test_off_policy_mc_control_terminal_states_have_no_action():
    np.random.seed(0)
    env = LineWorldEnv(num_cells=5)
    pi, Q = off_policy_monte_carlo_control(env, iterations_count=5_000)

    for s in env.terminal_states():
        assert pi[s].sum() == 0.0


def test_off_policy_mc_control_optimal_policy_always_goes_right_in_line_world():
    np.random.seed(0)
    env = LineWorldEnv(num_cells=5)
    pi, Q = off_policy_monte_carlo_control(env, iterations_count=8_000)

    for s in range(1, env.num_cells - 1):
        assert pi[s, 1] == 1.0


def test_off_policy_mc_control_hand_verifiable_on_smallest_grid_world():
    # Grille 2x2 : 0=(0,0) départ, 1=(0,1), 2=(1,0)=piège, 3=(1,1)=objectif.
    # Depuis 0, aller directement en bas tombe dans le piège (-1) ; il faut
    # d'abord aller à droite (vers 1) puis descendre pour atteindre l'objectif (+1).
    np.random.seed(0)
    env = GridWorldEnv(height=2, width=2, goal=(1, 1), trap=(1, 0))
    pi, Q = off_policy_monte_carlo_control(env, iterations_count=5_000)

    assert pi[0].argmax() == 1  # RIGHT
    assert pi[1].argmax() == DOWN
