import numpy as np
import pytest

from src.rl.algorithms.monte_carlo import on_policy_first_visit_monte_carlo_control
from src.rl.environments.grid_world import DOWN, GridWorldEnv
from src.rl.environments.line_world import LineWorldEnv


def test_on_policy_mc_control_hand_verifiable_on_smallest_line_world():
    # Avec 3 cases, le seul état non terminal est 1 : aller à droite donne
    # directement +1, aller à gauche donne directement -1. La politique
    # epsilon-soft doit donc jouer "aller à droite" avec une très forte
    # probabilité (1 - epsilon + epsilon / nb_actions).
    np.random.seed(0)
    env = LineWorldEnv(num_cells=3)
    pi, Q = on_policy_first_visit_monte_carlo_control(env, iterations_count=2_000, epsilon=0.1)

    assert pi[1].argmax() == 1
    assert pi[1, 1] == pytest.approx(0.95, abs=1e-9)
    assert Q[1, 1] == pytest.approx(1.0, abs=0.05)


def test_on_policy_mc_control_terminal_states_have_no_action():
    np.random.seed(0)
    env = LineWorldEnv(num_cells=5)
    pi, Q = on_policy_first_visit_monte_carlo_control(env, iterations_count=5_000)

    for s in env.terminal_states():
        assert pi[s].sum() == 0.0


def test_on_policy_mc_control_optimal_policy_always_goes_right_in_line_world():
    np.random.seed(0)
    env = LineWorldEnv(num_cells=5)
    pi, Q = on_policy_first_visit_monte_carlo_control(env, iterations_count=8_000)

    for s in range(1, env.num_cells - 1):
        assert pi[s].argmax() == 1


def test_on_policy_mc_control_hand_verifiable_on_smallest_grid_world():
    # Grille 2x2 : 0=(0,0) départ, 1=(0,1), 2=(1,0)=piège, 3=(1,1)=objectif.
    # Depuis 0, aller directement en bas tombe dans le piège (-1) ; il faut
    # d'abord aller à droite (vers 1) puis descendre pour atteindre l'objectif (+1).
    np.random.seed(0)
    env = GridWorldEnv(height=2, width=2, goal=(1, 1), trap=(1, 0))
    pi, Q = on_policy_first_visit_monte_carlo_control(env, iterations_count=5_000)

    assert pi[0].argmax() == 1  # RIGHT
    assert pi[1].argmax() == DOWN
