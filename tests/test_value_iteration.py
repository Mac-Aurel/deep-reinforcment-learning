import pytest

from src.rl.algorithms.dynamic_programming import value_iteration
from src.rl.environments.line_world import LineWorldEnv


def test_value_iteration_hand_verifiable_on_smallest_line_world():
    # Avec 3 cases, le seul état non terminal est 1 : aller à droite donne
    # directement +1, aller à gauche donne directement -1. La valeur optimale
    # de l'état 1 doit donc être exactement 1.0, quel que soit gamma.
    env = LineWorldEnv(num_cells=3)
    pi, V = value_iteration(env)

    assert V[1] == pytest.approx(1.0)
    assert pi[1, 1] == 1.0
    assert pi[1, 0] == 0.0


def test_value_iteration_terminal_states_have_zero_value_and_no_action():
    env = LineWorldEnv(num_cells=5)
    pi, V = value_iteration(env)

    for s in env.terminal_states():
        assert V[s] == 0.0
        assert pi[s].sum() == 0.0


def test_value_iteration_optimal_policy_always_goes_right_in_line_world():
    # Sur Line World, quel que soit l'état de départ, avancer vers la droite
    # atteint toujours la récompense +1 sans jamais risquer le -1 : la
    # politique optimale doit donc être "aller à droite" partout.
    env = LineWorldEnv(num_cells=5)
    pi, V = value_iteration(env)

    for s in range(1, env.num_cells - 1):
        assert pi[s, 1] == 1.0
        assert V[s] == pytest.approx(1.0, abs=1e-3)
