import pytest

from src.rl.algorithms.dynamic_programming import policy_iteration, value_iteration
from src.rl.environments.line_world import LineWorldEnv


def test_policy_iteration_hand_verifiable_on_smallest_line_world():
    # Même cas simple que pour Value Iteration : avec 3 cases, l'état 1 doit
    # obtenir une valeur optimale de 1.0 en choisissant toujours l'action droite.
    env = LineWorldEnv(num_cells=3)
    pi, V = policy_iteration(env)

    assert V[1] == pytest.approx(1.0)
    assert pi[1, 1] == 1.0
    assert pi[1, 0] == 0.0


def test_policy_iteration_terminal_states_have_zero_value_and_no_action():
    env = LineWorldEnv(num_cells=5)
    pi, V = policy_iteration(env)

    for s in env.terminal_states():
        assert V[s] == 0.0
        assert pi[s].sum() == 0.0


def test_policy_iteration_optimal_policy_always_goes_right_in_line_world():
    env = LineWorldEnv(num_cells=5)
    pi, V = policy_iteration(env)

    for s in range(1, env.num_cells - 1):
        assert pi[s, 1] == 1.0
        assert V[s] == pytest.approx(1.0, abs=1e-3)


def test_policy_iteration_and_value_iteration_agree_on_line_world():
    # Les deux algorithmes de programmation dynamique doivent converger vers
    # la même politique optimale et la même fonction de valeur.
    env = LineWorldEnv(num_cells=5)
    pi_pol, V_pol = policy_iteration(env)
    pi_val, V_val = value_iteration(env)

    assert (pi_pol == pi_val).all()
    assert V_pol == pytest.approx(V_val, abs=1e-6)
