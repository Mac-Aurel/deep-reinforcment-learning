import numpy as np
import pytest

from src.rl.algorithms.planning import dyna_q
from src.rl.algorithms.td import q_learning
from src.rl.environments.grid_world import DOWN, GridWorldEnv
from src.rl.environments.line_world import LineWorldEnv


def test_dyna_q_hand_verifiable_on_smallest_line_world():
    # Avec 3 cases, le seul état non terminal est 1 : aller à droite donne
    # directement +1, aller à gauche donne directement -1. La politique
    # apprise doit converger vers "toujours aller à droite".
    np.random.seed(0)
    env = LineWorldEnv(num_cells=3)
    pi, Q = dyna_q(env, iterations_count=500)

    assert pi[1, 1] == 1.0
    assert pi[1, 0] == 0.0
    assert Q[1, 1] == pytest.approx(1.0, abs=0.05)


def test_dyna_q_terminal_states_have_no_action():
    np.random.seed(0)
    env = LineWorldEnv(num_cells=5)
    pi, Q = dyna_q(env, iterations_count=1_000)

    for s in env.terminal_states():
        assert pi[s].sum() == 0.0


def test_dyna_q_optimal_policy_always_goes_right_in_line_world():
    np.random.seed(0)
    env = LineWorldEnv(num_cells=5)
    pi, Q = dyna_q(env, iterations_count=1_500)

    for s in range(1, env.num_cells - 1):
        assert pi[s, 1] == 1.0


def test_dyna_q_hand_verifiable_on_smallest_grid_world():
    # Grille 2x2 : 0=(0,0) départ, 1=(0,1), 2=(1,0)=piège, 3=(1,1)=objectif.
    # Depuis 0, aller directement en bas tombe dans le piège (-1) ; il faut
    # d'abord aller à droite (vers 1) puis descendre pour atteindre l'objectif (+1).
    np.random.seed(0)
    env = GridWorldEnv(height=2, width=2, goal=(1, 1), trap=(1, 0))
    pi, Q = dyna_q(env, iterations_count=1_000)

    assert pi[0].argmax() == 1  # RIGHT
    assert pi[1].argmax() == DOWN


def test_dyna_q_converges_in_fewer_real_episodes_than_q_learning():
    # C'est tout l'intérêt de la planification : à nombre d'épisodes réels
    # égal (donc à nombre d'interactions avec l'environnement égal), Dyna-Q
    # doit avoir appris la politique optimale grâce aux mises à jour
    # supplémentaires simulées à partir du modèle, là où Q-Learning seul n'a
    # pas encore convergé.
    num_cells = 9
    iterations_count = 40

    np.random.seed(0)
    env = LineWorldEnv(num_cells=num_cells)
    pi_dyna, _ = dyna_q(env, iterations_count=iterations_count, planning_steps=20)
    dyna_converged = all(pi_dyna[s, 1] == 1.0 for s in range(1, num_cells - 1))

    np.random.seed(0)
    env = LineWorldEnv(num_cells=num_cells)
    pi_q, _ = q_learning(env, iterations_count=iterations_count)
    q_learning_converged = all(pi_q[s, 1] == 1.0 for s in range(1, num_cells - 1))

    assert dyna_converged
    assert not q_learning_converged
