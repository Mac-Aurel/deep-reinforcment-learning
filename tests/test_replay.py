import numpy as np

from src.rl.environments.line_world import LineWorldEnv
from src.rl.replay import replay_policy


def test_replay_policy_follows_the_policy_and_reaches_the_goal():
    env = LineWorldEnv(num_cells=3)
    pi = np.array([[0, 0], [0, 1], [0, 0]])  # depuis l'état 1, toujours aller à droite

    score = replay_policy(env, pi, pause=None)

    assert score == 1.0
    assert env.is_game_over()


def test_replay_policy_follows_a_losing_policy_just_as_faithfully():
    env = LineWorldEnv(num_cells=3)
    pi = np.array([[0, 0], [1, 0], [0, 0]])  # depuis l'état 1, toujours aller à gauche

    score = replay_policy(env, pi, pause=None)

    assert score == -1.0


def test_replay_policy_calls_pause_exactly_once_per_step():
    env = LineWorldEnv(num_cells=3)
    pi = np.array([[0, 0], [0, 1], [0, 0]])
    calls = []

    replay_policy(env, pi, pause=lambda: calls.append(1))

    # Depuis l'état 1 (milieu d'une ligne à 3 cases), un seul pas à droite
    # suffit pour atteindre la sortie : un seul appel à pause() attendu.
    assert len(calls) == 1


def test_replay_policy_does_not_hang_on_a_never_terminating_policy():
    # Politique invalide qui ne mène jamais à un état terminal (aller à
    # gauche puis à droite en boucle) : le garde-fou max_steps doit arrêter
    # le déroulé au lieu de tourner indéfiniment.
    env = LineWorldEnv(num_cells=5)
    pi = np.zeros((5, 2))
    pi[1, 1] = 1.0  # état 1 -> droite (état 2)
    pi[2, 0] = 1.0  # état 2 -> gauche (état 1) : boucle infinie 1 <-> 2

    score = replay_policy(env, pi, pause=None, max_steps=50)

    assert not env.is_game_over()
    assert score == 0.0
