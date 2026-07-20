from typing import Tuple

import numpy as np

from src.rl.envs import Environment


# Choisit une action dans l'état s selon une politique epsilon-gloutonne par
# rapport à Q : la plupart du temps la meilleure action connue, mais avec une
# probabilité epsilon une action au hasard parmi celles disponibles, pour
# continuer à explorer plutôt que de s'enfermer trop vite dans un choix
# sous-optimal.
def _epsilon_greedy_action(env: Environment, Q: np.ndarray, s: int, epsilon: float) -> int:
    available = env.available_actions()
    if np.random.random() < epsilon:
        return int(np.random.choice(available))
    q_available = Q[s, available]
    return int(available[int(np.argmax(q_available))])


# Sarsa (State-Action-Reward-State-Action) : contrairement aux méthodes Monte
# Carlo, qui attendent la fin d'un épisode complet pour connaître le vrai
# retour G et mettre Q à jour, Sarsa met à jour Q(s, a) après CHAQUE pas, en
# se servant de la valeur estimée Q(s', a') du pas suivant comme approximation
# du retour restant (c'est le "bootstrap" du Temporal Difference Learning).
# C'est une méthode on-policy : l'action a' utilisée pour cette estimation est
# la même qui sera réellement jouée au pas suivant (choisie par la politique
# epsilon-gloutonne courante), d'où le nom Sarsa. Un état terminal n'a pas
# d'action future : sa valeur restante est donc prise à 0 (Q(s, a) est juste
# mis à jour vers la récompense reçue).
def sarsa(
    env: Environment,
    iterations_count: int = 10_000,
    gamma: float = 0.999999,
    alpha: float = 0.1,
    epsilon: float = 0.1,
    max_steps_per_episode: int = 1_000,
) -> Tuple[np.ndarray, np.ndarray]:
    num_states = env.num_states()
    num_actions = env.num_actions()

    Q = np.random.random((num_states, num_actions))
    terminal_states_seen = set()

    for _ in range(iterations_count):
        env.reset()
        s = env.current_state()
        a = _epsilon_greedy_action(env, Q, s, epsilon)

        # Même garde-fou que pour les méthodes Monte Carlo : si la politique
        # courante boucle entre des états non terminaux, on abandonne
        # l'épisode plutôt que de tourner à l'infini.
        steps = 0
        while not env.is_game_over() and steps < max_steps_per_episode:
            prev_score = env.score()
            env.step(a)
            r = env.score() - prev_score
            s_p = env.current_state()

            if env.is_game_over():
                terminal_states_seen.add(s_p)
                Q[s, a] += alpha * (r - Q[s, a])
            else:
                a_p = _epsilon_greedy_action(env, Q, s_p, epsilon)
                Q[s, a] += alpha * (r + gamma * Q[s_p, a_p] - Q[s, a])
                s, a = s_p, a_p

            steps += 1

    pi = np.zeros((num_states, num_actions))
    pi[np.arange(num_states), np.argmax(Q, axis=1)] = 1.0
    for s in terminal_states_seen:
        pi[s, :] = 0.0

    return pi, Q
