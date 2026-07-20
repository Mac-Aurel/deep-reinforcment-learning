from typing import Tuple

import numpy as np

from src.rl.envs import ExploringStartsEnvironment


# Monte Carlo ES (Exploring Starts) : pour garantir que chaque paire
# (état, action) est explorée sans avoir besoin d'une politique de
# comportement aléatoire, chaque épisode démarre d'un état ET d'une action
# tirés au hasard (l'environnement doit donc savoir se placer directement
# dans un état donné). Le reste de l'épisode suit ensuite la politique
# gloutonne courante `pi`. À la fin de chaque épisode, Q(s, a) est mis à jour
# par la moyenne des retours de première visite pour chaque paire rencontrée,
# et la politique est rendue gloutonne par rapport à ce nouveau Q.
#
# `max_steps_per_episode` protège contre un cas piège : tant que la politique
# gloutonne courante n'est pas optimale, elle peut enfermer l'agent dans une
# boucle entre états non terminaux (ex: un état où l'action gloutonne le
# ramène sur lui-même) sans jamais atteindre d'état terminal. Monte Carlo a
# besoin d'un retour final pour apprendre : un épisode qui n'a pas terminé
# dans cette limite est simplement abandonné (pas de mise à jour de Q), comme
# le cap déjà utilisé pour l'évaluation de politique en programmation dynamique.
def monte_carlo_es(
    env: ExploringStartsEnvironment,
    iterations_count: int = 10_000,
    gamma: float = 0.999999,
    max_steps_per_episode: int = 1_000,
) -> Tuple[np.ndarray, np.ndarray]:
    num_states = env.num_states()
    num_actions = env.num_actions()

    Q = np.random.random((num_states, num_actions))
    Q_counts = np.zeros((num_states, num_actions))
    greedy_action = np.random.randint(0, num_actions, size=num_states)
    terminal_states_seen = set()

    for _ in range(iterations_count):
        # Exploring start : on retire un état au hasard jusqu'à en trouver un non terminal,
        # puis on force une première action elle aussi tirée au hasard.
        while True:
            env.reset()
            start_state = int(np.random.randint(num_states))
            env.set_state(start_state)
            if not env.is_game_over():
                break
        start_action = int(np.random.choice(env.available_actions()))

        trajectory_states = [start_state]
        trajectory_actions = [start_action]
        trajectory_rewards = []

        prev_score = env.score()
        env.step(start_action)
        trajectory_rewards.append(env.score() - prev_score)
        if env.is_game_over():
            terminal_states_seen.add(env.current_state())

        steps = 1
        while not env.is_game_over() and steps < max_steps_per_episode:
            s = env.current_state()
            a = int(greedy_action[s])
            prev_score = env.score()
            env.step(a)
            trajectory_states.append(s)
            trajectory_actions.append(a)
            trajectory_rewards.append(env.score() - prev_score)
            if env.is_game_over():
                terminal_states_seen.add(env.current_state())
            steps += 1

        if not env.is_game_over():
            continue

        G = 0.0
        t = len(trajectory_states) - 1
        for s, a, r in zip(reversed(trajectory_states), reversed(trajectory_actions), reversed(trajectory_rewards)):
            G = gamma * G + r
            if (s, a) not in zip(trajectory_states[:t], trajectory_actions[:t]):
                Q_counts[s, a] += 1
                Q[s, a] += (G - Q[s, a]) / Q_counts[s, a]
                greedy_action[s] = np.argmax(Q[s])
            t -= 1

    pi = np.zeros((num_states, num_actions))
    pi[np.arange(num_states), greedy_action] = 1.0
    for s in terminal_states_seen:
        pi[s, :] = 0.0

    return pi, Q
