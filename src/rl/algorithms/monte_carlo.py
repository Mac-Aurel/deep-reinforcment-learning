from typing import Dict, List, Tuple

import numpy as np

from src.rl.algorithms.td import build_greedy_policy, restricted_argmax
from src.rl.envs import Environment, ExploringStartsEnvironment


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
    terminal_states_seen = set()

    # Certains environnements n'offrent pas les mêmes actions dans tous les
    # états (ex: Monty Hall, où le tout premier choix porte sur bien plus
    # d'options que les décisions "garder/changer" qui suivent) : on relève
    # donc à l'avance les actions valides état par état, pour ne jamais
    # piocher au hasard une action que l'environnement refuserait.
    valid_actions = []
    for s in range(num_states):
        env.reset()
        env.set_state(s)
        valid_actions.append([] if env.is_game_over() else env.available_actions())

    greedy_action = np.array(
        [int(np.random.choice(valid_actions[s])) if valid_actions[s] else 0 for s in range(num_states)]
    )

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
                best_index = int(np.argmax(Q[s, valid_actions[s]]))
                greedy_action[s] = valid_actions[s][best_index]
            t -= 1

    pi = np.zeros((num_states, num_actions))
    pi[np.arange(num_states), greedy_action] = 1.0
    for s in terminal_states_seen:
        pi[s, :] = 0.0

    return pi, Q


# On-policy first visit Monte Carlo Control : contrairement à Monte Carlo ES,
# on ne triche pas en démarrant l'épisode sur un état choisi à la main. Chaque
# épisode part donc toujours du même état de départ (reset()), et c'est la
# politique elle-même qui doit garantir l'exploration : elle est "epsilon-soft"
# (epsilon-greedy), c'est-à-dire qu'elle joue l'action jugée la meilleure avec
# une forte probabilité, mais garde toujours une petite chance epsilon de
# jouer une action au hasard pour continuer à découvrir l'environnement. Comme
# pour Monte Carlo ES, Q(s, a) est mis à jour par la moyenne des retours de
# première visite, et la politique est rendue gloutonne (à epsilon près) par
# rapport à ce nouveau Q après chaque épisode.
def on_policy_first_visit_monte_carlo_control(
    env: Environment,
    iterations_count: int = 10_000,
    gamma: float = 0.999999,
    epsilon: float = 0.1,
    max_steps_per_episode: int = 1_000,
) -> Tuple[np.ndarray, np.ndarray]:
    num_states = env.num_states()
    num_actions = env.num_actions()

    Q = np.random.random((num_states, num_actions))
    Q_counts = np.zeros((num_states, num_actions))
    # pi n'est initialisée qu'au fil de l'eau (voir plus bas) : certains
    # environnements (ex: Monty Hall) n'offrent pas les mêmes actions dans
    # tous les états, donc une distribution uniforme sur num_actions
    # n'aurait pas de sens pour un état qui n'en propose que quelques-unes.
    pi = np.zeros((num_states, num_actions))
    terminal_states_seen = set()

    for _ in range(iterations_count):
        env.reset()

        trajectory_states = []
        trajectory_actions = []
        trajectory_rewards = []
        trajectory_available_actions = []

        # Même garde-fou que pour Monte Carlo ES : si l'épisode ne termine pas
        # dans cette limite, on l'abandonne plutôt que de tourner à l'infini.
        steps = 0
        while not env.is_game_over() and steps < max_steps_per_episode:
            s = env.current_state()
            available = env.available_actions()
            probs = pi[s, available]
            probs = probs / probs.sum() if probs.sum() > 0 else np.full(len(available), 1.0 / len(available))
            a = int(np.random.choice(available, p=probs))

            prev_score = env.score()
            env.step(a)
            trajectory_states.append(s)
            trajectory_actions.append(a)
            trajectory_rewards.append(env.score() - prev_score)
            trajectory_available_actions.append(available)
            if env.is_game_over():
                terminal_states_seen.add(env.current_state())
            steps += 1

        if not env.is_game_over():
            continue

        G = 0.0
        t = len(trajectory_states) - 1
        for s, a, r, available in zip(
            reversed(trajectory_states),
            reversed(trajectory_actions),
            reversed(trajectory_rewards),
            reversed(trajectory_available_actions),
        ):
            G = gamma * G + r
            if (s, a) not in zip(trajectory_states[:t], trajectory_actions[:t]):
                Q_counts[s, a] += 1
                Q[s, a] += (G - Q[s, a]) / Q_counts[s, a]

                best_a = restricted_argmax(Q[s], available)
                pi[s, available] = epsilon / len(available)
                pi[s, best_a] = 1 - epsilon + epsilon / len(available)
            t -= 1

    for s in terminal_states_seen:
        pi[s, :] = 0.0

    return pi, Q


# Off-policy Monte Carlo Control : ici on sépare complètement la politique qui
# joue les épisodes (la politique "de comportement", b, tirée uniformément au
# hasard parmi les actions disponibles pour bien tout explorer) de la
# politique que l'on cherche à évaluer et améliorer (la politique "cible",
# gloutonne par rapport à Q). Comme les deux diffèrent, chaque retour observé
# doit être repondéré par un ratio d'échantillonnage préférentiel (importance
# sampling) W qui corrige le fait qu'on n'a pas joué avec la politique cible.
# En partant de la fin de l'épisode, W part de 1 et se multiplie par
# 1 / b(action jouée | état) à chaque pas ; dès que l'action jouée ne
# correspond plus à l'action gloutonne actuelle (la politique cible n'aurait
# pas joué ce coup-là), le reste de l'épisode (plus ancien) n'apporte plus
# d'information utile sur la politique cible et on l'ignore.
def off_policy_monte_carlo_control(
    env: Environment,
    iterations_count: int = 10_000,
    gamma: float = 0.999999,
    max_steps_per_episode: int = 1_000,
) -> Tuple[np.ndarray, np.ndarray]:
    num_states = env.num_states()
    num_actions = env.num_actions()

    Q = np.random.random((num_states, num_actions))
    C = np.zeros((num_states, num_actions))
    terminal_states_seen = set()
    known_available_actions: Dict[int, List[int]] = {}

    for _ in range(iterations_count):
        env.reset()

        trajectory_states = []
        trajectory_actions = []
        trajectory_rewards = []
        trajectory_behavior_probs = []
        trajectory_available_actions = []

        # Même garde-fou que pour les deux autres méthodes Monte Carlo : si
        # l'épisode ne termine pas dans cette limite, on l'abandonne plutôt
        # que de tourner à l'infini.
        steps = 0
        while not env.is_game_over() and steps < max_steps_per_episode:
            s = env.current_state()
            available = env.available_actions()
            known_available_actions[s] = available
            a = int(np.random.choice(available))
            prev_score = env.score()
            env.step(a)
            trajectory_states.append(s)
            trajectory_actions.append(a)
            trajectory_rewards.append(env.score() - prev_score)
            trajectory_behavior_probs.append(1.0 / len(available))
            trajectory_available_actions.append(available)
            if env.is_game_over():
                terminal_states_seen.add(env.current_state())
            steps += 1

        if not env.is_game_over():
            continue

        G = 0.0
        W = 1.0
        for s, a, r, b_prob, available in zip(
            reversed(trajectory_states),
            reversed(trajectory_actions),
            reversed(trajectory_rewards),
            reversed(trajectory_behavior_probs),
            reversed(trajectory_available_actions),
        ):
            G = gamma * G + r
            C[s, a] += W
            Q[s, a] += (W / C[s, a]) * (G - Q[s, a])

            if a != restricted_argmax(Q[s], available):
                break
            W /= b_prob

    pi = build_greedy_policy(Q, known_available_actions)
    for s in terminal_states_seen:
        pi[s, :] = 0.0

    return pi, Q
