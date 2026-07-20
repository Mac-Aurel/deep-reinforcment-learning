from typing import Dict, List, Tuple

import numpy as np

from src.rl.algorithms.td import build_greedy_policy, epsilon_greedy_action, restricted_max
from src.rl.envs import Environment


# Dyna-Q : reprend exactement la mise à jour de Q-Learning après chaque pas
# réellement joué dans l'environnement (l'apprentissage "direct"), mais y
# ajoute une étape de "planification" : à chaque pas, on mémorise dans un
# modèle ce qui s'est passé (depuis s, jouer a a mené à s' avec la récompense
# r), puis on rejoue `planning_steps` transitions tirées au hasard parmi
# celles déjà mémorisées, comme si on les vivait à nouveau, pour mettre Q à
# jour plusieurs fois à partir de la même expérience réelle. Ça accélère
# l'apprentissage sans avoir besoin de solliciter davantage l'environnement.
# Le modèle suppose l'environnement déterministe (il ne retient que la
# dernière transition observée pour chaque (s, a)), ce qui est vrai pour Line
# World et Grid World.
def dyna_q(
    env: Environment,
    iterations_count: int = 10_000,
    gamma: float = 0.999999,
    alpha: float = 0.1,
    epsilon: float = 0.1,
    planning_steps: int = 10,
    max_steps_per_episode: int = 1_000,
) -> Tuple[np.ndarray, np.ndarray]:
    num_states = env.num_states()
    num_actions = env.num_actions()

    Q = np.random.random((num_states, num_actions))
    terminal_states_seen = set()
    known_available_actions: Dict[int, List[int]] = {}
    # (s, a) -> (r, s', s' terminal ?, actions disponibles depuis s')
    model: Dict[Tuple[int, int], Tuple[float, int, bool, List[int]]] = {}

    for _ in range(iterations_count):
        env.reset()
        s = env.current_state()

        # Même garde-fou que pour les méthodes Monte Carlo et TD Learning :
        # si la politique courante boucle entre des états non terminaux, on
        # abandonne l'épisode plutôt que de tourner à l'infini.
        steps = 0
        while not env.is_game_over() and steps < max_steps_per_episode:
            known_available_actions[s] = env.available_actions()
            a = epsilon_greedy_action(env, Q, s, epsilon)
            prev_score = env.score()
            env.step(a)
            r = env.score() - prev_score
            s_p = env.current_state()
            is_terminal = env.is_game_over()
            available_at_s_p = [] if is_terminal else env.available_actions()
            if is_terminal:
                terminal_states_seen.add(s_p)
            else:
                known_available_actions[s_p] = available_at_s_p

            target = r if is_terminal else r + gamma * restricted_max(Q[s_p], available_at_s_p)
            Q[s, a] += alpha * (target - Q[s, a])
            model[(s, a)] = (r, s_p, is_terminal, available_at_s_p)

            known_pairs = list(model.keys())
            for _ in range(planning_steps):
                ps, pa = known_pairs[np.random.randint(len(known_pairs))]
                p_r, p_s_p, p_is_terminal, p_available = model[(ps, pa)]
                p_target = p_r if p_is_terminal else p_r + gamma * restricted_max(Q[p_s_p], p_available)
                Q[ps, pa] += alpha * (p_target - Q[ps, pa])

            s = s_p
            steps += 1

    pi = build_greedy_policy(Q, known_available_actions)
    for s in terminal_states_seen:
        pi[s, :] = 0.0

    return pi, Q
