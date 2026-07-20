from typing import Tuple

import numpy as np

from src.rl.envs import MDPEnvironment


# Calcule le retour espéré si on joue l'action `a` dans l'état `s`, en supposant
# connues les valeurs `V` de tous les états suivants possibles. C'est la brique
# de base réutilisée par les deux algorithmes de programmation dynamique.
def _action_value(env: MDPEnvironment, s: int, a: int, V: np.ndarray, gamma: float) -> float:
    rewards = env.rewards()
    total = 0.0
    for s_p in env.states():
        for r_index in range(len(rewards)):
            total += env.transition_probability(s, a, s_p, r_index) * (rewards[r_index] + gamma * V[s_p])
    return total


# À partir d'une fonction de valeur V, construit la politique gloutonne : dans
# chaque état, on choisit l'action qui mène au meilleur retour espéré selon V.
def _greedy_policy_from_v(env: MDPEnvironment, V: np.ndarray, gamma: float) -> np.ndarray:
    states = env.states()
    actions = env.actions()
    terminal_states = set(env.terminal_states())

    pi = np.zeros((len(states), len(actions)))
    for s in states:
        if s in terminal_states:
            continue
        action_values = [_action_value(env, s, a, V, gamma) for a in actions]
        best_a = int(np.argmax(action_values))
        pi[s, best_a] = 1.0
    return pi


# Value Iteration : au lieu d'alterner évaluation puis amélioration d'une politique
# (comme le fait Policy Iteration), on met directement à jour chaque état avec la
# valeur de sa MEILLEURE action, jusqu'à ce que la fonction de valeur ne bouge plus
# (delta < theta). La politique optimale est ensuite déduite de cette valeur finale.
def value_iteration(
    env: MDPEnvironment,
    gamma: float = 0.999999,
    theta: float = 0.000001,
) -> Tuple[np.ndarray, np.ndarray]:
    states = env.states()
    actions = env.actions()
    terminal_states = set(env.terminal_states())

    V = np.random.random(len(states))
    for s in terminal_states:
        V[s] = 0.0

    while True:
        delta = 0.0
        for s in states:
            if s in terminal_states:
                continue
            v = V[s]
            V[s] = max(_action_value(env, s, a, V, gamma) for a in actions)
            delta = max(delta, abs(v - V[s]))
        if delta < theta:
            break

    pi = _greedy_policy_from_v(env, V, gamma)
    return pi, V
