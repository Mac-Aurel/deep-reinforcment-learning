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
    max_iterations: int = 10_000,
) -> Tuple[np.ndarray, np.ndarray]:
    states = env.states()
    actions = env.actions()
    terminal_states = set(env.terminal_states())

    V = np.random.random(len(states))
    for s in terminal_states:
        V[s] = 0.0

    for _ in range(max_iterations):
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


# Calcule la valeur de chaque état pour une politique `pi` fixée, en répétant
# la mise à jour "moyenne pondérée par pi des actions" jusqu'à ce que V ne
# bouge plus (delta < theta). Utilisé comme étape d'évaluation par Policy Iteration.
#
# `max_iterations` protège contre un cas piège : une politique candidate (en
# particulier la politique initiale, tirée au hasard action par action) peut
# faire boucler l'agent indéfiniment entre des états non terminaux (ex: état 1
# -> état 2 -> état 1 -> ...), sans jamais atteindre de récompense. Avec un
# gamma très proche de 1 (0.999999 par défaut), converger exactement en dessous
# de theta sur ces états qui bouclent peut demander des millions de passes. La
# politique n'a de toute façon aucune chance d'être optimale dans ce cas : la
# borne évite l'attente et l'étape d'amélioration corrigera la politique ensuite.
def _evaluate_policy(
    env: MDPEnvironment,
    pi: np.ndarray,
    V: np.ndarray,
    gamma: float,
    theta: float,
    max_iterations: int = 10_000,
) -> None:
    states = env.states()
    actions = env.actions()
    terminal_states = set(env.terminal_states())

    for _ in range(max_iterations):
        delta = 0.0
        for s in states:
            if s in terminal_states:
                continue
            v = V[s]
            V[s] = sum(pi[s, a] * _action_value(env, s, a, V, gamma) for a in actions)
            delta = max(delta, abs(v - V[s]))
        if delta < theta:
            break


# Policy Iteration : alterne deux étapes jusqu'à ce que la politique ne change
# plus. 1) Évaluation : calculer V pour la politique actuelle. 2) Amélioration :
# rendre la politique gloutonne par rapport à ce V. Contrairement à Value
# Iteration, chaque étape d'évaluation va jusqu'à convergence avant d'améliorer.
#
# `max_evaluation_iterations` reste volontairement bas (200, contre 10 000
# pour Value Iteration) : la toute première politique est tirée au hasard
# action par action, et peut donc enfermer certains états dans une boucle
# sans intérêt (ex: un mur qu'on heurte sans arrêt dans Grid World). Avec un
# gamma aussi proche de 1, évaluer EXACTEMENT une telle politique jusqu'à
# theta demanderait des millions de passes pour un gain quasi nul, alors
# qu'une évaluation partielle suffit déjà à repérer la bonne direction et à
# corriger la politique dès l'étape d'amélioration suivante (vérifié : sur
# Grid World, converge vers exactement la même valeur que Value Iteration en
# une fraction de seconde, quel que soit le tirage aléatoire initial).
# `max_policy_iterations` protège en plus la boucle d'amélioration
# elle-même : si la politique ne se stabilise jamais, on s'arrête plutôt que
# de tourner indéfiniment.
def policy_iteration(
    env: MDPEnvironment,
    gamma: float = 0.999999,
    theta: float = 0.000001,
    max_evaluation_iterations: int = 200,
    max_policy_iterations: int = 1_000,
) -> Tuple[np.ndarray, np.ndarray]:
    states = env.states()
    actions = env.actions()
    terminal_states = set(env.terminal_states())

    V = np.random.random(len(states))
    for s in terminal_states:
        V[s] = 0.0

    pi = np.zeros((len(states), len(actions)))
    for s in states:
        if s in terminal_states:
            continue
        pi[s, np.random.randint(len(actions))] = 1.0

    for _ in range(max_policy_iterations):
        _evaluate_policy(env, pi, V, gamma, theta, max_evaluation_iterations)

        policy_stable = True
        for s in states:
            if s in terminal_states:
                continue
            old_action = int(np.argmax(pi[s]))
            action_values = [_action_value(env, s, a, V, gamma) for a in actions]
            best_action = int(np.argmax(action_values))
            pi[s, :] = 0.0
            pi[s, best_action] = 1.0
            if old_action != best_action:
                policy_stable = False

        if policy_stable:
            break

    return pi, V
