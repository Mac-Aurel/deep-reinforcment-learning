import os
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from src.rl.algorithms.td import restricted_argmax
from src.rl.envs import Environment
from src.rl.persistence import build_result_filename, save_policy


# Joue `episodes_count` épisodes complets avec une politique DÉJÀ entraînée
# (aucun apprentissage ici), en suivant à chaque pas l'action gloutonne de
# cette politique, pour mesurer sa performance réelle une fois convergée.
# Renvoie le score moyen et la longueur moyenne des épisodes (nombre de pas
# avant l'état terminal), les deux métriques de base pour comparer des
# politiques entre elles.
def evaluate_policy(
    env: Environment,
    pi: np.ndarray,
    episodes_count: int = 200,
    max_steps_per_episode: int = 1_000,
) -> Tuple[float, float]:
    scores = []
    lengths = []
    for _ in range(episodes_count):
        env.reset()
        steps = 0
        while not env.is_game_over() and steps < max_steps_per_episode:
            s = env.current_state()
            a = restricted_argmax(pi[s], env.available_actions())
            env.step(a)
            steps += 1
        scores.append(env.score())
        lengths.append(steps)
    return float(np.mean(scores)), float(np.mean(lengths))


# Entraîne un algorithme sur un environnement avec des hyperparamètres
# donnés, chronomètre l'entraînement, puis évalue la politique obtenue sur
# des épisodes fraîchement joués. C'est la brique de base pour comparer "quel
# algorithme est le plus judicieux sur quel environnement" : chaque appel
# renvoie des métriques directement comparables d'un algorithme ou d'un
# environnement à l'autre, quels que soient leurs hyperparamètres respectifs
# (le harness ne suppose rien de plus qu'un algorithme qui prend un
# environnement et renvoie (pi, V_ou_Q)).
def run_experiment(
    env_factory: Callable[[], Environment],
    algo: Callable[..., Tuple[np.ndarray, np.ndarray]],
    algo_kwargs: Optional[Dict[str, Any]] = None,
    eval_episodes_count: int = 200,
) -> Dict[str, Any]:
    algo_kwargs = algo_kwargs or {}

    train_env = env_factory()
    start = time.time()
    pi, value_function = algo(train_env, **algo_kwargs)
    training_time_seconds = time.time() - start

    eval_env = env_factory()
    mean_score, mean_episode_length = evaluate_policy(eval_env, pi, eval_episodes_count)

    return {
        "algo_name": algo.__name__,
        "hyperparameters": algo_kwargs,
        "training_time_seconds": training_time_seconds,
        "mean_score": mean_score,
        "mean_episode_length": mean_episode_length,
        "pi": pi,
        "value_function": value_function,
    }


# Comme run_experiment, mais sauvegarde en plus la politique (et V ou Q)
# obtenue sur disque, avec un nom de fichier standard, et ne renvoie que les
# métriques (pas les tableaux numpy) pour rester facilement exploitable dans
# un tableau récapitulatif pour le rapport final.
def run_and_save_experiment(
    env_name: str,
    env_factory: Callable[[], Environment],
    algo: Callable[..., Tuple[np.ndarray, np.ndarray]],
    results_dir: str,
    algo_kwargs: Optional[Dict[str, Any]] = None,
    eval_episodes_count: int = 200,
) -> Dict[str, Any]:
    result = run_experiment(env_factory, algo, algo_kwargs, eval_episodes_count)

    filename = build_result_filename(env_name, result["algo_name"], **result["hyperparameters"])
    filepath = os.path.join(results_dir, filename)
    value_function = result["value_function"]
    if value_function.ndim == 1:
        save_policy(filepath, pi=result["pi"], V=value_function)
    else:
        save_policy(filepath, pi=result["pi"], Q=value_function)

    return {
        "env_name": env_name,
        "algo_name": result["algo_name"],
        "hyperparameters": result["hyperparameters"],
        "training_time_seconds": result["training_time_seconds"],
        "mean_score": result["mean_score"],
        "mean_episode_length": result["mean_episode_length"],
        "saved_to": filepath,
    }


# Mesure la vitesse de convergence d'un algorithme sur un environnement, sans
# toucher aux algorithmes eux-mêmes : on relance l'entraînement depuis zéro
# pour différents budgets d'itérations et on regarde comment la performance
# de la politique obtenue évolue avec ce budget. `iterations_kwarg` permet de
# s'adapter au nom d'hyperparamètre utilisé par chaque algorithme pour son
# budget d'entraînement (`iterations_count` pour Monte Carlo/TD/Planning,
# `max_iterations` pour Value Iteration, etc.).
def measure_convergence(
    env_factory: Callable[[], Environment],
    algo: Callable[..., Tuple[np.ndarray, np.ndarray]],
    iteration_budgets: List[int],
    algo_kwargs: Optional[Dict[str, Any]] = None,
    eval_episodes_count: int = 200,
    iterations_kwarg: str = "iterations_count",
) -> List[Dict[str, Any]]:
    algo_kwargs = algo_kwargs or {}
    curve = []
    for budget in iteration_budgets:
        result = run_experiment(
            env_factory,
            algo,
            {**algo_kwargs, iterations_kwarg: budget},
            eval_episodes_count,
        )
        curve.append(
            {
                "iterations": budget,
                "mean_score": result["mean_score"],
                "training_time_seconds": result["training_time_seconds"],
            }
        )
    return curve
