import csv
import os
from typing import Any, Dict, List

import numpy as np

from src.rl.algorithms.dynamic_programming import value_iteration
from src.rl.algorithms.monte_carlo import on_policy_first_visit_monte_carlo_control
from src.rl.algorithms.planning import dyna_q
from src.rl.algorithms.td import q_learning, sarsa
from src.rl.environments.grid_world import GridWorldEnv
from src.rl.environments.line_world import LineWorldEnv
from src.rl.experiment import run_experiment

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
SEEDS = 5


# Étude n°1 : impact d'epsilon sur la fiabilité de Sarsa et de l'on-policy
# MC control sur Grid World. Trouvé pendant la comparaison de base : avec
# gamma proche de 1, une action "boucle sur elle-même" (ex: taper un mur)
# n'est presque jamais pénalisée par la mise à jour TD, donc avec un epsilon
# trop faible l'agent peut s'y enfermer durablement. On mesure ici
# directement à quel point augmenter epsilon compense ce problème.
def study_epsilon_on_grid_world() -> List[Dict[str, Any]]:
    rows = []
    for algo_name, algo in [("sarsa", sarsa), ("on_policy_first_visit_mc", on_policy_first_visit_monte_carlo_control)]:
        for epsilon in [0.1, 0.3, 0.5, 0.8]:
            scores = []
            successes = 0
            for seed in range(SEEDS):
                np.random.seed(seed)
                result = run_experiment(
                    env_factory=GridWorldEnv,
                    algo=algo,
                    algo_kwargs={"iterations_count": 10_000, "epsilon": epsilon},
                )
                scores.append(result["mean_score"])
                if result["mean_score"] >= 0.9:
                    successes += 1
            row = {
                "study": "epsilon_on_grid_world",
                "algorithm": algo_name,
                "epsilon": epsilon,
                "mean_score": float(np.mean(scores)),
                "success_rate": successes / SEEDS,
            }
            rows.append(row)
            print(row, flush=True)
    return rows


# Étude n°2 : impact du nombre de pas de planification simulés (Dyna-Q) sur
# la vitesse d'apprentissage à nombre d'épisodes réels égal. planning_steps=0
# revient exactement à du Q-Learning (aucune mise à jour simulée en plus).
def study_planning_steps_on_grid_world() -> List[Dict[str, Any]]:
    rows = []
    for planning_steps in [0, 5, 10, 20, 50]:
        scores = []
        successes = 0
        for seed in range(SEEDS):
            np.random.seed(seed)
            result = run_experiment(
                env_factory=GridWorldEnv,
                algo=dyna_q,
                algo_kwargs={"iterations_count": 500, "epsilon": 0.3, "planning_steps": planning_steps},
            )
            scores.append(result["mean_score"])
            if result["mean_score"] >= 0.9:
                successes += 1
        row = {
            "study": "planning_steps_on_grid_world",
            "planning_steps": planning_steps,
            "mean_score": float(np.mean(scores)),
            "success_rate": successes / SEEDS,
        }
        rows.append(row)
        print(row, flush=True)
    return rows


# Étude n°3 : impact de gamma sur la valeur optimale trouvée par Value
# Iteration. Plus gamma est petit, plus les récompenses lointaines sont
# dévaluées : sur un chemin de quelques pas jusqu'à la récompense, l'effet
# reste visible même pour de petits environnements.
def study_gamma_on_value_iteration() -> List[Dict[str, Any]]:
    rows = []
    for env_name, env_factory in [("line_world", LineWorldEnv), ("grid_world", GridWorldEnv)]:
        for gamma in [0.5, 0.9, 0.99, 0.999999]:
            env = env_factory()
            pi, V = value_iteration(env, gamma=gamma)
            env.reset()
            row = {
                "study": "gamma_on_value_iteration",
                "environment": env_name,
                "gamma": gamma,
                "start_state_value": float(V[env.current_state()]),
            }
            rows.append(row)
            print(row, flush=True)
    return rows


def write_csv(rows: List[Dict[str, Any]], filename: str) -> None:
    if not rows:
        return
    filepath = os.path.join(RESULTS_DIR, filename)
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with open(filepath, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {filepath}")


if __name__ == "__main__":
    os.makedirs(RESULTS_DIR, exist_ok=True)
    all_rows = []
    all_rows += study_epsilon_on_grid_world()
    all_rows += study_planning_steps_on_grid_world()
    all_rows += study_gamma_on_value_iteration()
    write_csv(all_rows, "hyperparameter_study.csv")
