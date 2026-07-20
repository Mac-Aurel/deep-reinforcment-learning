import csv
import os
import time
from typing import Any, Dict, List

import numpy as np

from src.rl.algorithms.dynamic_programming import policy_iteration, value_iteration
from src.rl.algorithms.monte_carlo import (
    monte_carlo_es,
    off_policy_monte_carlo_control,
    on_policy_first_visit_monte_carlo_control,
)
from src.rl.algorithms.planning import dyna_q
from src.rl.algorithms.td import q_learning, sarsa
from src.rl.environments.grid_world import GridWorldEnv
from src.rl.environments.line_world import LineWorldEnv
from src.rl.environments.monty_hall import MontyHallEnv
from src.rl.environments.two_round_rps import TwoRoundRPSEnv
from src.rl.experiment import run_and_save_experiment

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")

ENVIRONMENTS = {
    "line_world": LineWorldEnv,
    "grid_world": GridWorldEnv,
    "two_round_rps": TwoRoundRPSEnv,
    "monty_hall_3_doors": lambda: MontyHallEnv(num_doors=3),
    "monty_hall_5_doors": lambda: MontyHallEnv(num_doors=5),
}

# epsilon=0.3 (plutôt que le 0.1 par défaut) pour les 4 méthodes dont
# l'exploration dépend directement de Q : avec gamma proche de 1, une action
# "boucle sur elle-même" (ex: taper un mur dans Grid World) n'est presque
# jamais pénalisée par la mise à jour TD, donc si elle est choisie par
# malchance au début, epsilon=0.1 ne laisse pas assez de chances de s'en
# échapper avant la fin du budget d'épisodes (voir le rapport pour le détail
# de cette expérience). Monte Carlo ES (départs aléatoires) et Off-policy MC
# (comportement uniforme, indépendant de Q) n'ont pas ce problème et gardent
# leurs réglages par défaut.
ALGORITHMS = {
    "policy_iteration": (policy_iteration, {}),
    "value_iteration": (value_iteration, {}),
    "monte_carlo_es": (monte_carlo_es, {"iterations_count": 10_000}),
    "on_policy_first_visit_mc": (on_policy_first_visit_monte_carlo_control, {"iterations_count": 10_000, "epsilon": 0.3}),
    "off_policy_mc": (off_policy_monte_carlo_control, {"iterations_count": 10_000}),
    "sarsa": (sarsa, {"iterations_count": 10_000, "epsilon": 0.3}),
    "q_learning": (q_learning, {"iterations_count": 10_000, "epsilon": 0.3}),
    "dyna_q": (dyna_q, {"iterations_count": 5_000, "epsilon": 0.3}),
}

# Nombre d'entraînements indépendants par (environnement, algorithme) : les
# méthodes de programmation dynamique sont déterministes une fois convergées
# (peu de variance d'un tirage aléatoire à l'autre), les méthodes model-free
# beaucoup moins (dépendent de l'exploration), d'où plus de répétitions pour
# elles afin de mesurer leur fiabilité (taux de réussite), pas juste une
# performance moyenne qui masquerait des échecs isolés.
SEEDS_PER_ALGO = {
    "policy_iteration": 1,
    "value_iteration": 1,
}
DEFAULT_SEEDS = 5

# Marge tolérée sous le score optimal réel de l'environnement pour compter un
# run comme "réussi". Un seuil global (ex: 0.8) n'aurait pas de sens ici :
# le score optimal lui-même varie selon l'environnement (1.0 pour Line/Grid
# World, mais seulement 2/3 pour Monty Hall à 3 portes, 4/5 à 5 portes).
SUCCESS_MARGIN = 0.1


# Calcule le score optimal réel de chaque environnement via Value Iteration
# (déjà validé exact sur les 5 environnements), pour servir de référence à
# la notion de "réussite" plutôt qu'un seuil arbitraire identique partout.
def compute_optimal_scores() -> Dict[str, float]:
    optimal_scores = {}
    for env_name, env_factory in ENVIRONMENTS.items():
        env = env_factory()
        _, V = value_iteration(env)
        env.reset()
        start_state = env.current_state()
        optimal_scores[env_name] = float(V[start_state])
    return optimal_scores


def run_baseline_matrix() -> List[Dict[str, Any]]:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    optimal_scores = compute_optimal_scores()
    rows = []

    for env_name, env_factory in ENVIRONMENTS.items():
        optimal_score = optimal_scores[env_name]
        for algo_name, (algo, kwargs) in ALGORITHMS.items():
            seeds_count = SEEDS_PER_ALGO.get(algo_name, DEFAULT_SEEDS)
            scores = []
            successes = 0
            total_time = 0.0

            for seed in range(seeds_count):
                np.random.seed(seed)
                summary = run_and_save_experiment(
                    env_name=f"{env_name}_seed{seed}",
                    env_factory=env_factory,
                    algo=algo,
                    results_dir=RESULTS_DIR,
                    algo_kwargs=kwargs,
                )
                scores.append(summary["mean_score"])
                total_time += summary["training_time_seconds"]
                if summary["mean_score"] >= optimal_score - SUCCESS_MARGIN:
                    successes += 1

            rows.append(
                {
                    "environment": env_name,
                    "algorithm": algo_name,
                    "optimal_score": optimal_score,
                    "seeds_count": seeds_count,
                    "mean_score": float(np.mean(scores)),
                    "std_score": float(np.std(scores)),
                    "success_rate": successes / seeds_count,
                    "avg_training_time_seconds": total_time / seeds_count,
                }
            )
            print(
                f"{env_name:20s} {algo_name:26s} "
                f"mean_score={rows[-1]['mean_score']:.3f} (optimal={optimal_score:.3f}) "
                f"success_rate={rows[-1]['success_rate']:.2f} "
                f"time={rows[-1]['avg_training_time_seconds']:.2f}s",
                flush=True,
            )

    return rows


def write_csv(rows: List[Dict[str, Any]], filename: str) -> None:
    filepath = os.path.join(RESULTS_DIR, filename)
    with open(filepath, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {filepath}")


if __name__ == "__main__":
    t0 = time.time()
    baseline_rows = run_baseline_matrix()
    write_csv(baseline_rows, "baseline_comparison.csv")
    print(f"Done in {time.time() - t0:.1f}s")
