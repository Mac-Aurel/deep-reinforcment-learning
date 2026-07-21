import argparse
import json
import sys

import numpy as np

from src.rl.algorithms.monte_carlo import off_policy_monte_carlo_control, on_policy_first_visit_monte_carlo_control
from src.rl.algorithms.planning import dyna_q
from src.rl.algorithms.td import q_learning, sarsa
from src.rl.environments.secret_envs import SecretEnvAdapter
from src.rl.experiment import run_and_save_experiment

# Lance une seule combinaison (environnement secret, algorithme) dans un
# processus séparé. La bibliothèque compilée fournie par le cours peut
# planter tout le processus sur certains états internes rares (message
# "Forbidden action" observé pendant les tests) ; Python ne peut pas
# rattraper ce genre de plantage natif avec un simple try/except, donc
# isoler chaque run dans son propre processus permet au script principal de
# continuer avec les combinaisons suivantes même si l'une d'elles plante.

ALGOS = {
    "on_policy_first_visit_mc": on_policy_first_visit_monte_carlo_control,
    "off_policy_mc": off_policy_monte_carlo_control,
    "sarsa": sarsa,
    "q_learning": q_learning,
    "dyna_q": dyna_q,
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-index", type=int, required=True)
    parser.add_argument("--algo", choices=list(ALGOS.keys()), required=True)
    parser.add_argument("--iterations-count", type=int, required=True)
    parser.add_argument("--results-dir", required=True)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    algo = ALGOS[args.algo]
    kwargs = {"iterations_count": args.iterations_count}
    if args.algo != "off_policy_mc":
        kwargs["epsilon"] = 0.3
    if args.algo == "dyna_q":
        kwargs["planning_steps"] = 10

    np.random.seed(args.seed)
    summary = run_and_save_experiment(
        env_name=f"secret_env_{args.env_index}",
        env_factory=lambda: SecretEnvAdapter(args.env_index),
        algo=algo,
        results_dir=args.results_dir,
        algo_kwargs=kwargs,
    )
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
