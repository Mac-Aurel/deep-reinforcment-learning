import csv
import json
import os
import subprocess
import sys
import time

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results", "secret_envs")
WORKER_SCRIPT = os.path.join(os.path.dirname(__file__), "_run_single_secret_env_experiment.py")
REPO_ROOT = os.path.join(os.path.dirname(__file__), "..")

# Budget d'itérations par environnement, adapté à la taille de son espace
# d'états (de 8192 à plus de 2 millions d'états) pour garder un temps
# d'exécution raisonnable. Dyna-Q reçoit un budget réduit de moitié car
# chaque itération y coûte plus cher (planification simulée en plus).
ITERATIONS_BUDGET = {0: 10_000, 1: 8_000, 2: 3_000, 3: 8_000}
ALGORITHMS = ["on_policy_first_visit_mc", "off_policy_mc", "sarsa", "q_learning", "dyna_q"]


MAX_ATTEMPTS = 3


# La bibliothèque compilée fournie par le cours plante parfois tout le
# processus sur un état interne (message "Forbidden action"), de façon
# intermittente et non reproductible à volonté (observé : un même run peut
# réussir ou planter selon l'exécution, sans lien évident avec nos propres
# graines aléatoires). On retente donc quelques fois avec une graine
# différente avant d'abandonner cette combinaison.
def run_one(env_index: int, algo: str) -> dict:
    budget = ITERATIONS_BUDGET[env_index]
    if algo == "dyna_q":
        budget //= 2

    env = dict(os.environ, PYTHONPATH=REPO_ROOT)
    last_error = None

    for attempt in range(MAX_ATTEMPTS):
        command = [
            sys.executable,
            WORKER_SCRIPT,
            "--env-index", str(env_index),
            "--algo", algo,
            "--iterations-count", str(budget),
            "--results-dir", RESULTS_DIR,
            "--seed", str(attempt),
        ]
        result = subprocess.run(command, capture_output=True, text=True, env=env, cwd=REPO_ROOT)

        if result.returncode == 0:
            break
        last_error = (result.stdout + result.stderr).strip()[-500:]
    else:
        return {
            "environment": f"secret_env_{env_index}",
            "algorithm": algo,
            "status": "failed",
            "error": last_error,
        }

    summary = json.loads(result.stdout.strip().splitlines()[-1])
    return {
        "environment": f"secret_env_{env_index}",
        "algorithm": algo,
        "status": "ok",
        "mean_score": summary["mean_score"],
        "mean_episode_length": summary["mean_episode_length"],
        "training_time_seconds": summary["training_time_seconds"],
    }


def run_all(env_indices=range(4)):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    rows = []
    for env_index in env_indices:
        for algo in ALGORITHMS:
            t0 = time.time()
            row = run_one(env_index, algo)
            elapsed = time.time() - t0
            row["wall_time_seconds"] = elapsed
            rows.append(row)
            print(f"secret_env_{env_index}  {algo:26s} status={row['status']} "
                  f"score={row.get('mean_score')} time={elapsed:.1f}s", flush=True)
            if row["status"] == "failed":
                print("  error:", row["error"], flush=True)
    return rows


def write_csv(rows, filename):
    filepath = os.path.join(RESULTS_DIR, filename)
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with open(filepath, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print("Wrote", filepath)


if __name__ == "__main__":
    env_indices = [int(a) for a in sys.argv[1:]] if len(sys.argv) > 1 else list(range(4))
    t0 = time.time()
    rows = run_all(env_indices)
    write_csv(rows, f"secret_envs_comparison_{'_'.join(map(str, env_indices))}.csv")
    print(f"Done in {time.time() - t0:.1f}s")
