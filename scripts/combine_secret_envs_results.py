import csv
import os
import re

from src.rl.environments.secret_envs import SecretEnvAdapter
from src.rl.experiment import evaluate_policy
from src.rl.persistence import load_policy

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results", "secret_envs")

# Reconstruit un tableau de comparaison cohérent à partir de toutes les
# politiques déjà sauvegardées (results/secret_envs/*.npz), en les rejouant
# sur des épisodes frais, plutôt que de recombiner à la main des journaux de
# lancements séparés (certains ayant dû être relancés après un plantage de la
# bibliothèque fournie sur un état interne rare).

FILENAME_PATTERN = re.compile(r"^secret_env_(\d+)__(.+?)(?:__.*)?\.npz$")


def main():
    rows = []
    for filename in sorted(os.listdir(RESULTS_DIR)):
        if not filename.endswith(".npz"):
            continue
        match = FILENAME_PATTERN.match(filename)
        if not match:
            continue
        env_index = int(match.group(1))
        algo_name = match.group(2)

        loaded = load_policy(os.path.join(RESULTS_DIR, filename))
        env = SecretEnvAdapter(env_index)
        mean_score, mean_length = evaluate_policy(env, loaded["pi"], episodes_count=200)

        rows.append(
            {
                "environment": f"secret_env_{env_index}",
                "algorithm": algo_name,
                "mean_score": mean_score,
                "mean_episode_length": mean_length,
                "source_file": filename,
            }
        )
        print(rows[-1], flush=True)

    filepath = os.path.join(RESULTS_DIR, "secret_envs_comparison.csv")
    with open(filepath, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print("Wrote", filepath)


if __name__ == "__main__":
    main()
