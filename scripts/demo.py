import argparse
import os
from typing import List

from src.rl.environments.grid_world import GridWorldEnv
from src.rl.environments.line_world import LineWorldEnv
from src.rl.environments.monty_hall import MontyHallEnv
from src.rl.environments.secret_envs import SecretEnvAdapter
from src.rl.environments.two_round_rps import TwoRoundRPSEnv
from src.rl.envs import Environment
from src.rl.manual_play import play_manually
from src.rl.persistence import load_policy
from src.rl.replay import replay_policy

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
SECRET_RESULTS_DIR = os.path.join(RESULTS_DIR, "secret_envs")

# Associe chaque nom d'environnement (le même préfixe que dans les noms de
# fichiers de results/) à une fonction qui construit une instance fraîche de
# cet environnement, pour pouvoir le choisir depuis la ligne de commande.
ENVIRONMENTS = {
    "line_world": LineWorldEnv,
    "grid_world": GridWorldEnv,
    "two_round_rps": TwoRoundRPSEnv,
    "monty_hall_3_doors": lambda: MontyHallEnv(num_doors=3),
    "monty_hall_5_doors": lambda: MontyHallEnv(num_doors=5),
    "secret_env_0": lambda: SecretEnvAdapter(index=0),
    "secret_env_1": lambda: SecretEnvAdapter(index=1),
    "secret_env_2": lambda: SecretEnvAdapter(index=2),
    "secret_env_3": lambda: SecretEnvAdapter(index=3),
}


# Liste les fichiers de politiques déjà sauvegardées pour un environnement
# donné, pour savoir quoi passer à --policy sans avoir à fouiller le dossier
# results/ à la main.
def list_saved_policies(env_name: str) -> List[str]:
    matches = []
    for directory in (RESULTS_DIR, SECRET_RESULTS_DIR):
        if not os.path.isdir(directory):
            continue
        for filename in sorted(os.listdir(directory)):
            if filename.startswith(env_name) and filename.endswith(".npz"):
                matches.append(os.path.join(directory, filename))
    return matches


# Retrouve le chemin complet d'un fichier de politique à partir d'un chemin
# donné tel quel ou d'un simple nom de fichier (cherché dans results/ puis
# results/secret_envs/), pour ne pas avoir à taper le chemin complet à chaque
# essai.
def resolve_policy_path(policy: str) -> str:
    for candidate in (policy, os.path.join(RESULTS_DIR, policy), os.path.join(SECRET_RESULTS_DIR, policy)):
        if os.path.isfile(candidate):
            return candidate
    raise FileNotFoundError(f"Fichier de politique introuvable : {policy}")


# Rejoue pas à pas, dans le terminal, une politique déjà entraînée et
# sauvegardée : aucun apprentissage n'a lieu ici, seulement l'exécution de
# l'action gloutonne de la politique à chaque état rencontré.
def run_replay(env: Environment, policy_path: str) -> None:
    data = load_policy(resolve_policy_path(policy_path))
    score = replay_policy(env, data["pi"])
    print(f"Score final : {score}")


# Laisse un humain jouer manuellement un épisode complet sur l'environnement
# choisi, pour vérifier que ses règles sont bien celles attendues.
def run_manual(env: Environment) -> None:
    play_manually(env)


# Construit les arguments de la ligne de commande : le mode (replay, manual
# ou list), l'environnement à utiliser, et le fichier de politique à charger
# (nécessaire uniquement en mode replay).
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Démo du projet : rejeu pas à pas d'une politique entraînée, "
                     "ou mode manuel pour vérifier les règles d'un environnement."
    )
    parser.add_argument(
        "mode",
        choices=["replay", "manual", "list"],
        help="replay : rejoue une politique sauvegardée. "
             "manual : jouez vous-même au clavier. "
             "list : liste les politiques sauvegardées pour --env.",
    )
    parser.add_argument("--env", choices=sorted(ENVIRONMENTS), required=True, help="Environnement à utiliser.")
    parser.add_argument(
        "--policy",
        help="Fichier .npz à rejouer (mode replay uniquement) : chemin complet, "
             "ou simple nom de fichier cherché dans results/.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.mode == "list":
        for path in list_saved_policies(args.env):
            print(path)
    elif args.mode == "manual":
        run_manual(ENVIRONMENTS[args.env]())
    elif args.mode == "replay":
        if not args.policy:
            raise SystemExit("Le mode replay nécessite --policy (utilisez le mode list pour voir les fichiers disponibles).")
        run_replay(ENVIRONMENTS[args.env](), args.policy)
