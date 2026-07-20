from typing import Dict, Optional

import numpy as np


# Sauvegarde une politique (et éventuellement la fonction de valeur V et/ou
# la fonction action-valeur Q) dans un unique fichier .npz, pour pouvoir la
# recharger plus tard sans avoir à relancer l'apprentissage. C'est ce qui
# permet de fournir des résultats "prêts à être exécutés" pour le rapport et
# la soutenance, plutôt que de devoir tout ré-entraîner à chaque fois.
def save_policy(filepath: str, pi: np.ndarray, V: Optional[np.ndarray] = None, Q: Optional[np.ndarray] = None) -> None:
    arrays = {"pi": pi}
    if V is not None:
        arrays["V"] = V
    if Q is not None:
        arrays["Q"] = Q
    np.savez(filepath, **arrays)


# Recharge une politique sauvegardée par save_policy. Renvoie un dictionnaire
# contenant toujours "pi", et "V"/"Q" seulement s'ils avaient été sauvegardés.
def load_policy(filepath: str) -> Dict[str, np.ndarray]:
    with np.load(filepath) as data:
        return {key: data[key] for key in data.files}


# Construit un nom de fichier standard à partir de l'environnement, de
# l'algorithme utilisé et de ses hyperparamètres, pour que chaque résultat
# sauvegardé soit identifiable sans avoir à l'ouvrir (les hyperparamètres
# sont triés par nom pour que le nom de fichier ne dépende pas de l'ordre
# dans lequel ils ont été passés).
def build_result_filename(env_name: str, algo_name: str, **hyperparameters) -> str:
    params = "_".join(f"{key}={value}" for key, value in sorted(hyperparameters.items()))
    suffix = f"__{params}" if params else ""
    return f"{env_name}__{algo_name}{suffix}.npz"
