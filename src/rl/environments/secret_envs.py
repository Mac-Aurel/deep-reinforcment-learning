import os
import sys
from contextlib import contextmanager
from typing import List

from src.rl.envs import Environment

# Dossier contenant le wrapper et les bibliothèques compilées fournis par le
# cours (secret_envs/secret_envs_wrapper.py, secret_envs/libs/...).
SECRET_ENVS_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "secret_envs"))


# Le wrapper fourni charge la bibliothèque compilée via un chemin RELATIF
# ("./libs/..."), donc il faut se placer temporairement dans le dossier
# secret_envs/ pour que le chargement réussisse, quel que soit l'endroit
# d'où ce module est importé. On restaure le répertoire de travail et
# sys.path juste après, pour ne pas perturber le reste du programme.
@contextmanager
def _secret_envs_cwd():
    previous_cwd = os.getcwd()
    previous_path = list(sys.path)
    os.chdir(SECRET_ENVS_DIR)
    sys.path.insert(0, SECRET_ENVS_DIR)
    try:
        yield
    finally:
        os.chdir(previous_cwd)
        sys.path[:] = previous_path


# Adapte un des 4 environnements secrets fournis (SecretEnv0 à SecretEnv3) à
# l'interface commune `Environment` du projet, pour pouvoir les utiliser avec
# les mêmes algorithmes model-free que les environnements développés pour le
# projet (Sarsa, Q-Learning, Dyna-Q, Monte Carlo on-policy/off-policy).
#
# Deux limites assumées : ces environnements n'exposent aucun moyen de se
# replacer directement dans un état donné, donc Monte Carlo ES n'est pas
# utilisable ici. Leur espace d'états est aussi bien plus grand que celui des
# environnements du projet (de 8192 à plus de 2 millions d'états), ce qui
# rend Policy/Value Iteration impraticables avec l'implémentation du projet :
# elle énumère explicitement toutes les combinaisons (état, action, état
# suivant, récompense) à chaque passage, ce qui suppose un espace d'états de
# taille raisonnable. Seule l'interface model-free est donc exposée ici.
class SecretEnvAdapter(Environment):
    def __init__(self, index: int) -> None:
        if index not in (0, 1, 2, 3):
            raise ValueError("index must be 0, 1, 2 or 3")
        self.index = index
        with _secret_envs_cwd():
            from secret_envs_wrapper import SecretEnv0, SecretEnv1, SecretEnv2, SecretEnv3

            self._env = [SecretEnv0, SecretEnv1, SecretEnv2, SecretEnv3][index]()

    # Redémarre un nouvel épisode (tire une nouvelle configuration cachée côté bibliothèque fournie).
    def reset(self) -> None:
        self._env.reset()

    # Joue l'action donnée ; délègue entièrement les règles à la bibliothèque fournie.
    def step(self, action: int) -> None:
        self._env.step(action)

    def is_game_over(self) -> bool:
        return bool(self._env.is_game_over())

    # La bibliothèque fournie nomme cette méthode state_id() plutôt que current_state().
    def current_state(self) -> int:
        return int(self._env.state_id())

    def available_actions(self) -> List[int]:
        return [int(a) for a in self._env.available_actions()]

    def score(self) -> float:
        return float(self._env.score())

    def num_states(self) -> int:
        return int(self._env.num_states())

    def num_actions(self) -> int:
        return int(self._env.num_actions())

    # Pretty print déjà fourni par la bibliothèque, réutilisé tel quel.
    def display(self) -> None:
        self._env.display()
