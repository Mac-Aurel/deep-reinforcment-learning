from typing import Callable, Optional

import numpy as np

from src.rl.algorithms.td import restricted_argmax
from src.rl.envs import Environment


# Déroule un épisode complet avec une politique DÉJÀ entraînée, sans faire
# le moindre apprentissage : à chaque pas, affiche l'état courant, attend une
# confirmation (par défaut une touche Entrée, pratique pour ralentir le
# rythme pendant une soutenance), puis joue l'action gloutonne de la
# politique. Renvoie le score final de l'épisode. `pause=None` permet de
# dérouler sans attendre (utile pour les tests automatisés).
def replay_policy(
    env: Environment,
    pi: np.ndarray,
    pause: Optional[Callable[[], None]] = input,
    max_steps: int = 1_000,
) -> float:
    env.reset()
    env.display()

    steps = 0
    while not env.is_game_over() and steps < max_steps:
        if pause is not None:
            pause()
        s = env.current_state()
        a = restricted_argmax(pi[s], env.available_actions())
        env.step(a)
        env.display()
        steps += 1

    return env.score()
