from abc import ABC, abstractmethod
from typing import List

import numpy as np


class Environment(ABC):
    """
    Interface "model-free" commune à tous les environnements du projet.
    C'est celle qu'utilisent Monte Carlo, TD Learning et Planning : ces
    algorithmes n'ont pas besoin de connaître les probabilités de
    transition, ils apprennent en jouant des épisodes (reset/step) et en
    observant le score obtenu.
    """

    # Remet l'environnement dans son état de départ, pour commencer un nouvel épisode.
    @abstractmethod
    def reset(self) -> None:
        raise NotImplementedError

    # Joue l'action donnée et fait avancer l'environnement d'un pas.
    @abstractmethod
    def step(self, action: int) -> None:
        raise NotImplementedError

    # Indique si l'épisode en cours est terminé (état terminal atteint).
    @abstractmethod
    def is_game_over(self) -> bool:
        raise NotImplementedError

    # Renvoie l'identifiant de l'état courant de l'agent.
    @abstractmethod
    def current_state(self) -> int:
        raise NotImplementedError

    # Liste les actions jouables depuis l'état courant.
    @abstractmethod
    def available_actions(self) -> List[int]:
        raise NotImplementedError

    # Renvoie le score cumulé de l'épisode en cours.
    @abstractmethod
    def score(self) -> float:
        raise NotImplementedError

    # Nombre maximal d'états possibles, utile pour dimensionner V/Q sans lancer l'environnement.
    @abstractmethod
    def num_states(self) -> int:
        raise NotImplementedError

    # Nombre maximal d'actions possibles, utile pour dimensionner V/Q sans lancer l'environnement.
    @abstractmethod
    def num_actions(self) -> int:
        raise NotImplementedError

    # Affiche l'état courant de manière lisible (pretty print ou interface graphique).
    @abstractmethod
    def display(self) -> None:
        raise NotImplementedError


class ExploringStartsEnvironment(Environment):
    """
    Interface complémentaire pour les environnements qui peuvent être placés
    directement dans un état donné, plutôt que de toujours reprendre depuis
    l'état de départ fixe de `reset()`. Nécessaire pour Monte Carlo ES, qui
    doit pouvoir démarrer chaque épisode depuis une paire (état, action)
    tirée au hasard afin de garantir que tous les états sont explorés.
    """

    # Place directement l'environnement dans l'état demandé (doit être un état non terminal).
    @abstractmethod
    def set_state(self, state: int) -> None:
        raise NotImplementedError


class MDPEnvironment(ABC):
    """
    Interface complémentaire pour les environnements dont on connaît le
    modèle complet (dynamique de programmation). Un environnement qui
    l'implémente expose ses états, actions, récompenses possibles et la
    probabilité de chaque transition, ce qui permet à Policy Iteration et
    Value Iteration de calculer une politique sans jouer d'épisode.
    """

    # Liste de tous les états possibles de l'environnement.
    @abstractmethod
    def states(self) -> np.ndarray:
        raise NotImplementedError

    # Liste de toutes les actions possibles de l'environnement.
    @abstractmethod
    def actions(self) -> np.ndarray:
        raise NotImplementedError

    # Liste de toutes les valeurs de récompense possibles (pas leur probabilité, juste les valeurs).
    @abstractmethod
    def rewards(self) -> np.ndarray:
        raise NotImplementedError

    # Liste des états terminaux (où l'épisode s'arrête).
    @abstractmethod
    def terminal_states(self) -> np.ndarray:
        raise NotImplementedError

    # Probabilité d'atterrir dans l'état s_p et de recevoir la récompense rewards()[r_index],
    # sachant qu'on était dans l'état s et qu'on a joué l'action a.
    @abstractmethod
    def transition_probability(self, s: int, a: int, s_p: int, r_index: int) -> float:
        raise NotImplementedError
