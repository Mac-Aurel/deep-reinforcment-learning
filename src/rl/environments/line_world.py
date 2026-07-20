from typing import List

import numpy as np

from src.rl.envs import ExploringStartsEnvironment, MDPEnvironment

# Valeurs de récompense possibles : rien (0), sortir par la gauche (-1), sortir par la droite (+1).
REWARDS = np.array([0.0, -1.0, 1.0])


class LineWorldEnv(ExploringStartsEnvironment, MDPEnvironment):
    """
    Reprend l'environnement Line World du notebook `notebooks/Line World
    Experiments.ipynb` : un agent se déplace sur une ligne de `num_cells`
    cases numérotées de 0 à num_cells - 1. Il part du milieu et choisit à
    chaque tour d'aller à gauche (action 0) ou à droite (action 1).
    Atteindre la case 0 termine la partie avec un score de -1, atteindre la
    dernière case la termine avec un score de +1.
    """

    def __init__(self, num_cells: int = 5) -> None:
        self.num_cells = num_cells
        self.agent_pos = num_cells // 2

    # -- Interface model-free (Environment), utilisée par Monte Carlo / TD / Planning --

    # Replace l'agent au milieu de la ligne pour démarrer un nouvel épisode.
    def reset(self) -> None:
        self.agent_pos = self.num_cells // 2

    # Déplace l'agent d'une case à gauche (action 0) ou à droite (action 1).
    def step(self, action: int) -> None:
        if action not in self.available_actions():
            raise ValueError("Action is invalid in current state")
        if self.is_game_over():
            raise RuntimeError("Trying to play while game is over")
        self.agent_pos += 1 if action == 1 else -1

    # La partie est terminée quand l'agent atteint une des deux extrémités de la ligne.
    def is_game_over(self) -> bool:
        return self.agent_pos == 0 or self.agent_pos == self.num_cells - 1

    # Dans Line World, la position de l'agent EST l'état.
    def current_state(self) -> int:
        return self.agent_pos

    # Les deux actions (gauche, droite) sont toujours proposées.
    def available_actions(self) -> List[int]:
        return [0, 1]

    # +1 si l'agent est arrivé à droite, -1 s'il est arrivé à gauche, 0 si la partie est en cours.
    def score(self) -> float:
        if self.agent_pos == self.num_cells - 1:
            return 1.0
        if self.agent_pos == 0:
            return -1.0
        return 0.0

    def num_states(self) -> int:
        return self.num_cells

    def num_actions(self) -> int:
        return 2

    # Place directement l'agent sur la case demandée, sans jouer d'action (exploring starts).
    def set_state(self, state: int) -> None:
        self.agent_pos = state

    # Pretty print en ligne de commande : "X" pour la case de l'agent, "_" pour les autres.
    def display(self) -> None:
        print("".join("X" if s == self.agent_pos else "_" for s in range(self.num_cells)))

    # -- Interface modèle complet (MDPEnvironment), utilisée par Policy/Value Iteration --

    def states(self) -> np.ndarray:
        return np.arange(self.num_cells)

    def actions(self) -> np.ndarray:
        return np.array([0, 1])

    def rewards(self) -> np.ndarray:
        return REWARDS

    def terminal_states(self) -> np.ndarray:
        return np.array([0, self.num_cells - 1])

    # Probabilité de finir dans l'état s_p avec la récompense rewards()[r_index],
    # sachant qu'on part de l'état s en jouant l'action a. Line World est déterministe :
    # cette probabilité vaut toujours 0 ou 1.
    def transition_probability(self, s: int, a: int, s_p: int, r_index: int) -> float:
        if s in self.terminal_states():
            return 0.0

        if a == 0 and s_p == s - 1:
            expected_r_index = 1 if s == 1 else 0
            return 1.0 if r_index == expected_r_index else 0.0

        if a == 1 and s_p == s + 1:
            expected_r_index = 2 if s == self.num_cells - 2 else 0
            return 1.0 if r_index == expected_r_index else 0.0

        return 0.0
