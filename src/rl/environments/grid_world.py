from typing import List, Tuple

import numpy as np

from src.rl.envs import ExploringStartsEnvironment, MDPEnvironment

# Les 4 déplacements possibles.
LEFT, RIGHT, UP, DOWN = 0, 1, 2, 3

# Valeurs de récompense possibles : rien (0), tomber dans le piège (-1), atteindre l'objectif (+1).
REWARDS = np.array([0.0, -1.0, 1.0])


class GridWorldEnv(ExploringStartsEnvironment, MDPEnvironment):
    """
    Grille de `height` x `width` cases. L'agent part du coin en haut à
    gauche et doit atteindre la case objectif (récompense +1) tout en
    évitant une case piège (récompense -1) ; les deux sont des états
    terminaux. Une action qui ferait sortir l'agent de la grille le laisse
    sur place (il "tape le mur"), sans récompense.
    """

    def __init__(
        self,
        height: int = 5,
        width: int = 5,
        goal: Tuple[int, int] = (4, 4),
        trap: Tuple[int, int] = (4, 3),
    ) -> None:
        self.height = height
        self.width = width
        self.goal = goal
        self.trap = trap
        self.row = 0
        self.col = 0

    # -- Interface model-free (Environment), utilisée par Monte Carlo / TD / Planning --

    # Replace l'agent dans le coin haut-gauche pour démarrer un nouvel épisode.
    def reset(self) -> None:
        self.row = 0
        self.col = 0

    # Déplace l'agent d'une case dans la direction demandée. Si l'action le ferait
    # sortir de la grille, il reste simplement sur place (mur).
    def step(self, action: int) -> None:
        if action not in self.available_actions():
            raise ValueError("Action is invalid in current state")
        if self.is_game_over():
            raise RuntimeError("Trying to play while game is over")
        self.row, self.col = self._next_position(self.row, self.col, action)

    # La partie est terminée quand l'agent est sur l'objectif ou sur le piège.
    def is_game_over(self) -> bool:
        return (self.row, self.col) in (self.goal, self.trap)

    # L'état est l'index de la case dans la grille, en la lisant ligne par ligne.
    def current_state(self) -> int:
        return self._state_id(self.row, self.col)

    # Les 4 déplacements sont toujours proposés (près d'un bord, ils ne font juste rien).
    def available_actions(self) -> List[int]:
        return [LEFT, RIGHT, UP, DOWN]

    # +1 sur l'objectif, -1 sur le piège, 0 sinon (partie en cours).
    def score(self) -> float:
        if (self.row, self.col) == self.goal:
            return 1.0
        if (self.row, self.col) == self.trap:
            return -1.0
        return 0.0

    def num_states(self) -> int:
        return self.height * self.width

    def num_actions(self) -> int:
        return 4

    # Place directement l'agent sur la case demandée, sans jouer d'action (exploring starts).
    def set_state(self, state: int) -> None:
        self.row, self.col = divmod(state, self.width)

    # Pretty print en ligne de commande : "X" = agent, "G" = objectif, "T" = piège, "." = case vide.
    def display(self) -> None:
        for row in range(self.height):
            line = ""
            for col in range(self.width):
                if (row, col) == (self.row, self.col):
                    line += "X"
                elif (row, col) == self.goal:
                    line += "G"
                elif (row, col) == self.trap:
                    line += "T"
                else:
                    line += "."
            print(line)

    # -- Interface modèle complet (MDPEnvironment), utilisée par Policy/Value Iteration --

    def states(self) -> np.ndarray:
        return np.arange(self.height * self.width)

    def actions(self) -> np.ndarray:
        return np.array([LEFT, RIGHT, UP, DOWN])

    def rewards(self) -> np.ndarray:
        return REWARDS

    def terminal_states(self) -> np.ndarray:
        return np.array([self._state_id(*self.goal), self._state_id(*self.trap)])

    # Probabilité de finir dans l'état s_p avec la récompense rewards()[r_index],
    # sachant qu'on part de l'état s en jouant l'action a. Grid World est
    # déterministe : cette probabilité vaut toujours 0 ou 1.
    def transition_probability(self, s: int, a: int, s_p: int, r_index: int) -> float:
        if self._position(s) in (self.goal, self.trap):
            return 0.0

        row, col = self._position(s)
        next_row, next_col = self._next_position(row, col, a)
        if self._state_id(next_row, next_col) != s_p:
            return 0.0

        if (next_row, next_col) == self.goal:
            expected_r_index = 2
        elif (next_row, next_col) == self.trap:
            expected_r_index = 1
        else:
            expected_r_index = 0

        return 1.0 if r_index == expected_r_index else 0.0

    # -- Utilitaires internes --

    # Convertit une position (ligne, colonne) en identifiant d'état unique.
    def _state_id(self, row: int, col: int) -> int:
        return row * self.width + col

    # Convertit un identifiant d'état en position (ligne, colonne).
    def _position(self, state: int) -> Tuple[int, int]:
        return divmod(state, self.width)

    # Calcule la position obtenue en appliquant une action, en restant dans les limites de la grille.
    def _next_position(self, row: int, col: int, action: int) -> Tuple[int, int]:
        if action == LEFT:
            return row, max(col - 1, 0)
        if action == RIGHT:
            return row, min(col + 1, self.width - 1)
        if action == UP:
            return max(row - 1, 0), col
        if action == DOWN:
            return min(row + 1, self.height - 1), col
        raise ValueError("Unknown action")
