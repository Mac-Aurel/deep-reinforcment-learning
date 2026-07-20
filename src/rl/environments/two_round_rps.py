from typing import List

import numpy as np

from src.rl.envs import ExploringStartsEnvironment, MDPEnvironment

# Les 3 coups possibles.
ROCK, PAPER, SCISSORS = 0, 1, 2
MOVE_NAMES = ["Pierre", "Papier", "Ciseaux"]

# Valeurs de récompense possibles pour un round : défaite (-1), nul (0), victoire (+1).
REWARDS = np.array([-1.0, 0.0, 1.0])


class TwoRoundRPSEnv(ExploringStartsEnvironment, MDPEnvironment):
    """
    Partie de Pierre-Feuille-Ciseaux en 2 rounds contre un adversaire
    particulier : au round 1 il joue au hasard, mais au round 2 il est
    FORCÉ de rejouer le coup que l'AGENT a joué au round 1. Chaque round
    rapporte +1 (victoire), -1 (défaite) ou 0 (nul), et le score de la
    partie est la somme des deux rounds.

    Comme l'adversaire du round 2 est entièrement déterminé par le coup de
    l'agent au round 1, l'état du round 2 encode directement ce coup (3
    états possibles), ce qui permet à l'agent d'apprendre à toujours jouer
    le coup qui bat sa propre décision du round 1.
    """

    def __init__(self) -> None:
        self.round = 1
        self.agent_round1_action = None
        self.cumulative_score = 0.0

    # -- Interface model-free (Environment), utilisée par Monte Carlo / TD / Planning --

    # Remet la partie à zéro : retour au round 1, aucun score accumulé.
    def reset(self) -> None:
        self.round = 1
        self.agent_round1_action = None
        self.cumulative_score = 0.0

    # Joue un coup. Au round 1 l'adversaire est tiré au hasard ; au round 2
    # il est forcé de rejouer le coup de l'agent au round 1.
    def step(self, action: int) -> None:
        if action not in self.available_actions():
            raise ValueError("Action is invalid in current state")
        if self.is_game_over():
            raise RuntimeError("Trying to play while game is over")

        if self.round == 1:
            opponent_action = int(np.random.randint(3))
            self.cumulative_score += self._compare(action, opponent_action)
            self.agent_round1_action = action
            self.round = 2
        else:
            opponent_action = self.agent_round1_action
            self.cumulative_score += self._compare(action, opponent_action)
            self.round = 3

    # La partie est terminée une fois les 2 rounds joués.
    def is_game_over(self) -> bool:
        return self.round == 3

    # Round 1 -> état 0. Round 2 -> état 1 + coup joué au round 1 (3 états
    # possibles). Partie terminée -> état 4 (unique état terminal).
    def current_state(self) -> int:
        if self.round == 1:
            return 0
        if self.round == 2:
            return 1 + self.agent_round1_action
        return 4

    # Les 3 coups sont toujours jouables, aux deux rounds.
    def available_actions(self) -> List[int]:
        return [ROCK, PAPER, SCISSORS]

    # Score cumulé sur les 2 rounds déjà joués.
    def score(self) -> float:
        return self.cumulative_score

    def num_states(self) -> int:
        return 5

    def num_actions(self) -> int:
        return 3

    # Place directement la partie dans l'état demandé (doit être un état non terminal).
    def set_state(self, state: int) -> None:
        self.cumulative_score = 0.0
        if state == 0:
            self.round = 1
            self.agent_round1_action = None
        else:
            self.round = 2
            self.agent_round1_action = state - 1

    # Pretty print en ligne de commande de la situation actuelle.
    def display(self) -> None:
        if self.round == 1:
            print("Round 1 : à vous de jouer")
        elif self.round == 2:
            print(f"Round 1 joué. Round 2 : l'adversaire va rejouer {MOVE_NAMES[self.agent_round1_action]}")
        else:
            print(f"Partie terminée, score : {self.cumulative_score}")

    # -- Interface modèle complet (MDPEnvironment), utilisée par Policy/Value Iteration --

    def states(self) -> np.ndarray:
        return np.arange(5)

    def actions(self) -> np.ndarray:
        return np.array([ROCK, PAPER, SCISSORS])

    def rewards(self) -> np.ndarray:
        return REWARDS

    def terminal_states(self) -> np.ndarray:
        return np.array([4])

    # Round 1 (état 0) : l'adversaire est aléatoire, donc les 3 issues
    # (victoire/nul/défaite) ont chacune 1/3 de chance, et l'état suivant
    # est toujours "round 2 sachant que l'agent a joué a" (déterministe).
    # Round 2 (états 1 à 3) : l'adversaire est connu (forcé), donc l'issue
    # est déterministe, et l'état suivant est toujours l'état terminal (4).
    def transition_probability(self, s: int, a: int, s_p: int, r_index: int) -> float:
        if s == 4:
            return 0.0

        if s == 0:
            if s_p != 1 + a:
                return 0.0
            return 1.0 / 3.0

        if s_p != 4:
            return 0.0
        opponent_action = s - 1
        expected_r_index = self._reward_index(self._compare(a, opponent_action))
        return 1.0 if r_index == expected_r_index else 0.0

    # -- Utilitaires internes --

    # Résultat d'un round : +1 si `a` bat `b`, -1 si `a` perd contre `b`, 0 si égalité.
    # `(a - b) % 3 == 1` capture exactement les 3 cas de victoire (Pierre bat
    # Ciseaux, Papier bat Pierre, Ciseaux bat Papier).
    def _compare(self, a: int, b: int) -> float:
        if a == b:
            return 0.0
        if (a - b) % 3 == 1:
            return 1.0
        return -1.0

    # Convertit une récompense (-1, 0 ou 1) en indice dans REWARDS.
    def _reward_index(self, reward: float) -> int:
        return int(reward) + 1
