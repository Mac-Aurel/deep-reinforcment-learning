from typing import List

import numpy as np
import pytest

from src.rl.envs import Environment, MDPEnvironment


# Environnement jouet à 3 états (0, 1, 2) utilisé uniquement pour vérifier
# que les deux interfaces peuvent être combinées sur un même environnement
# et que leur contrat est respecté.
class TinyLineEnv(Environment, MDPEnvironment):
    def __init__(self) -> None:
        self.position = 1

    def reset(self) -> None:
        self.position = 1

    def step(self, action: int) -> None:
        self.position += 1 if action == 1 else -1

    def is_game_over(self) -> bool:
        return self.position in (0, 2)

    def current_state(self) -> int:
        return self.position

    def available_actions(self) -> List[int]:
        return [0, 1]

    def score(self) -> float:
        if self.position == 2:
            return 1.0
        if self.position == 0:
            return -1.0
        return 0.0

    def num_states(self) -> int:
        return 3

    def num_actions(self) -> int:
        return 2

    def display(self) -> None:
        print("".join("X" if s == self.position else "_" for s in range(3)))

    def states(self) -> np.ndarray:
        return np.array([0, 1, 2])

    def actions(self) -> np.ndarray:
        return np.array([0, 1])

    def rewards(self) -> np.ndarray:
        return np.array([0.0, -1.0, 1.0])

    def terminal_states(self) -> np.ndarray:
        return np.array([0, 2])

    def transition_probability(self, s: int, a: int, s_p: int, r_index: int) -> float:
        if s == 1 and a == 0 and s_p == 0 and r_index == 1:
            return 1.0
        if s == 1 and a == 1 and s_p == 2 and r_index == 2:
            return 1.0
        return 0.0


def test_environment_cannot_be_instantiated_directly():
    with pytest.raises(TypeError):
        Environment()


def test_mdp_environment_cannot_be_instantiated_directly():
    with pytest.raises(TypeError):
        MDPEnvironment()


def test_concrete_env_implements_both_interfaces():
    env = TinyLineEnv()
    assert isinstance(env, Environment)
    assert isinstance(env, MDPEnvironment)


def test_model_free_episode_runs_to_completion():
    env = TinyLineEnv()
    env.reset()
    while not env.is_game_over():
        env.step(1)
    assert env.score() == 1.0


def test_mdp_model_transition_probability_sums_to_one_per_action():
    env = TinyLineEnv()
    for a in env.actions():
        total = sum(
            env.transition_probability(1, a, s_p, r_index)
            for s_p in env.states()
            for r_index in range(len(env.rewards()))
        )
        assert total == pytest.approx(1.0)
