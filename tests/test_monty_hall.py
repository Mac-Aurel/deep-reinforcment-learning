import numpy as np
import pytest

from src.rl.algorithms.dynamic_programming import value_iteration
from src.rl.environments.monty_hall import STAY, SWITCH, MontyHallEnv
from src.rl.envs import Environment, ExploringStartsEnvironment, MDPEnvironment


def test_monty_hall_implements_all_interfaces():
    env = MontyHallEnv()
    assert isinstance(env, Environment)
    assert isinstance(env, MDPEnvironment)
    assert isinstance(env, ExploringStartsEnvironment)


def test_rejects_fewer_than_3_doors():
    with pytest.raises(ValueError):
        MontyHallEnv(num_doors=2)


def test_reset_starts_at_state_0_with_all_doors_in_play():
    env = MontyHallEnv(num_doors=3)
    env.step(0)
    env.reset()
    assert env.current_state() == 0
    assert len(env.remaining_doors) == 3


def test_level_1_requires_exactly_2_decisions():
    # 3 portes : 1 choix initial + 1 décision garder/changer, comme décrit
    # dans le sujet ("l'agent doit prendre 2 décisions successives").
    env = MontyHallEnv(num_doors=3)
    env.step(0)
    assert not env.is_game_over()
    env.step(STAY)
    assert env.is_game_over()


def test_level_2_requires_exactly_4_decisions():
    # 5 portes : 1 choix initial + 3 décisions garder/changer, comme décrit
    # dans le sujet ("l'agent doit effectuer 4 actions successives").
    env = MontyHallEnv(num_doors=5)
    env.step(0)
    for _ in range(3):
        assert not env.is_game_over()
        env.step(STAY)
    assert env.is_game_over()


def test_available_actions_are_doors_first_then_stay_or_switch():
    env = MontyHallEnv(num_doors=5)
    assert env.available_actions() == [0, 1, 2, 3, 4]
    env.step(0)
    assert env.available_actions() == [STAY, SWITCH]


def test_step_with_invalid_action_raises():
    env = MontyHallEnv(num_doors=3)
    with pytest.raises(ValueError):
        env.step(3)


def test_step_after_game_over_raises():
    env = MontyHallEnv(num_doors=3)
    env.step(0)
    env.step(STAY)
    with pytest.raises(RuntimeError):
        env.step(STAY)


def test_transition_probabilities_sum_to_one_for_non_terminal_states():
    env = MontyHallEnv(num_doors=5)
    for s in env.states():
        if s in env.terminal_states():
            continue
        for a in env.actions():
            total = sum(
                env.transition_probability(s, a, s_p, r_index)
                for s_p in env.states()
                for r_index in range(len(env.rewards()))
            )
            if total == 0.0:
                continue  # action non valide dans cet état (ex: garder/changer hors [0, 1])
            assert total == pytest.approx(1.0)


def test_value_iteration_matches_the_classic_two_thirds_result_on_3_doors():
    # Résultat classique du paradoxe de Monty Hall : toujours changer donne
    # 2/3 de chances de gagner (contre 1/3 en gardant son choix initial).
    env = MontyHallEnv(num_doors=3)
    pi, V = value_iteration(env)

    assert V[0] == pytest.approx(2.0 / 3.0, abs=1e-3)
    switch_state = 1  # seul état non terminal après le tout premier choix
    assert pi[switch_state].argmax() == SWITCH


def test_value_iteration_matches_four_fifths_result_on_5_doors():
    # Généralisation à 5 portes : la stratégie optimale est de garder son
    # choix jusqu'au tout dernier tour puis de changer, pour 4/5 de chances
    # de gagner au final.
    env = MontyHallEnv(num_doors=5)
    pi, V = value_iteration(env)

    assert V[0] == pytest.approx(4.0 / 5.0, abs=1e-3)


def test_simulation_matches_theory_always_switch_on_the_last_decision():
    np.random.seed(0)
    for num_doors, expected in [(3, 2.0 / 3.0), (5, 4.0 / 5.0)]:
        env = MontyHallEnv(num_doors=num_doors)
        trials = 5_000
        wins = 0.0
        for _ in range(trials):
            env.reset()
            while not env.is_game_over():
                action = SWITCH if len(env.remaining_doors) == 2 else STAY
                env.step(action)
            wins += env.score()
        assert wins / trials == pytest.approx(expected, abs=0.03)
