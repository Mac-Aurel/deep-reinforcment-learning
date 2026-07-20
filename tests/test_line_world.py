import pytest

from src.rl.environments.line_world import LineWorldEnv
from src.rl.envs import Environment, ExploringStartsEnvironment, MDPEnvironment


def test_line_world_implements_all_interfaces():
    env = LineWorldEnv()
    assert isinstance(env, Environment)
    assert isinstance(env, MDPEnvironment)
    assert isinstance(env, ExploringStartsEnvironment)


def test_set_state_moves_the_agent_directly():
    env = LineWorldEnv(num_cells=5)
    env.set_state(3)
    assert env.current_state() == 3


def test_reset_places_agent_in_the_middle():
    env = LineWorldEnv(num_cells=5)
    env.step(1)
    env.reset()
    assert env.current_state() == 2


def test_reaching_the_right_end_scores_plus_one():
    env = LineWorldEnv(num_cells=5)
    while not env.is_game_over():
        env.step(1)
    assert env.current_state() == 4
    assert env.score() == 1.0


def test_reaching_the_left_end_scores_minus_one():
    env = LineWorldEnv(num_cells=5)
    while not env.is_game_over():
        env.step(0)
    assert env.current_state() == 0
    assert env.score() == -1.0


def test_step_with_invalid_action_raises():
    env = LineWorldEnv(num_cells=5)
    with pytest.raises(ValueError):
        env.step(2)


def test_step_after_game_over_raises():
    env = LineWorldEnv(num_cells=5)
    while not env.is_game_over():
        env.step(1)
    with pytest.raises(RuntimeError):
        env.step(1)


def test_transition_probabilities_sum_to_one_for_non_terminal_states():
    env = LineWorldEnv(num_cells=5)
    for s in env.states():
        if s in env.terminal_states():
            continue
        for a in env.actions():
            total = sum(
                env.transition_probability(s, a, s_p, r_index)
                for s_p in env.states()
                for r_index in range(len(env.rewards()))
            )
            assert total == pytest.approx(1.0)


def test_transition_probabilities_match_original_notebook_model():
    # Régression : ces valeurs sont celles codées à la main dans
    # notebooks/Line World Experiments.ipynb pour LINE_WORLD_LENGTH = 5.
    env = LineWorldEnv(num_cells=5)
    expected_ones = {
        (1, 0, 0, 1),
        (3, 1, 4, 2),
        (2, 0, 1, 0),
        (3, 0, 2, 0),
        (1, 1, 2, 0),
        (2, 1, 3, 0),
    }
    for s in range(5):
        for a in range(2):
            for s_p in range(5):
                for r_index in range(3):
                    expected = 1.0 if (s, a, s_p, r_index) in expected_ones else 0.0
                    assert env.transition_probability(s, a, s_p, r_index) == expected
