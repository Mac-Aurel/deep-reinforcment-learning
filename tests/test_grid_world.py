import pytest

from src.rl.algorithms.dynamic_programming import value_iteration
from src.rl.environments.grid_world import DOWN, LEFT, RIGHT, UP, GridWorldEnv
from src.rl.envs import Environment, ExploringStartsEnvironment, MDPEnvironment


def test_grid_world_implements_all_interfaces():
    env = GridWorldEnv()
    assert isinstance(env, Environment)
    assert isinstance(env, MDPEnvironment)
    assert isinstance(env, ExploringStartsEnvironment)


def test_set_state_moves_the_agent_directly():
    env = GridWorldEnv(height=5, width=5)
    env.set_state(7)
    assert env.current_state() == 7
    assert (env.row, env.col) == (1, 2)


def test_reset_places_agent_top_left():
    env = GridWorldEnv()
    env.step(RIGHT)
    env.reset()
    assert env.current_state() == 0


def test_bumping_into_a_wall_keeps_the_agent_in_place():
    env = GridWorldEnv()
    env.step(LEFT)
    assert env.current_state() == 0
    env.step(UP)
    assert env.current_state() == 0


def test_reaching_the_goal_scores_plus_one():
    env = GridWorldEnv(height=1, width=3, goal=(0, 2), trap=(0, 0))
    env.row, env.col = 0, 1
    env.step(RIGHT)
    assert env.is_game_over()
    assert env.score() == 1.0


def test_reaching_the_trap_scores_minus_one_on_hand_verifiable_grid():
    # Grille 1x3 : partir de (0,0), le piège est juste à côté en (0,1).
    # Un seul pas à droite doit terminer la partie avec un score de -1.
    env = GridWorldEnv(height=1, width=3, goal=(0, 2), trap=(0, 1))
    env.step(RIGHT)
    assert env.current_state() == 1
    assert env.is_game_over()
    assert env.score() == -1.0


def test_step_with_invalid_action_raises():
    env = GridWorldEnv()
    with pytest.raises(ValueError):
        env.step(4)


def test_step_after_game_over_raises():
    env = GridWorldEnv(height=1, width=3, goal=(0, 2), trap=(0, 1))
    env.step(RIGHT)
    with pytest.raises(RuntimeError):
        env.step(RIGHT)


def test_transition_probabilities_sum_to_one_for_non_terminal_states():
    env = GridWorldEnv()
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


def test_value_iteration_avoids_the_trap_and_reaches_the_goal():
    # Le chemin le plus court de (0,0) à (4,4) qui évite le piège en (4,3)
    # fait 8 pas (4 à droite, 4 en bas). Avec gamma proche de 1, la valeur
    # optimale de l'état de départ doit donc être proche de 1.
    env = GridWorldEnv()
    pi, V = value_iteration(env)

    assert V[0] == pytest.approx(1.0, abs=1e-3)

    trap_state = env._state_id(*env.trap)
    for s in env.states():
        if s in env.terminal_states():
            continue
        best_action = int(pi[s].argmax())
        row, col = env._position(s)
        next_row, next_col = env._next_position(row, col, best_action)
        assert env._state_id(next_row, next_col) != trap_state
