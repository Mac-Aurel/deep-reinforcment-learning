import numpy as np
import pytest

from src.rl.algorithms.dynamic_programming import value_iteration
from src.rl.algorithms.td import q_learning
from src.rl.environments.two_round_rps import PAPER, ROCK, SCISSORS, TwoRoundRPSEnv
from src.rl.envs import Environment, ExploringStartsEnvironment, MDPEnvironment


def test_two_round_rps_implements_all_interfaces():
    env = TwoRoundRPSEnv()
    assert isinstance(env, Environment)
    assert isinstance(env, MDPEnvironment)
    assert isinstance(env, ExploringStartsEnvironment)


def test_set_state_moves_to_the_right_round():
    env = TwoRoundRPSEnv()
    env.set_state(1 + ROCK)
    assert env.current_state() == 1 + ROCK
    assert env.agent_round1_action == ROCK


def test_reset_starts_at_round_1_with_zero_score():
    env = TwoRoundRPSEnv()
    env.step(ROCK)
    env.reset()
    assert env.current_state() == 0
    assert env.score() == 0.0


def test_round_2_state_encodes_round_1_move():
    env = TwoRoundRPSEnv()
    env.step(PAPER)
    assert env.current_state() == 1 + PAPER
    assert not env.is_game_over()


def test_round_2_opponent_is_forced_to_replay_round_1_move():
    # L'adversaire du round 2 rejoue FORCÉMENT le coup de l'agent au round 1 :
    # jouer ce même coup au round 2 doit donc toujours donner un match nul.
    env = TwoRoundRPSEnv()
    env.step(SCISSORS)
    prev_score = env.score()
    env.step(SCISSORS)
    assert env.score() - prev_score == 0.0
    assert env.is_game_over()


def test_step_with_invalid_action_raises():
    env = TwoRoundRPSEnv()
    with pytest.raises(ValueError):
        env.step(3)


def test_step_after_game_over_raises():
    env = TwoRoundRPSEnv()
    env.step(ROCK)
    env.step(ROCK)
    with pytest.raises(RuntimeError):
        env.step(ROCK)


def test_transition_probabilities_sum_to_one_for_non_terminal_states():
    env = TwoRoundRPSEnv()
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


def test_value_iteration_learns_to_always_counter_round_1_move_in_round_2():
    # Round 1 : l'adversaire est aléatoire, donc l'espérance de gain y est
    # nulle quel que soit le coup joué. Round 2 : l'adversaire est connu
    # (il rejoue le coup de l'agent au round 1), donc la politique optimale
    # doit toujours jouer le coup qui bat ce coup connu, pour une victoire
    # garantie (+1). La valeur totale optimale doit donc être proche de 1.
    env = TwoRoundRPSEnv()
    pi, V = value_iteration(env)

    assert V[0] == pytest.approx(1.0, abs=1e-3)
    for a in range(3):
        counter_move = (a + 1) % 3
        assert pi[1 + a].argmax() == counter_move


def test_q_learning_converges_on_two_round_rps():
    np.random.seed(0)
    env = TwoRoundRPSEnv()
    pi, Q = q_learning(env, iterations_count=8_000)

    for a in range(3):
        counter_move = (a + 1) % 3
        assert pi[1 + a].argmax() == counter_move
        assert Q[1 + a, counter_move] == pytest.approx(1.0, abs=0.05)
