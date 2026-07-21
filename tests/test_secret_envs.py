import numpy as np
import pytest

from src.rl.algorithms.td import q_learning
from src.rl.envs import Environment
from src.rl.environments.secret_envs import SecretEnvAdapter
from src.rl.experiment import evaluate_policy


def test_rejects_invalid_index():
    with pytest.raises(ValueError):
        SecretEnvAdapter(4)


@pytest.mark.parametrize("index", [0, 1, 2, 3])
def test_secret_env_implements_environment_interface(index):
    env = SecretEnvAdapter(index)
    assert isinstance(env, Environment)
    assert env.num_states() > 0
    assert env.num_actions() > 0
    assert not env.is_game_over()
    assert len(env.available_actions()) > 0


@pytest.mark.parametrize("index", [0, 1, 2, 3])
def test_secret_env_can_play_a_full_episode(index):
    env = SecretEnvAdapter(index)
    env.reset()
    steps = 0
    while not env.is_game_over() and steps < 1_000:
        env.step(env.available_actions()[0])
        steps += 1
    assert env.is_game_over()


def test_q_learning_runs_on_secret_env_0():
    np.random.seed(0)
    env = SecretEnvAdapter(0)
    pi, Q = q_learning(env, iterations_count=500, epsilon=0.3)

    eval_env = SecretEnvAdapter(0)
    mean_score, mean_length = evaluate_policy(eval_env, pi, episodes_count=20)
    assert mean_length > 0
