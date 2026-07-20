import numpy as np
import pytest

from src.rl.algorithms.dynamic_programming import value_iteration
from src.rl.algorithms.td import q_learning
from src.rl.environments.line_world import LineWorldEnv
from src.rl.experiment import evaluate_policy, measure_convergence, run_and_save_experiment, run_experiment
from src.rl.persistence import load_policy


def test_evaluate_policy_on_a_hand_verifiable_optimal_policy():
    env = LineWorldEnv(num_cells=3)
    pi = np.array([[0, 0], [0, 1], [0, 0]])  # depuis l'état 1 (seul état non terminal), toujours à droite

    mean_score, mean_length = evaluate_policy(env, pi, episodes_count=50)

    assert mean_score == pytest.approx(1.0)
    assert mean_length == pytest.approx(1.0)


def test_run_experiment_trains_and_evaluates_q_learning():
    np.random.seed(0)
    result = run_experiment(
        env_factory=lambda: LineWorldEnv(num_cells=5),
        algo=q_learning,
        algo_kwargs={"iterations_count": 3000},
    )

    assert result["algo_name"] == "q_learning"
    assert result["mean_score"] == pytest.approx(1.0, abs=0.1)
    assert result["training_time_seconds"] >= 0.0
    assert result["pi"].shape == (5, 2)
    assert result["value_function"].shape == (5, 2)


def test_run_experiment_works_with_dynamic_programming_too():
    # value_iteration renvoie (pi, V) au lieu de (pi, Q), et son hyperparamètre
    # de budget s'appelle max_iterations et non iterations_count : le harness
    # ne doit rien supposer de plus qu'un algorithme qui renvoie (pi, quelque chose).
    result = run_experiment(
        env_factory=lambda: LineWorldEnv(num_cells=5),
        algo=value_iteration,
        algo_kwargs={"max_iterations": 1000},
    )

    assert result["algo_name"] == "value_iteration"
    assert result["mean_score"] == pytest.approx(1.0, abs=0.1)
    assert result["value_function"].ndim == 1


def test_run_and_save_experiment_persists_the_policy(tmp_path):
    np.random.seed(0)
    summary = run_and_save_experiment(
        env_name="line_world",
        env_factory=lambda: LineWorldEnv(num_cells=5),
        algo=q_learning,
        results_dir=str(tmp_path),
        algo_kwargs={"iterations_count": 3000},
    )

    assert summary["env_name"] == "line_world"
    assert summary["mean_score"] == pytest.approx(1.0, abs=0.1)
    loaded = load_policy(summary["saved_to"])
    assert "pi" in loaded
    assert "Q" in loaded


def test_measure_convergence_returns_one_result_per_budget():
    np.random.seed(0)
    curve = measure_convergence(
        env_factory=lambda: LineWorldEnv(num_cells=3),
        algo=q_learning,
        iteration_budgets=[50, 500, 2000],
    )

    assert [point["iterations"] for point in curve] == [50, 500, 2000]
    assert curve[-1]["mean_score"] == pytest.approx(1.0, abs=0.15)
