import numpy as np

from src.rl.persistence import build_result_filename, load_policy, save_policy


def test_save_and_load_policy_roundtrip_with_all_arrays(tmp_path):
    pi = np.array([[1.0, 0.0], [0.0, 1.0]])
    V = np.array([0.5, 0.9])
    Q = np.array([[0.1, 0.2], [0.3, 0.4]])
    filepath = str(tmp_path / "result.npz")

    save_policy(filepath, pi=pi, V=V, Q=Q)
    loaded = load_policy(filepath)

    assert np.array_equal(loaded["pi"], pi)
    assert np.array_equal(loaded["V"], V)
    assert np.array_equal(loaded["Q"], Q)


def test_save_and_load_policy_with_only_pi(tmp_path):
    pi = np.array([[1.0, 0.0]])
    filepath = str(tmp_path / "result.npz")

    save_policy(filepath, pi=pi)
    loaded = load_policy(filepath)

    assert np.array_equal(loaded["pi"], pi)
    assert "V" not in loaded
    assert "Q" not in loaded


def test_build_result_filename_is_deterministic_regardless_of_kwargs_order():
    name1 = build_result_filename("line_world", "q_learning", alpha=0.1, epsilon=0.2)
    name2 = build_result_filename("line_world", "q_learning", epsilon=0.2, alpha=0.1)

    assert name1 == name2
    assert name1 == "line_world__q_learning__alpha=0.1_epsilon=0.2.npz"


def test_build_result_filename_without_hyperparameters():
    assert build_result_filename("grid_world", "value_iteration") == "grid_world__value_iteration.npz"
