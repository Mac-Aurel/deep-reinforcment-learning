from src.rl.environments.grid_world import DOWN, GridWorldEnv, RIGHT
from src.rl.environments.line_world import LineWorldEnv
from src.rl.environments.monty_hall import SWITCH, MontyHallEnv
from src.rl.environments.two_round_rps import ROCK, TwoRoundRPSEnv
from src.rl.manual_play import play_manually


def _scripted_input(actions):
    actions_iter = iter(actions)
    return lambda: next(actions_iter)


def test_play_manually_on_line_world():
    env = LineWorldEnv(num_cells=3)
    score = play_manually(env, action_input=_scripted_input(["1"]))
    assert score == 1.0


def test_play_manually_retries_on_invalid_input():
    env = LineWorldEnv(num_cells=3)
    # "abc" n'est pas un nombre, "5" est hors du domaine des actions valides,
    # "1" est enfin une action valide (aller à droite).
    score = play_manually(env, action_input=_scripted_input(["abc", "5", "1"]))
    assert score == 1.0


def test_play_manually_on_grid_world():
    env = GridWorldEnv(height=2, width=2, goal=(1, 1), trap=(1, 0))
    score = play_manually(env, action_input=_scripted_input([str(RIGHT), str(DOWN)]))
    assert score == 1.0


def test_play_manually_on_two_round_rps():
    # Round 2 : l'adversaire est forcé de rejouer le coup du round 1 (Pierre) ;
    # rejouer Pierre au round 2 donne donc toujours un match nul (0), quel
    # que soit le résultat aléatoire du round 1 (-1, 0 ou +1).
    env = TwoRoundRPSEnv()
    score = play_manually(env, action_input=_scripted_input([str(ROCK), str(ROCK)]))
    assert score in (-1.0, 0.0, 1.0)


def test_play_manually_on_monty_hall():
    env = MontyHallEnv(num_doors=3)
    score = play_manually(env, action_input=_scripted_input(["0", str(SWITCH)]))
    assert score in (0.0, 1.0)
