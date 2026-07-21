from typing import Callable, List

from src.rl.envs import Environment


# Demande une action à l'humain jusqu'à obtenir une entrée valide (un entier
# parmi les actions actuellement disponibles) : on ne fait confiance à rien
# de ce qui vient d'une saisie clavier.
def _read_action(available: List[int], action_input: Callable[[], str]) -> int:
    while True:
        raw = action_input()
        try:
            action = int(raw)
        except ValueError:
            print("Entrée invalide, entrez un nombre.")
            continue
        if action not in available:
            print(f"Action invalide, choisissez parmi {available}.")
            continue
        return action


# Boucle de jeu générique pour qu'un humain joue manuellement un épisode
# complet sur n'importe quel environnement respectant l'interface commune
# (reset/step/is_game_over/current_state/available_actions/score/display) :
# à chaque tour, affiche l'état et les actions possibles, joue l'action
# choisie, puis affiche la récompense obtenue. Sert à la fois de mode de jeu
# manuel et d'outil de debug pour vérifier que les règles d'un environnement
# sont bien celles attendues. `action_input` est injectable (au lieu de
# toujours lire au clavier via `input`) pour pouvoir être testé sans
# interaction réelle.
def play_manually(env: Environment, action_input: Callable[[], str] = input) -> float:
    env.reset()
    env.display()

    while not env.is_game_over():
        available = env.available_actions()
        print(f"Actions disponibles : {available}")
        action = _read_action(available, action_input)

        prev_score = env.score()
        env.step(action)
        reward = env.score() - prev_score
        print(f"Récompense obtenue : {reward}")
        env.display()

    print(f"Partie terminée, score final : {env.score()}")
    return env.score()
