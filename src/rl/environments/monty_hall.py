from typing import Dict, List, Tuple, Union

import numpy as np

from src.rl.envs import ExploringStartsEnvironment, MDPEnvironment

# Les 2 décisions possibles une fois le premier choix fait : garder sa porte, ou en changer.
STAY, SWITCH = 0, 1

# Valeurs de récompense possibles à l'ouverture finale : perdu (0.0), gagné (1.0).
REWARDS = np.array([0.0, 1.0])


class MontyHallEnv(ExploringStartsEnvironment, MDPEnvironment):
    """
    Jeu de Monty Hall à `num_doors` portes (3 portes = "level 1", 5 portes =
    "level 2" du sujet, mêmes règles sinon). Une porte gagnante est tirée au
    hasard et cachée. L'agent choisit d'abord une porte parmi les
    `num_doors` (rien n'est encore révélé), puis le présentateur retire une
    porte perdante parmi celles non choisies. L'agent peut alors garder son
    choix ou en changer ; ceci se répète jusqu'à ce qu'il ne reste plus que
    2 portes, moment où son choix final est révélé (+1.0 si gagnant, 0.0
    sinon). Au total l'agent prend `num_doors - 1` décisions (1 choix
    initial + (num_doors - 2) décisions garder/changer), ce qui correspond
    exactement aux "2 décisions" du level 1 (3 portes) et aux "4 actions"
    du level 2 (5 portes) décrits dans le sujet.

    Toutes les portes sont interchangeables par symétrie : l'identité d'une
    porte précise n'apporte aucune information utile. L'état exposé aux
    algorithmes n'est donc pas "quelle porte est choisie", mais un état
    ABSTRAIT (nombre de portes encore en jeu, probabilité que le choix
    courant soit la porte gagnante). Cette probabilité évolue de façon
    purement déterministe selon les décisions garder/changer prises jusque
    là : garder la conserve à l'identique ; changer la remplace par
    (1 - proba) / (portes restantes - 1) (résultat classique du paradoxe de
    Monty Hall généralisé à plusieurs portes/tours). C'est donc un état
    markovien suffisant. La simulation de portes concrètes (tirage de la
    porte gagnante, retrait effectif) n'est utilisée que pour l'affichage
    et le calcul du résultat final, pas pour l'état exposé.
    """

    def __init__(self, num_doors: int = 3) -> None:
        if num_doors < 3:
            raise ValueError("Monty Hall needs at least 3 doors")
        self.num_doors = num_doors

        # Liste des états abstraits (portes_restantes, croyance) ; la
        # croyance vaut None pour l'état initial (avant tout choix).
        self._abstract_states: List[Tuple[int, float]] = [(num_doors, None)]
        # (state_id, action) -> soit un state_id suivant (transition
        # déterministe, rien n'est encore gagné/perdu), soit un tuple
        # ("final", proba_de_gain) pour la toute dernière décision.
        self._transitions: Dict[Tuple[int, int], Union[int, Tuple[str, float]]] = {}
        self._build_state_graph()
        self._terminal_state_id = len(self._abstract_states)
        self._abstract_states.append((0, None))

        self.winning_door = 0
        self.remaining_doors: List[int] = []
        self.current_choice = -1
        self.state_id = 0
        self.last_reward = 0.0
        # reset() tire la porte gagnante et remet toutes les portes en jeu,
        # pour que l'environnement soit directement jouable sans appel préalable.
        self.reset()

    # -- Construction de l'espace d'états abstrait (portes restantes, croyance) --

    # Retrouve l'identifiant d'un état abstrait déjà connu, ou le crée s'il
    # n'existe pas encore (deux chemins de décisions différents peuvent
    # mener exactement à la même croyance, auquel cas ils partagent le même état).
    def _get_or_create_state(self, doors_left: int, belief: float) -> int:
        for i, (k, b) in enumerate(self._abstract_states):
            if k == doors_left and b is not None and abs(b - belief) < 1e-9:
                return i
        self._abstract_states.append((doors_left, belief))
        return len(self._abstract_states) - 1

    # Explore récursivement (par une simple pile) tous les états atteignables
    # à partir du premier choix, en appliquant les formules garder/changer.
    def _build_state_graph(self) -> None:
        first_belief = 1.0 / self.num_doors
        first_id = self._get_or_create_state(self.num_doors - 1, first_belief)
        for a in range(self.num_doors):
            self._transitions[(0, a)] = first_id

        frontier = [(first_id, self.num_doors - 1, first_belief)]
        seen = {first_id}
        while frontier:
            state_id, doors_left, belief = frontier.pop()
            is_final_round = doors_left == 2

            for action, new_belief in ((STAY, belief), (SWITCH, (1.0 - belief) / (doors_left - 1))):
                if is_final_round:
                    self._transitions[(state_id, action)] = ("final", new_belief)
                    continue

                next_state_id = self._get_or_create_state(doors_left - 1, new_belief)
                self._transitions[(state_id, action)] = next_state_id
                if next_state_id not in seen:
                    seen.add(next_state_id)
                    frontier.append((next_state_id, doors_left - 1, new_belief))

    # -- Interface model-free (Environment), utilisée par Monte Carlo / TD / Planning --

    # Tire au hasard la porte gagnante (cachée) et remet toutes les portes en jeu.
    def reset(self) -> None:
        self.winning_door = int(np.random.randint(self.num_doors))
        self.remaining_doors = list(range(self.num_doors))
        self.current_choice = -1
        self.state_id = 0
        self.last_reward = 0.0

    # Premier appel : choisit une porte parmi `num_doors`. Appels suivants :
    # garde son choix (STAY) ou en change pour une autre porte restante
    # tirée au hasard (SWITCH, toutes les autres portes étant équivalentes
    # par symétrie). Le présentateur retire ensuite une porte perdante,
    # sauf s'il ne reste plus que 2 portes : la révélation finale a alors lieu.
    def step(self, action: int) -> None:
        if action not in self.available_actions():
            raise ValueError("Action is invalid in current state")
        if self.is_game_over():
            raise RuntimeError("Trying to play while game is over")

        if self.state_id == 0:
            self.current_choice = action
        elif action == SWITCH:
            others = [d for d in self.remaining_doors if d != self.current_choice]
            self.current_choice = int(np.random.choice(others))

        next_state_id = self._transitions[(self.state_id, action)]

        if len(self.remaining_doors) > 2:
            removable = [d for d in self.remaining_doors if d != self.current_choice and d != self.winning_door]
            removed = int(np.random.choice(removable))
            self.remaining_doors.remove(removed)
            self.state_id = next_state_id
        else:
            self.last_reward = 1.0 if self.current_choice == self.winning_door else 0.0
            self.state_id = self._terminal_state_id

    # La partie est terminée une fois la révélation finale faite.
    def is_game_over(self) -> bool:
        return self.state_id == self._terminal_state_id

    def current_state(self) -> int:
        return self.state_id

    # Choisir une porte au premier tour (num_doors options), garder/changer ensuite (2 options).
    def available_actions(self) -> List[int]:
        if self.state_id == 0:
            return list(range(self.num_doors))
        return [STAY, SWITCH]

    # 0.0 tant que la partie n'est pas terminée, 1.0 ou 0.0 une fois la porte finale ouverte.
    def score(self) -> float:
        return self.last_reward

    def num_states(self) -> int:
        return len(self._abstract_states)

    def num_actions(self) -> int:
        return self.num_doors

    # Place directement l'environnement dans l'état abstrait demandé (doit
    # être un état non terminal) : reconstruit une situation concrète
    # (portes renumérotées arbitrairement, choix courant fixé par
    # convention) dont la porte gagnante est tirée au hasard de façon à
    # respecter exactement la croyance associée à cet état.
    def set_state(self, state: int) -> None:
        doors_left, belief = self._abstract_states[state]
        self.state_id = state
        self.last_reward = 0.0

        if belief is None:
            self.winning_door = int(np.random.randint(self.num_doors))
            self.remaining_doors = list(range(self.num_doors))
            self.current_choice = -1
            return

        self.remaining_doors = list(range(doors_left))
        self.current_choice = 0
        if np.random.random() < belief:
            self.winning_door = self.current_choice
        else:
            others = [d for d in self.remaining_doors if d != self.current_choice]
            self.winning_door = int(np.random.choice(others))

    # Pretty print en ligne de commande : porte choisie entre crochets,
    # portes retirées barrées d'une croix.
    def display(self) -> None:
        labels = []
        for d in range(self.num_doors):
            name = chr(ord("A") + d)
            if d not in self.remaining_doors:
                labels.append("X")
            elif d == self.current_choice:
                labels.append(f"[{name}]")
            else:
                labels.append(name)
        print(" ".join(labels))
        if self.is_game_over():
            result = "gagné" if self.last_reward == 1.0 else "perdu"
            print(f"Porte gagnante : {chr(ord('A') + self.winning_door)} -> {result}")

    # -- Interface modèle complet (MDPEnvironment), utilisée par Policy/Value Iteration --

    def states(self) -> np.ndarray:
        return np.arange(self.num_states())

    def actions(self) -> np.ndarray:
        return np.arange(self.num_doors)

    def rewards(self) -> np.ndarray:
        return REWARDS

    def terminal_states(self) -> np.ndarray:
        return np.array([self._terminal_state_id])

    def transition_probability(self, s: int, a: int, s_p: int, r_index: int) -> float:
        if s in self.terminal_states():
            return 0.0
        if (s, a) not in self._transitions:
            return 0.0

        transition = self._transitions[(s, a)]
        if isinstance(transition, tuple):
            _, win_probability = transition
            if s_p != self._terminal_state_id:
                return 0.0
            if r_index == 1:
                return win_probability
            if r_index == 0:
                return 1.0 - win_probability
            return 0.0

        if s_p == transition and r_index == 0:
            return 1.0
        return 0.0
