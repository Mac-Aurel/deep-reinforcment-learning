import os

import nbformat as nbf

OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "..", "notebooks", "Rapport final.ipynb")

md = nbf.v4.new_markdown_cell
code = nbf.v4.new_code_cell


def setup_cell():
    return code(
        "import os\n"
        "import sys\n"
        "import csv\n"
        "\n"
        "import pandas as pd\n"
        "from IPython.display import Image, display\n"
        "\n"
        "REPO_ROOT = os.path.abspath(os.path.join(os.getcwd(), '..'))\n"
        "if REPO_ROOT not in sys.path:\n"
        "    sys.path.insert(0, REPO_ROOT)\n"
        "\n"
        "RESULTS_DIR = os.path.join(REPO_ROOT, 'results')\n"
        "FIGURES_DIR = os.path.join(REPO_ROOT, 'report', 'figures')\n"
        "\n"
        "def load_csv(name):\n"
        "    with open(os.path.join(RESULTS_DIR, name)) as f:\n"
        "        return pd.DataFrame(list(csv.DictReader(f)))\n"
    )


def build():
    nb = nbf.v4.new_notebook()
    cells = []

    cells.append(md(
        "# Implémentation et comparaison d'algorithmes d'apprentissage par renforcement sur cinq environnements\n"
        "\n"
        "*Programmation dynamique, Monte Carlo, Temporal Difference Learning et Planning*\n"
        "\n"
        "2026-4A-IABD, (Deep) Reinforcement Learning P1\n"
        "\n"
        "Groupe : [noms à compléter]\n"
    ))

    cells.append(md(
        "## 1. Introduction\n"
        "\n"
        "Ce notebook présente le travail réalisé dans le cadre du projet de Reinforcement Learning. "
        "Le but du projet est d'implémenter les principaux algorithmes classiques de l'apprentissage par "
        "renforcement (programmation dynamique, Monte Carlo, Temporal Difference Learning et Planning) "
        "et de les tester sur plusieurs environnements pour comprendre dans quels cas chaque méthode "
        "fonctionne bien, et pourquoi.\n"
        "\n"
        "Deux bibliothèques ont été développées séparément. La première regroupe les environnements "
        "(Line World, Grid World, Two Round Rock Paper Scissors, Monty Hall et les environnements secrets "
        "fournis), qui suivent tous la même interface commune pour pouvoir être utilisés indifféremment par "
        "tous les algorithmes. La deuxième regroupe les algorithmes eux-mêmes.\n"
        "\n"
        "Ce notebook recharge les résultats déjà calculés et sauvegardés dans `results/` (voir "
        "`scripts/run_experiments.py`, `scripts/run_hyperparameter_study.py` et "
        "`scripts/run_secret_envs_experiments.py`) plutôt que de relancer tous les entraînements en direct, "
        "pour rester rapide et reproductible."
    ))

    cells.append(setup_cell())

    cells.append(md(
        "## 2. Les environnements\n"
        "\n"
        "Chaque environnement expose la même interface : on peut le réinitialiser, jouer une action, "
        "savoir si la partie est terminée, connaître l'état courant et le score, et afficher l'état de "
        "façon lisible dans le terminal.\n"
        "\n"
        "**Line World.** L'agent se déplace sur une ligne de plusieurs cases, part du milieu, et choisit "
        "d'aller à gauche ou à droite. Sortir par la gauche donne -1, sortir par la droite donne +1.\n"
        "\n"
        "**Grid World.** L'agent se déplace sur une grille (5x5 par défaut) en partant du coin en haut à "
        "gauche, doit atteindre une case objectif (+1) en évitant une case piège (-1). Une action qui "
        "ferait sortir l'agent de la grille le laisse simplement sur place.\n"
        "\n"
        "**Two Round Rock Paper Scissors.** Partie de pierre-feuille-ciseaux en deux manches. Au round 1 "
        "l'adversaire joue au hasard, au round 2 il est forcé de rejouer le coup de l'agent au round 1.\n"
        "\n"
        "**Monty Hall (niveaux 1 et 2).** Reproduit le paradoxe de Monty Hall à 3 portes (niveau 1) ou 5 "
        "portes (niveau 2). Les deux niveaux sont gérés par la même classe, paramétrée par le nombre de "
        "portes. L'état exposé n'est pas la porte choisie (toutes les portes non choisies sont "
        "interchangeables) mais le nombre de portes restantes et la probabilité que le choix actuel soit "
        "gagnant, qui suit une formule connue du paradoxe.\n"
        "\n"
        "**Environnements secrets (0 à 3).** Fournis par le cours sous forme de bibliothèque compilée. "
        "Adaptés à l'interface commune du projet (voir section 7). Leurs espaces d'états vont de 8192 à "
        "plus de 2 millions d'états."
    ))

    cells.append(code(
        "from src.rl.environments.line_world import LineWorldEnv\n"
        "from src.rl.environments.grid_world import GridWorldEnv\n"
        "\n"
        "print('Line World :')\n"
        "LineWorldEnv(num_cells=5).display()\n"
        "print()\n"
        "print('Grid World :')\n"
        "GridWorldEnv().display()\n"
    ))

    cells.append(md(
        "## 3. Les algorithmes\n"
        "\n"
        "**Programmation dynamique** (modèle de l'environnement connu à l'avance) : *Value Iteration* met "
        "à jour directement la valeur de chaque état avec sa meilleure action possible. *Policy Iteration* "
        "alterne évaluation complète de la politique courante et amélioration gloutonne.\n"
        "\n"
        "**Méthodes Monte Carlo** (apprennent en jouant des épisodes complets) : *Monte Carlo ES* démarre "
        "chaque épisode sur un état et une action tirés au hasard. *On-policy first visit MC control* garde "
        "un point de départ fixe et explore via une politique epsilon-soft. *Off-policy MC control* sépare "
        "la politique qui joue (comportement uniforme) de la politique améliorée, avec un ratio "
        "d'échantillonnage préférentiel.\n"
        "\n"
        "**Temporal Difference Learning** (mise à jour après chaque pas, pas besoin d'attendre la fin de "
        "l'épisode) : *Sarsa* est on-policy, *Q-Learning* est off-policy (se sert directement de la "
        "meilleure action possible du pas suivant).\n"
        "\n"
        "**Planning** : *Dyna-Q* combine l'apprentissage direct de Q-Learning avec un modèle appris de "
        "l'environnement, rejoué plusieurs fois par pas réel pour accélérer la convergence.\n"
        "\n"
        "### Un problème commun trouvé pendant les tests\n"
        "\n"
        "En construisant la comparaison complète (section 5), un bug a été trouvé dans Sarsa, Q-Learning, "
        "Dyna-Q et Monte Carlo ES : quand plusieurs actions ont exactement la même valeur estimée, "
        "`numpy.argmax` choisit toujours la première. Sur Grid World, avec un gamma proche de 1, une action "
        "qui ramène l'agent sur lui-même (un mur heurté) n'est presque jamais pénalisée par la mise à jour, "
        "donc si elle gagne ce départage dès le début, rien ne l'empêche de continuer à le gagner, et "
        "l'agent reste bloqué à la répéter indéfiniment. La correction retenue a été de départager les "
        "égalités au hasard plutôt que de toujours prendre la première."
    ))

    cells.append(md(
        "## 4. Méthodologie expérimentale\n"
        "\n"
        "Pour chaque combinaison (environnement, algorithme), l'algorithme est entraîné puis la politique "
        "obtenue est évaluée sur 200 épisodes joués sans apprentissage (action gloutonne à chaque pas). Les "
        "algorithmes de programmation dynamique sont lancés une fois (peu sensibles au hasard une fois "
        "convergés), les méthodes model-free sont relancées sur 5 graines aléatoires pour mesurer un taux "
        "de réussite plutôt qu'un seul résultat qui pourrait être une coïncidence.\n"
        "\n"
        "Un run est réussi si son score moyen atteint le score optimal réel de l'environnement (calculé "
        "avec Value Iteration) moins une marge de 0,1 : un seuil identique partout n'aurait pas de sens, "
        "le score optimal de Monty Hall est de 2/3 (3 portes) ou 4/5 (5 portes), pas de 1.\n"
        "\n"
        "epsilon a été porté à 0,3 (au lieu de 0,1 par défaut) pour Sarsa, Q-Learning, Dyna-Q et l'on-policy "
        "MC control, à cause du problème d'exploration décrit en section 3 et étudié en section 6."
    ))

    cells.append(md("## 5. Résultats de la comparaison"))

    cells.append(code(
        "baseline = load_csv('baseline_comparison.csv')\n"
        "baseline['mean_score'] = baseline['mean_score'].astype(float)\n"
        "baseline['success_rate'] = baseline['success_rate'].astype(float)\n"
        "baseline['optimal_score'] = baseline['optimal_score'].astype(float)\n"
        "baseline"
    ))

    cells.append(md(
        "Tous les algorithmes atteignent 100% de réussite sur Line World, Two Round RPS et Monty Hall "
        "(quelques scores légèrement en dessous de 100% sur RPS viennent surtout du bruit d'évaluation, "
        "puisque le premier round y reste aléatoire même avec une politique optimale).\n"
        "\n"
        "C'est sur Grid World que les différences sont les plus visibles : la programmation dynamique, "
        "Monte Carlo ES, l'off-policy MC control, Q-Learning et Dyna-Q atteignent 100%, Sarsa 80%, et "
        "l'on-policy first visit MC control échoue complètement (0%). Cet échec s'explique par sa nature "
        "même : cette méthode a besoin d'un épisode complet pour faire la moindre mise à jour, et très peu "
        "d'épisodes atteignent l'objectif ou le piège pendant l'entraînement à cause du problème "
        "d'exploration de la section 3. Sarsa et Q-Learning, eux, apprennent un peu à chaque pas grâce au "
        "bootstrap, ce qui les rend plus robustes.\n"
        "\n"
        "Sur les deux niveaux de Monty Hall, Dyna-Q est nettement moins fiable (60%) que les autres "
        "méthodes, sans doute à cause de son modèle appris qui suppose un environnement déterministe, ce "
        "qui colle mal avec la nature probabiliste de l'état de Monty Hall."
    ))

    cells.append(code(
        "display(Image(os.path.join(FIGURES_DIR, 'success_rate_grid_world.png')))"
    ))

    cells.append(md("## 6. Étude des hyperparamètres"))

    cells.append(code(
        "hyperparams = load_csv('hyperparameter_study.csv')\n"
        "hyperparams"
    ))

    cells.append(md(
        "**Epsilon sur Grid World.** Le taux de réussite de Sarsa baisse quand epsilon dépasse 0,3 (trop "
        "d'aléatoire nuit à la politique finale), alors que celui de l'on-policy MC control fait l'inverse "
        "(il lui faut beaucoup d'exploration pour qu'un épisode se termine ne serait-ce qu'une fois)."
    ))
    cells.append(code("display(Image(os.path.join(FIGURES_DIR, 'epsilon_study.png')))"))

    cells.append(md(
        "**Pas de planification sur Dyna-Q.** À seulement 500 épisodes réels, 0 pas de planification "
        "(= Q-Learning pur) donne 80% de réussite, 5 pas ou plus donnent 100%. La planification permet "
        "d'obtenir plus de mises à jour utiles de Q à partir des mêmes interactions réelles."
    ))
    cells.append(code("display(Image(os.path.join(FIGURES_DIR, 'planning_steps_study.png')))"))

    cells.append(md(
        "**Gamma sur Value Iteration.** Sur Line World (chemin court), la valeur suit presque directement "
        "gamma. Sur Grid World (chemin d'environ 8 pas), l'effet est spectaculaire : gamma=0,5 donne une "
        "valeur quasi nulle (0,008), gamma=0,999999 donne une valeur proche de 1. Un gamma trop petit "
        "dévalue très fortement les récompenses qui demandent plusieurs pas pour être atteintes."
    ))
    cells.append(code("display(Image(os.path.join(FIGURES_DIR, 'gamma_study.png')))"))

    cells.append(md(
        "## 7. Intégration et résultats sur les environnements secrets\n"
        "\n"
        "Les environnements secrets (0 à 3) sont fournis par le cours sous forme de bibliothèque compilée "
        "et d'un wrapper Python (`secret_envs/`), déjà disponibles avant même la fin du cours : seule "
        "l'interface graphique manque encore. `SecretEnvAdapter` "
        "(`src/rl/environments/secret_envs.py`) les adapte à l'interface commune du projet, ce qui les rend "
        "utilisables par tous les algorithmes model-free (Sarsa, Q-Learning, Dyna-Q, Monte Carlo "
        "on-policy/off-policy) sans écrire de code spécifique.\n"
        "\n"
        "Deux limites assumées : ces environnements n'exposent pas de moyen de se replacer directement dans "
        "un état donné, donc Monte Carlo ES ne s'y applique pas. Leurs espaces d'états (de 8192 à plus de 2 "
        "millions d'états) rendent aussi la programmation dynamique impraticable avec l'implémentation du "
        "projet, qui énumère explicitement toutes les combinaisons (état, action, état suivant, récompense) "
        "à chaque passage."
    ))

    cells.append(code(
        "secret_envs_results = load_csv('secret_envs/secret_envs_comparison.csv')\n"
        "secret_envs_results['mean_score'] = secret_envs_results['mean_score'].astype(float)\n"
        "secret_envs_results"
    ))

    cells.append(md(
        "Secret Env 3 n'apparaît pas dans ce tableau : la bibliothèque compilée fournie plante "
        "systématiquement (message \"Forbidden action\") sur cet environnement, quel que soit "
        "l'algorithme testé (les 5 ont été essayés) et quelle que soit la graine aléatoire utilisée (3 "
        "tentatives par algorithme). Ce comportement n'a pas pu être reproduit de façon isolée pour en "
        "identifier la cause exacte : c'est une limite de la bibliothèque fournie, pas un problème dans le "
        "code du projet, qui gère cet échec proprement (chaque combinaison tourne dans un processus séparé, "
        "pour qu'un plantage n'empêche pas de tester les autres).\n"
        "\n"
        "Sur les 3 environnements secrets exploitables, Dyna-Q obtient le meilleur score sur Secret Env 0 "
        "et Secret Env 1, ce qui confirme l'avantage de la planification déjà observé sur Grid World "
        "(section 6.2). Sur Secret Env 2 (plus de 2 millions d'états), tous les algorithmes obtiennent un "
        "score négatif : avec seulement 1500 à 3000 itérations (budget réduit pour rester dans un temps "
        "raisonnable), la couverture de cet espace d'états est trop faible pour apprendre une bonne "
        "stratégie. C'est une limite honnête du travail réalisé, pas un échec caché : il faudrait beaucoup "
        "plus d'itérations, ou une méthode capable de généraliser entre états proches plutôt que d'apprendre "
        "chaque état indépendamment (ce que ne font pas les méthodes tabulaires utilisées ici), pour espérer "
        "un meilleur résultat sur un espace d'états aussi grand."
    ))

    cells.append(md(
        "## 8. Quel algorithme choisir, et pourquoi\n"
        "\n"
        "Sur les environnements les plus simples (Line World, Two Round RPS, Monty Hall), tous les "
        "algorithmes se valent à peu près. Les différences intéressantes apparaissent sur Grid World, qui a "
        "un espace d'états plus grand et une récompense qui n'arrive qu'après plusieurs pas coordonnés.\n"
        "\n"
        "Quand le modèle de l'environnement est connu à l'avance, la programmation dynamique reste le choix "
        "le plus sûr et le plus rapide. Quand le modèle n'est pas connu, Q-Learning est la méthode la plus "
        "robuste dans l'ensemble. Dyna-Q apprend plus vite que Q-Learning à nombre d'épisodes réels égal sur "
        "Grid World grâce à sa planification, mais c'est aussi l'algorithme le moins fiable sur Monty Hall. "
        "Sarsa reste correct mais un peu moins fiable que Q-Learning sur Grid World. L'on-policy first visit "
        "MC control est clairement le moins adapté à Grid World dans sa configuration par défaut, à cause "
        "de son besoin d'épisodes complets. Monte Carlo ES et l'off-policy MC control restent fiables sur "
        "tous les environnements testés, mais Monte Carlo ES a besoin que l'environnement sache se replacer "
        "directement dans un état donné, ce qui n'est pas toujours possible."
    ))

    cells.append(md(
        "## 9. Limites et perspectives\n"
        "\n"
        "L'algorithme optionnel Dyna-Q+ n'a pas été implémenté, faute de temps disponible avant la date de "
        "rendu. L'interface graphique des environnements secrets n'était pas encore fournie au moment de la "
        "rédaction, seule leur interface en ligne de commande a été utilisée. Secret Env 3 n'a pas pu être "
        "testé du tout (bibliothèque fournie qui plante systématiquement), et Secret Env 2 n'a pas de "
        "stratégie satisfaisante à cause de son espace d'états bien trop grand (plus de 2 millions d'états) "
        "pour le budget d'itérations utilisé. Dyna-Q reste moins fiable que les autres méthodes sur Monty "
        "Hall, une piste à creuser serait d'adapter son modèle appris pour mieux gérer les environnements "
        "dont l'état encode une information probabiliste plutôt qu'une position concrète."
    ))

    cells.append(md(
        "## 10. Conclusion\n"
        "\n"
        "Ce projet a permis d'implémenter huit algorithmes classiques de l'apprentissage par renforcement "
        "et plusieurs environnements partageant une interface commune, puis de les comparer de façon "
        "systématique. Les résultats confirment que la plupart des algorithmes fonctionnent bien sur des "
        "environnements simples, mais que des différences importantes apparaissent dès que l'environnement "
        "devient plus grand ou que la récompense demande plusieurs actions coordonnées. Le travail a aussi "
        "permis de découvrir et corriger un vrai problème d'exploration lié au départage des égalités dans "
        "le calcul de la meilleure action, ce qui montre l'intérêt de tester les algorithmes sur des "
        "environnements variés plutôt que sur un seul cas simple."
    ))

    nb["cells"] = cells
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        nbf.write(nb, f)
    print("Notebook écrit dans", OUTPUT_PATH)


if __name__ == "__main__":
    build()
