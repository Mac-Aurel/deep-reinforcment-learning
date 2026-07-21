import csv
import os

import matplotlib.pyplot as plt

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
FIGURES_DIR = os.path.join(os.path.dirname(__file__), "..", "report", "figures")

ALGO_LABELS = {
    "policy_iteration": "Policy Iteration",
    "value_iteration": "Value Iteration",
    "monte_carlo_es": "Monte Carlo ES",
    "on_policy_first_visit_mc": "On-policy MC",
    "off_policy_mc": "Off-policy MC",
    "sarsa": "Sarsa",
    "q_learning": "Q-Learning",
    "dyna_q": "Dyna-Q",
}

ENV_LABELS = {
    "line_world": "Line World",
    "grid_world": "Grid World",
    "two_round_rps": "Two Round RPS",
    "monty_hall_3_doors": "Monty Hall (3 portes)",
    "monty_hall_5_doors": "Monty Hall (5 portes)",
}


def read_csv(filename):
    with open(os.path.join(RESULTS_DIR, filename)) as f:
        return list(csv.DictReader(f))


def figure_success_rate_grid_world(rows):
    grid_rows = [r for r in rows if r["environment"] == "grid_world"]
    names = [ALGO_LABELS[r["algorithm"]] for r in grid_rows]
    values = [float(r["success_rate"]) * 100 for r in grid_rows]

    plt.figure(figsize=(7, 4))
    bars = plt.bar(names, values, color="#4C72B0")
    plt.ylabel("Taux de réussite (%)")
    plt.title("Taux de réussite de chaque algorithme sur Grid World")
    plt.xticks(rotation=35, ha="right")
    plt.ylim(0, 105)
    for bar, value in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width() / 2, value + 2, f"{value:.0f}%", ha="center", fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "success_rate_grid_world.png"), dpi=150)
    plt.close()


def figure_epsilon_study(rows):
    epsilon_rows = [r for r in rows if r["study"] == "epsilon_on_grid_world"]
    sarsa = [r for r in epsilon_rows if r["algorithm"] == "sarsa"]
    on_policy = [r for r in epsilon_rows if r["algorithm"] == "on_policy_first_visit_mc"]

    plt.figure(figsize=(6, 4))
    plt.plot(
        [float(r["epsilon"]) for r in sarsa],
        [float(r["success_rate"]) * 100 for r in sarsa],
        marker="o",
        label="Sarsa",
        color="#4C72B0",
    )
    plt.plot(
        [float(r["epsilon"]) for r in on_policy],
        [float(r["success_rate"]) * 100 for r in on_policy],
        marker="o",
        label="On-policy MC",
        color="#DD8452",
    )
    plt.xlabel("epsilon")
    plt.ylabel("Taux de réussite (%)")
    plt.title("Effet d'epsilon sur Grid World")
    plt.legend()
    plt.ylim(-5, 105)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "epsilon_study.png"), dpi=150)
    plt.close()


def figure_planning_steps_study(rows):
    planning_rows = [r for r in rows if r["study"] == "planning_steps_on_grid_world"]
    planning_rows.sort(key=lambda r: int(r["planning_steps"]))

    plt.figure(figsize=(6, 4))
    x = [r["planning_steps"] for r in planning_rows]
    y = [float(r["success_rate"]) * 100 for r in planning_rows]
    bars = plt.bar(x, y, color="#55A868")
    plt.xlabel("Nombre de pas de planification simulés")
    plt.ylabel("Taux de réussite (%)")
    plt.title("Effet de la planification sur Dyna-Q (500 épisodes réels)")
    plt.ylim(0, 105)
    for bar, value in zip(bars, y):
        plt.text(bar.get_x() + bar.get_width() / 2, value + 2, f"{value:.0f}%", ha="center", fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "planning_steps_study.png"), dpi=150)
    plt.close()


def figure_gamma_study(rows):
    gamma_rows = [r for r in rows if r["study"] == "gamma_on_value_iteration"]

    plt.figure(figsize=(6, 4))
    for env_name, color in [("line_world", "#4C72B0"), ("grid_world", "#DD8452")]:
        env_rows = [r for r in gamma_rows if r["environment"] == env_name]
        env_rows.sort(key=lambda r: float(r["gamma"]))
        plt.plot(
            [float(r["gamma"]) for r in env_rows],
            [float(r["start_state_value"]) for r in env_rows],
            marker="o",
            label=ENV_LABELS[env_name],
            color=color,
        )
    plt.xlabel("gamma")
    plt.ylabel("Valeur de l'état de départ")
    plt.title("Effet de gamma sur la valeur optimale (Value Iteration)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "gamma_study.png"), dpi=150)
    plt.close()


if __name__ == "__main__":
    os.makedirs(FIGURES_DIR, exist_ok=True)
    baseline_rows = read_csv("baseline_comparison.csv")
    hyperparameter_rows = read_csv("hyperparameter_study.csv")

    figure_success_rate_grid_world(baseline_rows)
    figure_epsilon_study(hyperparameter_rows)
    figure_planning_steps_study(hyperparameter_rows)
    figure_gamma_study(hyperparameter_rows)
    print("Figures written to", FIGURES_DIR)
