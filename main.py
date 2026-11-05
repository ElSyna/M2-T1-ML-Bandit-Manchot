from __future__ import annotations

import random
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from bandit import Bandit, Ban10, MovingBan10
from players import GreedyPlayer, OptimistGreedyPlayer, UCBGreedyPlayer

OUTPUT_DIR = Path(__file__).resolve().parent / "figures"
OUTPUT_DIR.mkdir(exist_ok=True)


def plot_ex1_single_bandit():
    random.seed(0)
    np.random.seed(0)
    samples = [Bandit().play() for _ in range(1000)]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(samples, bins=30, density=True, alpha=0.8, color="steelblue")
    ax.set_title("Exercice 1.3 : Distribution d'un bandit")
    ax.set_xlabel("Récompense")
    ax.set_ylabel("Densité")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "ex1_3_un_bandit.png", dpi=200)
    plt.close(fig)


def plot_ex1_three_bandits():
    random.seed(1)
    np.random.seed(1)
    means = [0.0, 1.0, 2.0]
    fig, ax = plt.subplots(figsize=(8, 5))
    for mean, color in zip(means, ["#1f77b4", "#ff7f0e", "#2ca02c"]):
        samples = [Bandit(avg=mean).play() for _ in range(1000)]
        ax.hist(samples, bins=30, density=True, alpha=0.5, color=color, label=f"moyenne {mean}")
    ax.set_title("Exercice 1.4 : 3 bandits avec différentes espérances")
    ax.set_xlabel("Récompense")
    ax.set_ylabel("Densité")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "ex1_4_trois_bandits.png", dpi=200)
    plt.close(fig)


def simulate_ex5_comparison(horizon=1000, runs=2000):
    bans = [Ban10() for _ in range(runs)]
    greedy = [GreedyPlayer(10, 0.1) for _ in range(runs)]
    optimistic = [OptimistGreedyPlayer(10, 0.1, q0=5) for _ in range(runs)]

    greedy_rewards = np.zeros(horizon)
    greedy_optimal = np.zeros(horizon)
    optimistic_rewards = np.zeros(horizon)
    optimistic_optimal = np.zeros(horizon)

    for t in range(horizon):
        for ban, g, o in zip(bans, greedy, optimistic):
            a_g = g.get_action()
            r_g = ban.play(a_g)
            g.reward(a_g, r_g)
            greedy_rewards[t] += r_g
            greedy_optimal[t] += (a_g == ban.best_action)

            a_o = o.get_action()
            r_o = ban.play(a_o)
            o.reward(a_o, r_o)
            optimistic_rewards[t] += r_o
            optimistic_optimal[t] += (a_o == ban.best_action)

    return (
        greedy_rewards / runs,
        greedy_optimal / runs,
        optimistic_rewards / runs,
        optimistic_optimal / runs,
    )


def simulate_epsilon_experiment(horizon=1000, runs=2000):
    results = {}
    for eps in [0.0, 0.01, 0.1]:
        bans = [Ban10() for _ in range(runs)]
        players = [GreedyPlayer(10, eps) for _ in range(runs)]
        rewards = np.zeros(horizon)
        optimal = np.zeros(horizon)
        for t in range(horizon):
            for ban, player in zip(bans, players):
                action = player.get_action()
                reward = ban.play(action)
                player.reward(action, reward)
                rewards[t] += reward
                optimal[t] += (action == ban.best_action)
        results[eps] = (rewards / runs, optimal / runs)
    return results


def simulate_moving_bandit_experiment(horizon=1000, runs=2000):
    results = {}
    for eps in [0.0, 0.01, 0.1]:
        envs = [MovingBan10() for _ in range(runs)]
        players = [GreedyPlayer(10, eps) for _ in range(runs)]
        rewards = np.zeros(horizon)
        optimal = np.zeros(horizon)
        for t in range(horizon):
            for env, player in zip(envs, players):
                action = player.get_action()
                reward = env.play(action)
                player.reward(action, reward)
                rewards[t] += reward
                optimal[t] += (action == env.best_action)
        results[eps] = (rewards / runs, optimal / runs)
    return results


def plot_ex4_single_player_reward():
    env = Ban10()
    player = GreedyPlayer(10, 0.1)
    rewards = []

    for _ in range(1000):
        action = player.get_action()
        reward = env.play(action)
        player.reward(action, reward)
        rewards.append(reward)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(range(1, 1001), rewards, linewidth=0.8)
    ax.set_title("Exercice 4.1 : Récompense d'un GreedyPlayer sur 1000 étapes")
    ax.set_xlabel("Itération")
    ax.set_ylabel("Récompense")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "ex4_1_recompense_joueur.png", dpi=200)
    plt.close(fig)


def plot_ex4_epsilon_curves():
    results = simulate_epsilon_experiment()
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for eps, (rewards, optimal) in results.items():
        axes[0].plot(rewards, label=f"ε = {eps}")
        axes[1].plot(optimal, label=f"ε = {eps}")
    axes[0].set_title("Exercice 4.3 : Récompense moyenne")
    axes[0].set_xlabel("Étape")
    axes[0].set_ylabel("Récompense")
    axes[0].legend()
    axes[1].set_title("Exercice 4.4 : Proportion de choix optimaux")
    axes[1].set_xlabel("Étape")
    axes[1].set_ylabel("Pourcentage")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "ex4_epsilon_curves.png", dpi=200)
    plt.close(fig)


def plot_ex5_optimist_vs_greedy():
    (g_rewards, g_opt, o_rewards, o_opt) = simulate_ex5_comparison()

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].plot(g_rewards, label="Epsilon-greedy (ε=0.1)", alpha=0.8)
    axes[0].plot(o_rewards, label="Optimistic greedy (Q0=5)", alpha=0.8)
    axes[0].set_title("Exercice 5.3 : Récompense moyenne")
    axes[0].set_xlabel("Étape")
    axes[0].set_ylabel("Récompense")
    axes[0].legend()

    axes[1].plot(g_opt, label="Epsilon-greedy (ε=0.1)", alpha=0.8)
    axes[1].plot(o_opt, label="Optimistic greedy (Q0=5)", alpha=0.8)
    axes[1].set_title("Exercice 5.3 : Proportion de choix optimaux")
    axes[1].set_xlabel("Étape")
    axes[1].set_ylabel("Pourcentage")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "ex5_3_optimiste_vs_greedy.png", dpi=200)
    plt.close(fig)


def plot_ex7_moving_bandit_curves():
    results = simulate_moving_bandit_experiment()
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for eps, (rewards, optimal) in results.items():
        axes[0].plot(rewards, label=f"ε = {eps}")
        axes[1].plot(optimal, label=f"ε = {eps}")
    axes[0].set_title("Exercice 7.3 : Récompense moyenne (environnement non stationnaire)")
    axes[0].set_xlabel("Étape")
    axes[0].set_ylabel("Récompense")
    axes[0].legend()
    axes[1].set_title("Exercice 7.4 : Proportion de choix optimaux")
    axes[1].set_xlabel("Étape")
    axes[1].set_ylabel("Pourcentage")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "ex7_moving_bandit.png", dpi=200)
    plt.close(fig)


def main():
    random.seed(0)
    np.random.seed(0)
    plot_ex1_single_bandit()
    plot_ex1_three_bandits()
    plot_ex4_single_player_reward()
    plot_ex4_epsilon_curves()
    plot_ex5_optimist_vs_greedy()
    plot_ex7_moving_bandit_curves()
    print(f"Figures generated in {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
