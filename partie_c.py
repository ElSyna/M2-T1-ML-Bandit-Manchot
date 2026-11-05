import os
import random
import sys

import matplotlib.pyplot as plt
import numpy as np

from bandit import Ban10
from players import GreedyPlayer, UCBGreedyPlayer
from simulation import simulate

SEED = 42
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#8a5cc2"]
plt.rcParams.update({
    "figure.dpi": 110, "savefig.dpi": 150,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": "#e5e5e5", "lines.linewidth": 1.6,
})


def play_one_run(player, ban, steps=1000, verbose=True):
    rewards, optimal = [], []
    for t in range(1, steps + 1):
        action = player.get_action()          # le joueur choisit un levier
        value = ban.play(action)              # le Ban10 renvoie la récompense
        player.reward(action, value)          # le joueur met à jour Q et N
        is_optimal = action == ban.best_action
        rewards.append(value)
        optimal.append(is_optimal)
        if verbose and (t <= 10 or t > steps - 5):
            print(f"t = {t:4d} | action = {action} | récompense = {value:+.3f}"
                  f" | optimal = {is_optimal}")
        elif verbose and t == 11:
            print("   ...")
    return np.array(rewards), np.array(optimal)


def ex3_test():
    random.seed(SEED)
    ban = Ban10()
    player = GreedyPlayer(10, eps=0.1)
    print("meilleur bras :", ban.best_action,
          "| vraies moyennes :", [round(b.avg, 2) for b in ban.bandits])
    rewards, optimal = play_one_run(player, ban)
    print(f"\nrécompense moyenne = {rewards.mean():.3f}"
          f" | choix optimaux = {optimal.mean():.1%}")
    print("N =", player.eval_count)
    print("Q =", [round(q, 2) for q in player.action_values])

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    steps = np.arange(1, len(rewards) + 1)
    ax1.plot(steps, rewards, color=COLORS[0], alpha=0.6, linewidth=0.8)
    ax1.axhline(max(b.avg for b in ban.bandits), color="#555555",
                linestyle="--", linewidth=1, label="moyenne du meilleur bras")
    ax1.set_ylabel("Récompense")
    ax1.set_title("Ex 3.11 - Un GreedyPlayer (ε = 0.1) sur un Ban10, 1000 itérations")
    ax1.legend(frameon=False, loc="lower right")
    ax2.plot(steps, 100 * np.cumsum(optimal) / steps, color=COLORS[1])
    ax2.set_ylabel("% cumulé de choix optimaux")
    ax2.set_xlabel("Itération")
    ax2.set_ylim(0, 100)
    fig.tight_layout()
    fig.savefig("figures/ex3_11_une_partie.png")
    plt.close(fig)


def plot_comparison(results, title, path):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 7), sharex=True)
    for i, (name, (rewards, optimal)) in enumerate(results.items()):
        steps = np.arange(1, len(rewards) + 1)
        ax1.plot(steps, rewards, color=COLORS[i], label=name)
        ax2.plot(steps, optimal, color=COLORS[i], label=name)
    ax1.set_ylabel("Récompense moyenne")
    ax1.set_title(title)
    ax1.legend(frameon=False, loc="lower right")
    ax2.set_ylabel("% de choix optimal")
    ax2.set_xlabel("Étape")
    ax2.set_ylim(0, 100)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def ex6_compare(runs=2000, steps=1000):
    random.seed(SEED)
    results = simulate({
        "UCB, c = 2": lambda: UCBGreedyPlayer(10, c=2),
        "ε-greedy, ε = 0.1": lambda: GreedyPlayer(10, eps=0.1),
    }, runs=runs, steps=steps)
    plot_comparison(results, f"Ex 6.3 - UCB vs ε-greedy ({runs} runs)",
                    "figures/ex6_3_ucb_vs_greedy.png")
    for name, (rewards, optimal) in results.items():
        print(f"{name:20s} | début (étapes 1-10) : récomp. {rewards[:10].mean():.3f},"
              f" optimal {optimal[:10].mean():.1f} %"
              f" | fin (200 dernières) : récomp. {rewards[-200:].mean():.3f},"
              f" optimal {optimal[-200:].mean():.1f} %")
    return results


if __name__ == "__main__":
    os.makedirs("figures", exist_ok=True)
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    if what in ("ex3", "all"):
        ex3_test()
    if what in ("ex6", "all"):
        ex6_compare()
