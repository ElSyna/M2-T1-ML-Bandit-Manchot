import numpy as np

from bandit import Ban10


def simulate(players, runs=2000, steps=1000):
    """Fait jouer chaque type de joueur sur les mêmes `runs` Ban10, étape par étape.

    players : dict nom -> fonction qui crée un joueur
    Renvoie un dict nom -> (récompense moyenne, % de choix optimal) par étape.
    """
    bans = [Ban10() for _ in range(runs)]
    results = {}
    for name, make_player in players.items():
        instances = [make_player() for _ in range(runs)]
        rewards = np.zeros(steps)
        optimal = np.zeros(steps)
        for t in range(steps):
            for ban, player in zip(bans, instances):
                action = player.get_action()
                value = ban.play(action)
                player.reward(action, value)
                rewards[t] += value
                optimal[t] += action == ban.best_action
        results[name] = (rewards / runs, 100 * optimal / runs)
    return results
