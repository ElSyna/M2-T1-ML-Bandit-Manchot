import copy
import math
import random

import numpy as np

from bandit import Bandit, Ban10


class MovingBandit(Bandit):

    def __init__(self):
        super().__init__()                    # tire un premier avg (inutilisé ensuite)
        self.period = random.uniform(100, 500)
        a, b = random.gauss(0, 1), random.gauss(0, 1)
        self.low, self.high = min(a, b), max(a, b)
        self.phase = random.uniform(0, 2 * math.pi)
        self.t = 0
        self._update_avg()

    def _update_avg(self):
        center = (self.low + self.high) / 2
        amplitude = (self.high - self.low) / 2
        self.avg = center + amplitude * math.sin(2 * math.pi * self.t / self.period + self.phase)

    def play(self):
        self.t += 1                           # la moyenne change à chaque coup
        self._update_avg()
        return super().play()                 # gauss(self.avg, 1), avec la nouvelle moyenne


class MovingBan10(Ban10):
    def __init__(self):
        self.bandits = [MovingBandit() for _ in range(10)]

    @property
    def best_action(self):
        avgs = [b.avg for b in self.bandits]
        return avgs.index(max(avgs))


def simulate_moving(players, runs=2000, steps=1000, make_ban=MovingBan10):
    pristine = [make_ban() for _ in range(runs)]
    results = {}
    for name, make_player in players.items():
        bans = copy.deepcopy(pristine)
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
