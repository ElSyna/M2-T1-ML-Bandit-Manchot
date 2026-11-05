import math
import random


class Bandit:
    """Bandit manchot simple avec moyenne gaussienne et bruit gaussien."""

    def __init__(self, avg=None):
        self.avg = random.gauss(0, 1) if avg is None else avg

    def play(self):
        return random.gauss(self.avg, 1)


class Ban10:
    """Environnement à 10 bras. Le meilleur bras est celui d'espérance maximale."""

    def __init__(self):
        self.bandits = [Bandit() for _ in range(10)]
        self.best_action = max(range(len(self.bandits)), key=lambda i: self.bandits[i].avg)

    def play(self, action):
        if not 0 <= action < len(self.bandits):
            raise ValueError("action must be in [0, 9]")
        return self.bandits[action].play()


class MovingBandit(Bandit):
    """Bandit non stationnaire dont la moyenne bouge selon un sinus."""

    def __init__(self, avg=None, period=None):
        base_avg = random.gauss(0, 1) if avg is None else avg
        super().__init__(avg=base_avg)
        self.base_avg = base_avg
        self.period = random.uniform(100, 500) if period is None else period
        self.amplitude = random.uniform(1.0, 3.0)
        self.phase = random.uniform(0, 2 * 3.141592653589793)
        self.step = 0

    def play(self):
        self.step += 1
        self.avg = self.base_avg + self.amplitude * math.sin(
            self.phase + 2 * math.pi * self.step / self.period
        )
        return random.gauss(self.avg, 1)


class MovingBan10(Ban10):
    """Ban10 non stationnaire : chaque bras est un MovingBandit."""

    def __init__(self):
        self.bandits = [MovingBandit() for _ in range(10)]
        self.best_action = max(range(len(self.bandits)), key=lambda i: self.bandits[i].avg)

    def play(self, action):
        if not 0 <= action < len(self.bandits):
            raise ValueError("action must be in [0, 9]")
        reward = self.bandits[action].play()
        self.best_action = max(range(len(self.bandits)), key=lambda i: self.bandits[i].avg)
        return reward
