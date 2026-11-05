import random


class Bandit:
    def __init__(self):
        self.avg = random.gauss(0, 1)

    def play(self):
        return random.gauss(self.avg, 1)


class Ban10:
    def __init__(self):
        self.bandits = [Bandit() for _ in range(10)]
        avgs = [b.avg for b in self.bandits]
        self.best_action = avgs.index(max(avgs))

    def play(self, action):
        return self.bandits[action].play()
