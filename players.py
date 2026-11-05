import math
import random


class GreedyPlayer:
    def __init__(self, n, eps):
        self.n = n
        self.eps = eps
        self.action_values = [0.0] * n  # Q
        self.eval_count = [0] * n       # N

    def get_action(self):
        explore = random.random() < self.eps
        if explore:
            return self._random_action()
        return self._greedy_action()

    def _greedy_action(self):
        best = max(self.action_values)
        best_actions = [a for a in range(self.n) if self.action_values[a] == best]
        return random.choice(best_actions)

    def _random_action(self):
        return random.randrange(self.n)

    def reward(self, action, value):
        self.eval_count[action] += 1
        q = self.action_values[action]
        self.action_values[action] = q + (value - q) / self.eval_count[action]


class OptimistGreedyPlayer(GreedyPlayer):
    def __init__(self, n, eps=0, q0=5):
        super().__init__(n, eps)
        self.action_values = [float(q0)] * n


class UCBGreedyPlayer(GreedyPlayer):
    def __init__(self, n, c=2, eps=0):
        super().__init__(n, eps)
        self.c = c
        self.t = 0

    def reward(self, action, value):
        self.t += 1
        super().reward(action, value)

    def _greedy_action(self):
        # une action jamais jouée est considérée comme maximale
        untried = [a for a in range(self.n) if self.eval_count[a] == 0]
        if untried:
            return random.choice(untried)

        log_t = math.log(self.t)
        scores = [
            self.action_values[a] + self.c * math.sqrt(log_t / self.eval_count[a])
            for a in range(self.n)
        ]
        best = max(scores)
        return random.choice([a for a in range(self.n) if scores[a] == best])
