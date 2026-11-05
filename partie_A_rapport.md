# Partie A — Bandit manchot

## 1. Implémentation

- `Bandit`: initialise une moyenne `avg` tirée selon une loi normale centrée en 0 et de variance 1 ; la fonction `play()` renvoie un échantillon gaussien autour de cette moyenne.
- `Ban10`: contient 10 bandits et garde la meilleure action (`best_action`) correspondant au bras d'espérance maximale.
- `GreedyPlayer`: stratégie epsilon-greedy avec mise à jour de `Q(a)` selon la formule de moyenne cumulative.
- `OptimistGreedyPlayer`: hérite de `GreedyPlayer` et initialise les valeurs `Q(a)` à 5, ce qui favorise l'exploration initiale.

## 2. Résultats observés

### Exercice 1

Les histogrammes obtenus montrent bien une loi normale autour de la moyenne du bandit. Avec trois bandits aux moyennes différentes, on observe trois pics distincts, ce qui confirme la bonne implémentation de 3 distributions différentes.

### Exercice 4

L'expérience avec ε = 0, 0.01 et 0.1 montre que le comportement initial est très bruité et que la stratégie la plus exploratrice (ε = 0.1) obtient une meilleure dynamique de découverte au début. À long terme, les petites valeurs de ε favorisent une meilleure exploitation du meilleur bras, tandis que ε = 0.1 reste plus fluctuant.

### Exercice 5

L'initialisation optimiste donne de meilleures performances au début : les agents explorent davantage parce que toutes les actions démarrent à une valeur élevée. Cela les pousse à tester rapidement les bras sous-explorés. Au bout d'un certain temps, les deux méthodes convergent vers des performances similaires, l'optimisme ne donne plus d'avantage net car les estimations sont déjà bien apprises.

### Exercice 7

Dans un environnement non stationnaire, la moyenne des bras évolue au fil du temps. Les algorithmes habituels à ε-greedy doivent continuellement réexplorer, ce qui explique pourquoi les courbes restent plus instables que dans le cas stationnaire. L'optimisme et UCB sont utiles car ils gardent une part de découverte même quand les meilleurs bras changent.

## 3. Interprétation de l'optimisme

- Comportement initial : meilleure exploration, plus grande probabilité de trouver le bon bras au début.
- Comportement asymptotique : le gain initial se réduit progressivement car les estimations deviennent fiables. Le comportement se stabilise vers l'exploitation du meilleur bras connu.

## 4. Fichiers à exécuter

```bash
python main.py
```

Les figures sont sauvegardées dans le dossier `figures/`.
