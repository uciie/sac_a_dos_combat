# Sac a dos combat

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.6.0-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.8.0-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Flask](https://img.shields.io/badge/Flask-3.1.3-000000?style=for-the-badge&logo=flask&logoColor=white)](https://flask.palletsprojects.com/)
[![License](https://img.shields.io/github/license/uciie/sac_a_dos_combat?style=for-the-badge)](./LICENSE)

Projet Optimisation Combinatoire - Apport du Machine Learning pour l'amélioration des métaheuristiques
<div align="center">

[Rapport PDF](./RapportOC.pdf)

</div>

---

## Table des matières
1. [Description du projet : "Le Sac à Dos de Combat"](#description-du-projet--le-sac-à-dos-de-combat)
2. [Modélisation du problème](#modélisation-du-problème)
3. [Architecture technique](#architecture-technique)
4. [Structure du dépôt](#structure-du-dépôt)
5. [Installation & Lancement](#installation--lancement)
6. [Visualisation du problème](#visualisation-du-problème)
7. [Équipe](#équipe)

---

## Description du projet : "Le Sac à Dos de Combat"

Le projet s'inscrit dans la catégorie du **problème du sac à dos multidimensionnel** avec des contraintes de placement spatial et de dépendances.

* **Variables de décision :**
  * **Achat :** Quels objets (`Items` ou `Container`) acheter parmi la sélection aléatoire du magasin (`Store`) avec un budget limité (pièces).
  * **Placement :** Coordonnées et rotation de chaque objet dans la grille du sac.
  * **Économie :** Décision de garder ou dépenser l'or pour le tour suivant.

* **Fonction Objectif :** Maximiser la "Puissance de Combat" totale. Elle ne se calcule pas juste par la somme des valeurs, mais par la somme des valeurs individuelles **plus** les bonus de synergie (ex: une épée placée à côté d'une pierre à aiguiser gagne +2 dégâts).


* **Contraintes :** Espace limité (`Container` puis la grille `GRID_SIZE`), forme géométrique des objets (polyominos), et budget financier restreint.
  
### Approche hybride : Métaheuristiques + Machine Learning

L'objectif principal est de concevoir et d'évaluer une **approche hybride** combinant :

| Métaheuristique | Surrogate Model (ML) | Rôle du ML |
|---|---|---|
| **Algorithme Génétique (AG)** | `MLPRegressor` (Scikit-learn) | Filtre rapide des candidats mutants |
| **Recuit Simulé (RS)** | `CNN` PyTorch (`BackpackScoreNet`) | Guidage de la sélection "Top-K Candidates" |

Le Machine Learning agit comme un **modèle de substitution** (*Surrogate Model*) : il remplace l'évaluation coûteuse de la fonction objectif réelle (calcul géométrique + synergies) par une **prédiction matricielle instantanée**, accélérant ainsi la convergence des algorithmes.

---

## Modélisation du problème

### Variables de décision

Le problème se définit par **trois niveaux de variables imbriquées** :

- **Décision d'achat** $x_i ∈ {0,1}$ : quels items acheter, sous contrainte $\sum prix_i \times x_i \leq Budget$.
- **Décision de placement** $(x_i, y_i, z_i) \in N^2 \times {0°, 90°, 180°, 270°}$ : position et rotation de chaque item acheté dans la grille.
- **Décision économique** : gestion dynamique du budget restant pour les achats futurs.

### Fonction objectif

La **puissance de combat totale** à maximiser se décompose en deux termes additifs :

$$f(S) = \sum_{i\in S} puissanceBase(i) + \sum_{(i,j) \in Adj(S)} bonusSynergie(tags(i), tags(j))$$

Le second terme est la somme des bonus de synergie pour chaque paire $(i, j)$ d'items adjacents orthogonalement dont les **tags sont compatibles** selon la `SYNERGY_TABLE`.

### Table des synergies

| Tags | Bonus | Effet |
|------|-------|-------|
| Arme + Feu | +2.2 | Lame enflammée |
| Feu + Soufre | +3.0 | Combustion spontanée |
| Magie + Cristal | +2.5 | Amplification cristalline |
| Arme + Poison | +1.5 | Lame empoisonnée |
| Bouclier + Armure | +1.8 | Défense renforcée |
| Arc + Plume | +2.0 | Flèches légères |
| Glace + Vent | +1.8 | Blizzard |
| Magie + Étoile | +2.0 | Magie stellaire |

-----

## Architecture technique

### Métaheuristiques

#### Algorithme Génétique (AG)

- **Population** de 20 individus (sacs à dos).
- **Opérateurs de mutation** : `ADD` (remplacement d'item), `MOVE` (déplacement spatial), `ROTATE` (rotation 90°).
- **Critère d'arrêt** : 50 générations.
- **Hybridation** : le réseau `MLPRegressor` prédit les scores de 100 candidats (20 individus \times 5 enfants) pour ne retenir que les 20 les plus prometteurs avant évaluation réelle.

#### Recuit Simulé (RS)

| Paramètre | Valeur | Rôle |
|-----------|--------|------|
| Température initiale T₀ | 1 000 | Amplitude de l'exploration initiale |
| Coefficient alpha | 0.995 | Vitesse de refroidissement par itération |
| Critère d'arrêt | T < 0.5 | Convergence quasi-déterministe |
| Itérations par appel API | 50 | Granularité des snapshots |
| Nombre de runs | 8 | Robustesse statistique |
| Nombre total de mouvements | 400 | Effort computationnel total |

Les quatre opérateurs de voisinage (`MOVE`, `ROTATE`, `ADD`, `REMOVE`) sont tirés aléatoirement à chaque itération. Le **critère de Metropolis** accepte un mouvement dégradant avec la probabilité `P = exp(Δf / T)`.

### Modèles de Machine Learning

#### MLPRegressor — Surrogate pour l'AG

- **Fichier** : `src/ml_surrogate.py`
- **Architecture** : 2 couches cachées (32 → 16 neurones).
- **Entrée** : vecteur de comptage d'items, extrait par `extract_features()`.
- **Avantage** : inférence en quelques microsecondes par prédiction.
- **Persistance** : sauvegarde/rechargement via `pickle` (`save()` / `load()`).

#### CNN PyTorch — Surrogate pour le RS

- **Fichier** : `src/nn_backpack.py`
- **Encodeur `GridEncoder`** : transforme l'état du sac en tenseur multi-canaux `(C, 10, 10)` — occupation des cases, puissance normalisée, zones containers, encodage one-hot des tags.
- **Architecture `BackpackScoreNet`** : 3 couches convolutives (32→64→64 filtres) + `AdaptiveAvgPool2d` + tête dense (256→64→1) + activation `Softplus` (score ≥ 0).
- **Stratégie "Top-K Candidates"** : génère k=5 états candidats, prédit leurs scores en un seul passage batch, sélectionne le meilleur avant d'appliquer Metropolis avec le score exact.

---

## Structure du dépôt

```
sac_a_dos_combat/
│
├── README.md                        # Ce fichier
├── RapportOC.pdf                    # Rapport complet du projet
├── requirements.txt                 # Dépendances Python (versions figées)
├── experiments.py                   # Script d'expérimentations rapides
│
└── src/
    │
    ├── app.py                       # Serveur Flask — API REST (8 endpoints)
    ├── index.html                   # Interface web de visualisation
    │
    ├── models.py                    # Moteur de jeu : BackpackManager, Item,
    │                                #   Container, Store, SimulatedAnnealing
    ├── models_ga.py                 # Algorithme Génétique (GeneticAlgorithmML)
    ├── ml_surrogate.py              # Surrogate MLP Scikit-learn pour l'AG
    ├── nn_backpack.py               # Surrogate CNN PyTorch pour le RS :
    │                                #   GridEncoder, BackpackScoreNet,
    │                                #   ExperienceBuffer, NNTrainer, NNGuidedSA
    │
    ├── benchmark_backpack.py        # Benchmark : SA classique vs SA + CNN
    ├── benchmark_ag.py              # Benchmark : AG classique vs AG + MLP
    ├── plot_benchmark.py            # Génération des graphiques RS (9 figures)
    ├── plot_benchmark_ga.py         # Génération des graphiques AG (4 figures)
    │
    ├── plots/                       # Graphiques générés 
    │
    └── out/                         # Résultats JSON et modèles entraînés
        ├── benchmark_results.json   # Résultats benchmark RS (SA vs SA+NN)
        ├── benchmark_ga_results.json# Résultats benchmark AG (AG vs AG+MLP)
        ├── backpack_nn.pt           # Modèle CNN PyTorch sauvegardé
        └── nn_surrogate.pkl         # Modèle MLP Scikit-learn sauvegardé
```
----

## Installation & Lancement

### Prérequis

- Python **3.10+**
- `pip`

### 1. Cloner le dépôt

```bash
git clone https://github.com/uciie/sac_a_dos_combat.git
cd sac_a_dos_combat
```

### 2. Instructions d'installation de l'environnement
```bash
python3 -m venv venv
# Mac/ Linux 
source venv/bin/activate
# Windows 
venv\Scripts\activate
```

### 3. Installer les dépendances

```bash
pip install -r requirements.txt
```

Les dépendances principales incluent : `torch==2.6.0`, `scikit-learn==1.8.0`, `flask==3.1.3`, `numpy==2.4.4`, `matplotlib==3.10.8`.

-----

## Visualisation du problème
L'interface web permet de jouer manuellement au Sac à Dos de Combat et de visualiser les algorithmes en temps réel (grille, synergies actives, courbe d'évolution du score).

1. Exécuter le fichier `src/app.py`
```bash
cd src
python3 app.py
```
2. Ouvrir le fichier `src/index.html`

Le sélecteur d'algorithme propose trois modes : **Recuit Simulé classique**, **Algorithme Génétique sans ML**, et **Algorithme Génétique + Machine Learning**. 

/!\ **Recuit Simulé + CNN** n'est pas encore dans le choix de selection. Cependant il est fonctionnel.

---

### Benchmarks

#### Benchmark principal — Recuit Simulé classique vs RS + CNN

```bash
cd src
python benchmark_backpack.py
```

Ce script exécute le pipeline complet en trois phases :

1. Génération d'un dataset d'entraînement (150 configurations, ~450 exemples).
2. Entraînement du réseau `BackpackScoreNet` (30 époques max, early stopping).
3. Exécution de **8 runs** indépendants \times **400 itérations SA** pour chaque méthode.

Sorties produites : `out/benchmark_results.json` et `out/backpack_nn.pt`.

---

#### Benchmark AG — Algorithme Génétique classique vs AG + MLP

```bash
cd src
python benchmark_ag.py
```

Ce script exécute :

1. Génération de 5 000 exemples d'entraînement pour le `MLPRegressor`.
2. Entraînement du Surrogate MLP.
3. Exécution de **8 runs** \times **50 générations** pour chaque méthode.

Sortie produite : `out/benchmark_ga_results.json`.

---

### Génération des graphiques

#### Graphiques RS + CNN — 9 figures PNG

> Nécessite d'avoir exécuté `benchmark_backpack.py` au préalable.

```bash
cd src
python plot_benchmark.py
```

| Fichier généré | Contenu |
|----------------|---------|
| `00_synthese_2x2.png` | Panneau de synthèse (4 sous-graphiques) |
| `01_convergence_moyenne.png` | Courbes de convergence avec enveloppe ±1σ |
| `02_boxplot_scores.png` | Distribution des scores finaux (boxplot) |
| `03_scores_par_run.png` | Comparaison individuelle score par run |
| `04_temps_execution.png` | Temps d'exécution et corrélation SA vs SA+NN |
| `05_correlation_score_temps.png` | Nuage de points score vs durée |
| `06_courbe_perte_nn.png` | Courbe de loss + tableau de métriques CNN |
| `07_heatmap_gains.png` | Heatmap des gains delta score / delta temps |
| `08_radar_synthese.png` | Radar multi-critères normalisé [0,1] |

#### Graphiques AG + MLP — 4 figures PNG

> Nécessite d'avoir exécuté `benchmark_ag.py` au préalable.

```bash
cd src
python plot_benchmark_ga.py
```

Produit dans `src/plots/` : `ga_01_convergence.png`, `ga_02_boxplot_scores.png`, `ga_03_temps_execution.png`, `ga_04_correlation_score_temps.png`.

-----

## Équipe

| Membre | GitHub | Numéro Étudiant | 
|---------|---------|----|
| **Lucie Pan** | [@uciie](https://github.com/uciie) | 45004162 | ? |
| **Sylvain Huang** | [@Kusanagies](https://github.com/Kusanagies) | 41005688 |

-----

> Projet universitaire M1 MIAGE 2025-2026 — Université Paris Nanterre.