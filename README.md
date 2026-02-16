# Sac a dos combat

[![License](https://img.shields.io/github/license/uciie/sac_a_doc_combat)](./LICENSE)
[![Version](https://img.shields.io/github/v/tag/uciie/sac_a_doc_combat)](https://github.com/uciie/sac_a_doc_combat/releases)

Projet Optimisation Combinatoire - Apport du Machine Learning pour l’amélioration des métaheuristiques

## 1. Modélisation du Problème : "Le Sac à Dos de Combat"

Le projet s'inscrit dans la catégorie du **problème du sac à dos multidimensionnel** avec des contraintes de placement spatial et de dépendances.

* **Variables de décision :**
  * **Achat :** Quels objets (`Items` ou `Container`) acheter parmi la sélection aléatoire du magasin (`Store`) avec un budget limité (pièces).
  * **Placement :** Coordonnées et rotation de chaque objet dans la grille du sac.
  * **Économie :** Décision de garder ou dépenser l'or pour le tour suivant.

* **Fonction Objectif :** Maximiser la "Puissance de Combat" totale. Elle ne se calcule pas juste par la somme des valeurs, mais par la somme des valeurs individuelles **plus** les bonus de synergie (ex: une épée placée à côté d'une pierre à aiguiser gagne +2 dégâts).


* **Contraintes :** Espace limité (`Container` puis la grille `GRID_SIZE`), forme géométrique des objets (polyominos), et budget financier restreint.

-----

## Équipe

| Membre | GitHub | Numéro Étudiant | 
|---------|---------|----|
| **Lucie Pan** | [@uciie](https://github.com/uciie) | 45004162 | ? |
| **Sylvain Huang** | [@Kusanagies](https://github.com/Kusanagies) | 41005688 |

-----

> Projet universitaire M1 MIAGE 2025-2026 — Université Paris Nanterre.