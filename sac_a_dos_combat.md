## Equipe : 
- Pan Lucie
- Huang Sylvain

---

## 1. Modélisation du Problème : "Le Sac à Dos de Combat"

Le projet s'inscrit dans la catégorie du **problème du sac à dos multidimensionnel** avec des contraintes de placement spatial et de dépendances.

* **Variables de décision :**
  * **Achat :** Quels objets (`Items` ou `Container`) acheter parmi la sélection aléatoire du magasin avec un budget limité (pièces).
  * **Placement :** Coordonnées et rotation de chaque objet dans la grille du sac.
  * **Économie :** Décision de garder ou dépenser l'or pour le tour suivant.

* **Fonction Objectif :** Maximiser la "Puissance de Combat" totale. Elle ne se calcule pas juste par la somme des valeurs, mais par la somme des valeurs individuelles **plus** les bonus de synergie (ex: une épée placée à côté d'une pierre à aiguiser gagne +2 dégâts).


* **Contraintes :** Espace limité (`Container` puis la grille `GRID_SIZE`), forme géométrique des objets (polyominos), et budget financier restreint.

---

## 2. Métaheuristique : Recuit Simulé

Voici comment **recuit simulé** va fonctionner dans notre contexte :

* **Principe :** L'algorithme explore différentes configurations de sacs (achats et placements). Au début, il accepte des configurations moins performantes pour éviter de rester bloqué dans un optimum local, puis il se stabilise vers la meilleure solution.


A chaque itération, l'une des deux actions suivantes :
* **Action A : Remplacement (Mutation de contenu)**:
  
  On retire un objet du sac pour en acheter un nouveau dans le magasin.

  Utilité : Permet de tester de nouvelles synergies et de s'adapter au budget.
* **Action B : Déplacement (Optimisation spatiale)**:
  
  On change les coordonnées $(x, y)$ ou la rotation d'un objet déjà présent dans la grille.

  Utilité : Permet de libérer de la place pour de futurs objets ou de coller deux objets qui ont un bonus de proximité (ex: épée + pierre à aiguiser)

---

## 3. Apport du Machine Learning : Réseaux de Neurones

L'objectif est d'utiliser un **Réseau de Neurones** pour "intelligence" le recuit simulé.

* **Rôle du ML (Guidage de la recherche) :** Au lieu de tester des placements aléatoires, le réseau de neurones peut **prédire la qualité d'une configuration** avant même de lancer le combat.


* **Entraînement :** Générer des milliers de combinaisons d'objets et leur attribuer un score de victoire. Le réseau apprend à reconnaître que "Objet A + Objet B = Forte puissance".


* **Hybridation :** Dans le recuit simulé, au moment de choisir un "prochain mouvement", on interroge le réseau de neurones. Si le réseau dit "cette combinaison d'objets a un fort potentiel", le recuit simulé explore cette zone en priorité.

