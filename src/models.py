"""
=====================================================================
    models.py - Moteur de jeu Backpack Battles                    
    M1 MIAGE - Optimisation Combinatoire                          
                                                            
    Classes :                                                     
      Container     -> Zone rectangulaire achetable sur la grille  
      Item          -> Objet à placer (forme 2D + tags + synergies)
      BackpackManager -> Logique de placement + fonction objectif f
      Store         -> Marché aléatoire + gestion du budget        
=====================================================================
"""

from __future__ import annotations
import numpy as np
import copy
import random
from dataclasses import dataclass, field
from typing import Optional


# ==============================================================================
# 1. CONSTANTES ET DONNÉES DE CATALOGUE
# ==============================================================================

GRID_SIZE: int = 10  # Grille globale 10*10

# Table de synergies : frozenset de deux tags -> bonus additif au score
# Activé si deux items avec ces tags sont adjacents (4-voisinage orthogonal)
SYNERGY_TABLE: dict[frozenset, float] = {
    frozenset({"Arme",    "Ail"     }): 2.0,  # Classique Backpack Battles
    frozenset({"Arme",    "Poison"  }): 1.5,  # Lame empoisonnée
    frozenset({"Bouclier","Armure"  }): 1.8,  # Défense renforcée
    frozenset({"Magie",   "Cristal" }): 2.5,  # Amplification cristalline
    frozenset({"Arc",     "Plume"   }): 2.0,  # Flèches légères
    frozenset({"Feu",     "Soufre"  }): 3.0,  # Combustion spontanée
    frozenset({"Glace",   "Vent"    }): 1.8,  # Blizzard
    frozenset({"Arme",    "Feu"     }): 2.2,  # Lame enflammée
    frozenset({"Poison",  "Herbe"   }): 1.5,  # Toxine botanique
    frozenset({"Magie",   "Étoile"  }): 2.0,  # Magie stellaire
    frozenset({"Bouclier","Pierre"  }): 1.2,  # Rempart de pierre
    frozenset({"Arc",     "Vent"    }): 1.6,  # Flèches guidées par le vent
    frozenset({"Arme",    "Arme"    }): 0.8,  # Combo double arme (faible)
    frozenset({"Magie",   "Poison"  }): 1.7,  # Magie toxique
}

# Catalogue d'items disponibles dans le magasin
ITEM_CATALOGUE: list[dict] = [
    # -- Armes --------------------------------------------------------------
    dict(nom="Épée Longue",       prix=25, puissance=12.0,
         tags=["Arme"],           forme=[[1,1,1]]),
    dict(nom="Dague du Chaos",    prix=15, puissance=7.0,
         tags=["Arme","Poison"],  forme=[[1,0],[1,1]]),
    dict(nom="Hache de Guerre",   prix=30, puissance=18.0,
         tags=["Arme"],           forme=[[0,1],[1,1],[1,0]]),
    dict(nom="Rapière Enflammée", prix=32, puissance=14.0,
         tags=["Arme","Feu"],     forme=[[1,1,0],[0,1,1]]),
    # -- Arcs --------------------------------------------------------------
    dict(nom="Arc Elfique",       prix=20, puissance=10.0,
         tags=["Arc","Vent"],     forme=[[0,1],[1,1],[0,1]]),
    dict(nom="Arc Long",          prix=22, puissance=11.0,
         tags=["Arc"],            forme=[[1],[1],[1],[1]]),
    dict(nom="Carquois de Plumes",prix=12, puissance=4.0,
         tags=["Arc","Plume"],    forme=[[1,1]]),
    # -- Magie --------------------------------------------------------------
    dict(nom="Orbe Cristallin",   prix=40, puissance=20.0,
         tags=["Magie","Cristal"],forme=[[0,1,0],[1,1,1],[0,1,0]]),
    dict(nom="Bâton de Feu",      prix=35, puissance=15.0,
         tags=["Magie","Feu"],    forme=[[1],[1],[1],[1]]),
    dict(nom="Gemme Stellaire",   prix=28, puissance=11.0,
         tags=["Magie","Étoile"], forme=[[1,0,1],[0,1,0],[1,0,1]]),
    dict(nom="Fiole de Poison",   prix=10, puissance=5.0,
         tags=["Poison","Herbe"], forme=[[1,1]]),
    # -- Défense ------------------------------------------------
    dict(nom="Bouclier Renforcé", prix=18, puissance=6.0,
         tags=["Bouclier","Armure"],forme=[[1,1],[1,1]]),
    dict(nom="Pavois de Pierre",  prix=22, puissance=8.0,
         tags=["Bouclier","Pierre"],forme=[[1,1,1],[1,0,1]]),
    # -- Objets spéciaux ----------------------------------------
    dict(nom="Tête d'Ail",        prix=8,  puissance=2.0,
         tags=["Ail"],            forme=[[1]]),
    dict(nom="Cristal de Glace",  prix=20, puissance=9.0,
         tags=["Glace","Cristal"],forme=[[1,0],[0,1]]),
    dict(nom="Souffre Volcanique",prix=14, puissance=6.0,
         tags=["Soufre","Feu"],   forme=[[1,1,0],[0,1,0]]),
]

# Catalogue de Containers achetables
CONTAINER_CATALOGUE: list[dict] = [
    dict(nom="Sacoche",           prix=10, largeur=2, hauteur=2),
    dict(nom="Ceinturon",        prix=15, largeur=3, hauteur=2),
    dict(nom="Sac de Voyage",    prix=20, largeur=3, hauteur=3),
    dict(nom="Coffre Portable",  prix=30, largeur=4, hauteur=3),
    dict(nom="Besace de Mage",   prix=25, largeur=2, hauteur=4),
    dict(nom="Grand Coffre",     prix=45, largeur=5, hauteur=4),
    dict(nom="Sac à Dos Elfique",prix=35, largeur=4, hauteur=4),
]


# ==============================================================================
# 2. CLASSE CONTAINER
# ==============================================================================

@dataclass
class Container:
    """
    Zone rectangulaire achetable définissant les cases "autorisées"
    pour le placement d'items sur la grille globale GRID_SIZE*GRID_SIZE.

    Attributs
    ---------
    id         : identifiant unique auto-incrémenté
    nom        : nom du sac (ex: "Sacoche")
    prix       : coût d'achat en pièces d'or
    largeur    : largeur de la zone (colonnes)
    hauteur    : hauteur de la zone (lignes)
    position_x : colonne du coin supérieur-gauche sur la grille globale
    position_y : ligne du coin supérieur-gauche sur la grille globale
    """
    id:         int
    nom:        str
    prix:       int
    largeur:    int
    hauteur:    int
    position_x: int = 0  # Placé dynamiquement lors de l'achat
    position_y: int = 0

    _next_id: int = field(default=1, init=False, repr=False)

    def cells(self) -> set[tuple[int, int]]:
        """
        Retourne l'ensemble des cases (row, col) couvertes par ce container.
        Complexité : O(hauteur * largeur)
        """
        return {
            (self.position_y + r, self.position_x + c)
            for r in range(self.hauteur)
            for c in range(self.largeur)
        }

    def to_dict(self) -> dict:
        """
        Retourne une représentation en dictionnaire de ce container.
        
        :param self: Description
        :return: Description
        :rtype: dict
        """
        return {
            "id": self.id, "nom": self.nom, "prix": self.prix,
            "largeur": self.largeur, "hauteur": self.hauteur,
            "position_x": self.position_x, "position_y": self.position_y,
        }

    @classmethod
    def from_catalogue(cls, template: dict, container_id: int) -> "Container":
        """Crée un Container à partir d'un template du catalogue et d'un ID assigné.
        """
        return cls(
            id=container_id,
            nom=template["nom"],
            prix=template["prix"],
            largeur=template["largeur"],
            hauteur=template["hauteur"],
        )


# ==============================================================================
# 3. CLASSE ITEM
# ==============================================================================

@dataclass
class Item:
    """
    Objet à placer dans le sac à dos.
    Sa forme 2D binaire détermine les cases qu'il occupe.

    Attributs
    ---------
    id             : identifiant unique
    nom            : nom de l'item
    forme          : numpy array 2D binaire (1=occupé, 0=vide)
    prix           : coût d'achat
    puissance_base : dégâts/points avant synergies
    tags           : liste de tags sémantiques (ex: ["Arme","Feu"])
    rotation       : angle courant (0, 90, 180, 270)
    """
    id:             int
    nom:            str
    forme:          np.ndarray   # shape (h, w), dtype int
    prix:           int
    puissance_base: float
    tags:           list[str] = field(default_factory=list)
    rotation:       int = 0      # degrés (0/90/180/270)

    # -- Forme et rotation ----------------------------------------

    def get_rotated_shape(self) -> np.ndarray:
        """
        Retourne la matrice de l'item après application de la rotation courante.

        k rotations de 90° anti-horaire via np.rot90.
        k=0->0°, k=1->90°, k=2->180°, k=3->270°

        Complexité : O(h * w) - proportionnel à la bounding box.
        """
        k = (self.rotation // 90) % 4
        return np.rot90(self.forme, k=k)

    def rotate_cw(self) -> "Item":
        """Retourne un nouvel Item avec rotation de +90° (sens horaire)."""
        return Item(
            id=self.id, nom=self.nom,
            forme=self.forme.copy(),
            prix=self.prix,
            puissance_base=self.puissance_base,
            tags=self.tags[:],
            rotation=(self.rotation + 90) % 360,
        )

    def all_rotations(self) -> list["Item"]:
        """Retourne les 4 orientations possibles."""
        rotations = []
        item = self
        for _ in range(4):
            rotations.append(item)
            item = item.rotate_cw()
        return rotations

    @property
    def height(self) -> int:
        """Retourne la hauteur actuelle de l'item selon sa rotation"""
        return self.get_rotated_shape().shape[0]

    @property
    def width(self) -> int:
        """Retourne la largeur actuelle de l'item selon sa rotation"""
        return self.get_rotated_shape().shape[1]

    def active_cells(self, x: int, y: int) -> list[tuple[int, int]]:
        """
        Retourne la liste des cases (row, col) occupées si l'item
        est placé avec son coin supérieur-gauche en (y, x).
        """
        shape = self.get_rotated_shape()
        return [
            (y + r, x + c)
            for r in range(shape.shape[0])
            for c in range(shape.shape[1])
            if shape[r, c] == 1
        ]

    def to_dict(self) -> dict:
        return {
            "id": self.id, "nom": self.nom, "prix": self.prix,
            "puissance_base": self.puissance_base, "tags": self.tags,
            "forme": self.get_rotated_shape().tolist(),
            "rotation": self.rotation,
        }

    @classmethod
    def from_catalogue(cls, template: dict, item_id: int) -> "Item":
        """Crée un Item à partir d'un template du catalogue et d'un ID assigné.
        """
        return cls(
            id=item_id,
            nom=template["nom"],
            forme=np.array(template["forme"], dtype=int),
            prix=template["prix"],
            puissance_base=template["puissance"],
            tags=template.get("tags", []),
        )


# ==============================================================================
# 4. CLASSE BACKPACKMANAGER
# ==============================================================================

class BackpackManager:
    """
    Coeur du moteur de jeu. Gère :
      - La grille globale (GRID_SIZE x GRID_SIZE) avec les items placés
      - Les containers achetés (zones autorisées)
      - Le placement/retrait d'items
      - La fonction objectif f(config) = \sum puissances + \sum synergies

    Encodage de la grille principale :
      grid[r][c] = 0        -> case vide
      grid[r][c] = item.id  -> case occupée par cet item

    Encodage de la grille de containers :
      container_grid[r][c] = 0           -> case non achetée (interdite)
      container_grid[r][c] = container.id-> case appartenant à ce container
    """

    def __init__(self):
        """
        self.grid : matrice 2D de la grille globale, avec les IDs des items placés
        self.container_grid : matrice 2D indiquant les zones achetées (IDs de containers)
        self.items_placed : dict {item_id: (Item, x, y)} pour retrouver les items et leurs positions
        self.containers_owned : dict {container_id: Container} pour gérer les containers achetés
        self._item_counter et self._container_counter : compteurs pour assigner des IDs uniques
        """
        # Grille d'items placés (ID ou 0)
        self.grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=int)
        # Grille des containers achetés (ID ou 0)
        self.container_grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=int)

        self.items_placed: dict[int, tuple[Item, int, int]] = {}  # id->(item,x,y)
        self.containers_owned: dict[int, Container] = {}  # id->container

        self._item_counter: int = 1
        self._container_counter: int = 1

    # -- Containers ----------------------------------------------

    def find_container_position(self, container: Container) -> Optional[tuple[int, int]]:
        """
        Trouve la première position libre pour placer un container sur la grille.
        Stratégie : balayage ligne par ligne, premier emplacement libre.

        Complexité : O(GRID_SIZE^2 * h * w) dans le pire cas.
        """
        for r in range(GRID_SIZE - container.hauteur + 1):
            for c in range(GRID_SIZE - container.largeur + 1):
                # Vérifier que toutes les cases sont libres dans la grille containers
                region = self.container_grid[r:r+container.hauteur, c:c+container.largeur]
                if np.all(region == 0):
                    return c, r   # (x=col, y=row)
        return None  # Grille pleine

    def add_container(self, container: Container) -> bool:
        """
        Place un container sur la grille globale.
        Assigne automatiquement la position si non définie.

        Returns True si le placement réussit.
        """
        pos = self.find_container_position(container)
        if pos is None:
            return False

        container.position_x, container.position_y = pos
        container.id = self._container_counter
        self._container_counter += 1

        # Marquer les cases du container
        for r in range(container.hauteur):
            for c in range(container.largeur):
                self.container_grid[container.position_y + r][container.position_x + c] = container.id

        self.containers_owned[container.id] = container
        return True

    def remove_container(self, container_id: int) -> bool:
        """
        Retire un container et tous les items qu'il contient.
        Complexité : O(GRID_SIZE^2) pour le nettoyage.
        """
        if container_id not in self.containers_owned:
            return False

        container = self.containers_owned[container_id]
        freed_cells = container.cells()

        # Retirer les items sur ces cases
        items_to_remove = set()
        for (r, c) in freed_cells:
            if self.grid[r][c] != 0:
                items_to_remove.add(self.grid[r][c])
        for item_id in items_to_remove:
            self.remove_item(item_id)

        # Nettoyer la grille containers
        for (r, c) in freed_cells:
            self.container_grid[r][c] = 0

        del self.containers_owned[container_id]
        return True

    # -- Validation et placement d'items ------------------------

    def is_valid(self, item: Item, x: int, y: int) -> bool:
        """
        Vérifie si l'item peut être placé avec son coin sup-gauche en (x, y).

        Deux conditions CUMULATIVES :
          1. Aucun chevauchement avec un item déjà placé (grid != 0)
          2. CRITIQUE : chaque case '1' de l'item doit se trouver dans
             une case couverte par un container acheté (container_grid != 0)

        Complexité : O(h * w) - parcours de la bounding box.

        Parameters
        ----------
        item : Item    objet à tester
        x    : int     colonne du coin supérieur-gauche
        y    : int     ligne du coin supérieur-gauche

        Returns True si le placement est légal.
        """
        shape = item.get_rotated_shape()
        h, w = shape.shape

        for dr in range(h):
            for dc in range(w):
                if shape[dr, dc] == 0:
                    continue  # Case inactive -> skip
                
                # Coordonnées globales de la case à vérifier
                r, c = y + dr, x + dc

                # 1. Dépassement de la grille globale
                if r < 0 or r >= GRID_SIZE or c < 0 or c >= GRID_SIZE:
                    return False

                # 2. Chevauchement avec un item existant
                if self.grid[r][c] != 0:
                    return False

                # 3. CONDITION CRITIQUE : la case doit être dans un container acheté
                if self.container_grid[r][c] == 0:
                    return False

        return True

    def place_item(self, item: Item, x: int, y: int) -> bool:
        """
        Place un item sur la grille si is_valid() retourne True.
        Complexité : O(h * w).

        Returns True si placé avec succès.
        """
        if not self.is_valid(item, x, y):
            return False

        shape = item.get_rotated_shape()
        for dr in range(shape.shape[0]):
            for dc in range(shape.shape[1]):
                if shape[dr, dc] == 1:
                    self.grid[y + dr][x + dc] = item.id

        self.items_placed[item.id] = (item, x, y)
        return True

    def remove_item(self, item_id: int) -> bool:
        """
        Retire un item de la grille.
        Complexité : O(GRID_SIZE^2) - scan complet pour nettoyer.
        Optimisable avec un index {id -> cells} si GRID_SIZE est grand.
        """
        # Vérifier que l'item existe
        if item_id not in self.items_placed:
            return False
        self.grid[self.grid == item_id] = 0
        # Supprimer de items_placed
        del self.items_placed[item_id]
        return True

    def valid_positions(self, item: Item) -> list[tuple[int, int]]:
        """
        Énumère toutes les positions légales (x, y) pour un item.
        Complexité : O(GRID_SIZE^2 * h * w) - goulot d'étranglement du solveur.
        """
        positions = []
        for r in range(GRID_SIZE):
            for c in range(GRID_SIZE):
                if self.is_valid(item, c, r):
                    positions.append((c, r))
        return positions

    # -- Fonction objectif f --------------------------------------

    def _synergy_multiplier(self, tags_a: list[str], tags_b: list[str]) -> float:
        """
        Calcule le bonus de synergie entre deux listes de tags.
        Teste toutes les paires et retourne la somme des bonus actifs.
        Complexité : O(|tags_a| * |tags_b|) - négligeable en pratique.
        """
        bonus = 0.0
        for ta in tags_a:
            for tb in tags_b:
                key = frozenset({ta, tb})
                if key in SYNERGY_TABLE:
                    bonus += SYNERGY_TABLE[key]
        return bonus

    def calculate_score(self) -> dict:
        """
        Calcule la fonction objectif complète :

            f = \sum puissance_base(i) + \sum bonus_synergie(i, j)

        Le bonus s'active si deux items i et j ont des cases adjacentes
        (voisinage orthogonal 4-directions) ET des tags compatibles.

        Algorithme :
          1. Pour chaque case occupée (r,c), récupérer l'item i
          2. Pour chaque voisin (r+-1,c) et (r,c+-1) contenant un item j != i
          3. Calculer le bonus synergie -> l'ajouter UNE SEULE FOIS (via ensemble)
          4. Sommer puissances de base + bonus

        Complexité : O(GRID_SIZE^2 * 4) = O(GRID_SIZE^2) - linéaire en la grille.

        Returns
        -------
        dict : {
          "total"         : float  - score total,
          "base_power"    : float  - somme des puissances de base,
          "synergy_bonus" : float  - total des bonus synergies,
          "details"       : list   - détail par item (id, nom, puissance_base),
          "synergies"     : list   - synergies actives,
        }
        """
        # Cas trivial : aucune item placé -> score 0
        if not self.items_placed:
            return {"total": 0.0, "base_power": 0.0, "synergy_bonus": 0.0,
                    "details": [], "synergies": []}

        DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        base_power = 0.0
        synergy_bonus = 0.0
        item_details = []
        synergy_list = []

        processed_items = set() # items déjà comptabilisés en puissance de base
        processed_synergies = set() # paires (id_a, id_b) déjà traitées

        for r in range(GRID_SIZE):
            for c in range(GRID_SIZE):
                item_id = self.grid[r][c]
                if item_id == 0:
                    continue

                item, ix, iy = self.items_placed[item_id]

                # -- Puissance de base (une seule fois par item) ----------
                if item_id not in processed_items:
                    base_power += item.puissance_base
                    item_details.append({
                        "id": item_id, "nom": item.nom,
                        "puissance_base": item.puissance_base,
                    })
                    processed_items.add(item_id)

                # -- Synergies avec les 4 voisins orthogonaux ------------
                for dr, dc in DIRS:
                    nr, nc = r + dr, c + dc
                    # Vérifier les limites de la grille
                    if nr < 0 or nr >= GRID_SIZE or nc < 0 or nc >= GRID_SIZE:
                        continue

                    neighbor_id = self.grid[nr][nc]
                    # Ignorer les cases vides et éviter le double-comptage (i,j) vs (j,i)
                    if neighbor_id == 0 or neighbor_id == item_id:
                        continue

                    # Clé symétrique pour éviter le double-comptage
                    pair_key = frozenset({item_id, neighbor_id})
                    if pair_key in processed_synergies:
                        continue
                    processed_synergies.add(pair_key)

                    neighbor_item, _, _ = self.items_placed[neighbor_id]
                    bonus = self._synergy_multiplier(item.tags, neighbor_item.tags)

                    if bonus > 0:
                        synergy_bonus += bonus
                        synergy_list.append({
                            "item_a": item.nom, "item_b": neighbor_item.nom,
                            "tags_a": item.tags, "tags_b": neighbor_item.tags,
                            "bonus": bonus,
                        })

        total = base_power + synergy_bonus
        return {
            "total": round(total, 2),
            "base_power": round(base_power, 2),
            "synergy_bonus": round(synergy_bonus, 2),
            "details": item_details,
            "synergies": synergy_list,
        }

    def get_grid_state(self) -> dict:
        """
        Sérialise l'état complet de la grille pour l'API / le frontend.
        """
        return {
            "item_grid": self.grid.tolist(),
            "container_grid": self.container_grid.tolist(),
            "items_placed": {
                str(iid): {
                    "item": item.to_dict(),
                    "x": x,
                    "y": y,
                }
                for iid, (item, x, y) in self.items_placed.items()
            },
            "containers": {
                str(cid): c.to_dict()
                for cid, c in self.containers_owned.items()
            },
        }

    def clone(self) -> "BackpackManager":
        """
        Copie profonde du manager pour les algorithmes d'exploration
        (recuit simulé, branch & bound, etc.).
        Complexité : O(GRID_SIZE^2).
        """
        new = BackpackManager()
        new.grid = self.grid.copy()
        new.container_grid = self.container_grid.copy()
        new.items_placed  = copy.deepcopy(self.items_placed)
        new.containers_owned = copy.deepcopy(self.containers_owned)
        new._item_counter = self._item_counter
        new._container_counter = self._container_counter
        return new


# ==============================================================================
# 5. CLASSE STORE
# ==============================================================================

class Store:
    """
    Marché aléatoire : génère 5 articles par tour (mélange Items + Containers).
    Gère le budget du joueur en pièces d'or.

    Règles :
      - Chaque tour propose exactement 5 articles au total.
      - Au moins 1 Container et au moins 2 Items sont garantis par tour.
      - Le reste est tiré aléatoirement dans les deux catalogues.
    """

    BUDGET_INITIAL: int = 1000

    def __init__(self):
        self.budget: int = self.BUDGET_INITIAL
        self.current_offers: list[dict] = [] # Offres du tour courant
        self._offer_counter: int = 1 # ID d'offre global

    # -- Génération de marché ------------------------------------

    def generate_market(self, seed: Optional[int] = None) -> list[dict]:
        """
        Génère 5 offres aléatoires pour le tour courant.
        Garantit au moins 1 Container et 2 Items.

        Returns
        -------
        list[dict] : liste des offres avec type, données et id d'offre.
        """
        if seed is not None:
            random.seed(seed)

        offers = []

        # -- Garanties minimales ----------------------------------
        # 1 container obligatoire
        c_template = random.choice(CONTAINER_CATALOGUE)
        offers.append(self._make_container_offer(c_template))

        # 2 items obligatoires
        for template in random.sample(ITEM_CATALOGUE, 2):
            offers.append(self._make_item_offer(template))

        # -- 2 articles aléatoires supplémentaires ----------------
        for _ in range(2):
            if random.random() < 0.35:   # 35% de chance d'être un container
                template = random.choice(CONTAINER_CATALOGUE)
                offers.append(self._make_container_offer(template))
            else:
                template = random.choice(ITEM_CATALOGUE)
                offers.append(self._make_item_offer(template))

        random.shuffle(offers)
        self.current_offers = offers
        return offers

    def _make_item_offer(self, template: dict) -> dict:
        oid = self._offer_counter; self._offer_counter += 1
        return {
            "offer_id": oid,
            "type": "item",
            "nom": template["nom"],
            "prix": template["prix"],
            "puissance": template["puissance"],
            "tags": template.get("tags", []),
            "forme": template["forme"],
            "affordable": self.budget >= template["prix"],
        }

    def _make_container_offer(self, template: dict) -> dict:
        oid = self._offer_counter; self._offer_counter += 1
        return {
            "offer_id": oid,
            "type": "container",
            "nom": template["nom"],
            "prix": template["prix"],
            "largeur": template["largeur"],
            "hauteur": template["hauteur"],
            "affordable": self.budget >= template["prix"],
        }

    # -- Transactions --------------------------------------------

    def can_afford(self, prix: int) -> bool:
        """Vérifie si le budget actuel permet d'acheter un article à ce prix."""
        return self.budget >= prix

    def spend(self, prix: int) -> bool:
        """Débite le budget. Retourne False si fonds insuffisants."""
        if not self.can_afford(prix):
            return False
        self.budget -= prix
        return True

    def buy_item(self, offer_id: int) -> Optional[Item]:
        """
        Achète un item via son offer_id.
        Retourne l'objet Item créé ou None si achat impossible.
        """
        # Rechercher l'offre correspondante
        offer = next((o for o in self.current_offers if o["offer_id"] == offer_id), None)
        if offer is None or offer["type"] != "item":
            return None
        if not self.spend(offer["prix"]):
            return None

        item = Item(
            id=offer["offer_id"] * 100 + random.randint(1, 99),
            nom=offer["nom"],
            forme=np.array(offer["forme"], dtype=int),
            prix=offer["prix"],
            puissance_base=offer["puissance"],
            tags=offer["tags"],
        )
        return item

    def buy_container(self, offer_id: int) -> Optional[Container]:
        """
        Achète un container via son offer_id.
        Retourne le Container créé ou None si achat impossible.
        """
        # Rechercher l'offre correspondante
        offer = next((o for o in self.current_offers if o["offer_id"] == offer_id), None)
        if offer is None or offer["type"] != "container":
            return None
        if not self.spend(offer["prix"]):
            return None

        container = Container(
            id=0,  # Sera assigné par BackpackManager.add_container()
            nom=offer["nom"],
            prix=offer["prix"],
            largeur=offer["largeur"],
            hauteur=offer["hauteur"],
        )
        return container

    def to_dict(self) -> dict:
        return {
            "budget":   self.budget,
            "offers":   self.current_offers,
        }


# ==============================================================================
# 6. RECUIT SIMULÉ (Simulated Annealing) - Moteur d'optimisation
# ==============================================================================

