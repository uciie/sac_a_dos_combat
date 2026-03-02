"""
===================================================================
|  app.py - Serveur Flask : API REST pour Backpack Battles        |
|                                                                  |
|  Endpoints :                                                     |
|    GET  /                  => Interface HTML                      |
|    GET  /market            => Génère 5 offres aléatoires         |
|    GET  /state             => État complet du jeu                |
|    POST /buy               => Acheter un item ou container       |
|    POST /place             => Placer un item sur la grille       |
|    POST /calculate         => Calculer le score de la config     |
|    POST /optimize          => Lancer N itérations de recuit      |
|    POST /reset             => Réinitialiser le jeu               |
===================================================================
"""

from flask import Flask, jsonify, request, render_template_string, send_from_directory
from flask_cors import CORS
from flask.json.provider import DefaultJSONProvider
import os
import random
import numpy as np

from models import (
    BackpackManager, Store, Item, Container,
    SimulatedAnnealing, ITEM_CATALOGUE, CONTAINER_CATALOGUE, GRID_SIZE,
    SYNERGY_TABLE
)

# ===================================================================
# ENCODEUR JSON PERSONNALISÉ
# ===================================================================

class NumpyJSONProvider(DefaultJSONProvider):
    """
    Encodeur JSON étendu pour sérialiser les types numpy (int64, float64, ndarray)
    produits par BackpackManager et SimulatedAnnealing.
    Sans cela, Flask lève TypeError sur les valeurs issues de numpy.
    """
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


# ===================================================================
# INITIALISATION
# ===================================================================

app = Flask(__name__, static_folder="static")
app.json_provider_class = NumpyJSONProvider
app.json = NumpyJSONProvider(app)
CORS(app)  # Autoriser les requêtes cross-origin (dev)

# ── État global de la session de jeu ────────────────────────────
# Dans une vraie application, cet état serait en session/DB par joueur.
game_state = {
    "manager":   BackpackManager(),
    "store":     Store(),
    "sa_engine": None,         # Moteur de recuit simulé
    "inventory": [],           # Items achetés non encore placés
    "sa_history": [],          # Historique des scores SA
    "sa_running": False,
}


def reset_game():
    """Réinitialise toute la partie."""
    game_state["manager"]   = BackpackManager()
    game_state["store"]     = Store()
    game_state["sa_engine"] = None
    game_state["inventory"] = []
    game_state["sa_history"] = []
    game_state["sa_running"] = False


# ===================================================================
# ROUTES - INTERFACE
# ===================================================================

@app.route("/")
def index():
    # Si index.html est dans le même dossier ou un dossier /templates
    return send_from_directory('.', 'index.html')


# ===================================================================
# ROUTES - API
# ===================================================================

@app.route("/market", methods=["GET"])
def get_market():
    """
    GET /market
    Génère et retourne 5 offres aléatoires (Items + Containers).
    Indique si chaque offre est achetable selon le budget courant.

    Response 200 :
    {
      "budget": int,
      "offers": [
        {
          "offer_id": int,
          "type": "item" | "container",
          "nom": str,
          "prix": int,
          "affordable": bool,
          // item: puissance, tags, forme
          // container: largeur, hauteur
        }
      ]
    }
    """
    store = game_state["store"]
    offers = store.generate_market()

    # Mettre à jour le flag affordable selon le budget actuel
    for offer in offers:
        offer["affordable"] = store.budget >= offer["prix"]

    return jsonify({
        "budget": store.budget,
        "offers": offers,
    })


@app.route("/state", methods=["GET"])
def get_state():
    """
    GET /state
    Retourne l'état complet : grille, items, containers, budget, score.

    Response 200 :
    {
      "grid_state": { item_grid, container_grid, items_placed, containers },
      "score":      { total, base_power, synergy_bonus, details, synergies },
      "budget":     int,
      "inventory":  [ item.to_dict() ],
      "sa_history": [ { iteration, temperature, score, best_score } ],
      "sa_running": bool,
    }
    """
    mgr   = game_state["manager"]
    store = game_state["store"]

    return jsonify({
        "grid_state":  mgr.get_grid_state(),
        "score":       mgr.calculate_score(),
        "budget":      store.budget,
        "grid_size":   GRID_SIZE,   # Source de vérité unique pour la taille de la grille
        "inventory":   [it.to_dict() for it in game_state["inventory"]],
        "sa_history":  game_state["sa_history"],
        "sa_running":  game_state["sa_running"],
    })


@app.route("/synergies", methods=["GET"])
def get_synergies():
    """
    GET /synergies
    Retourne SYNERGY_TABLE sérialisée pour le frontend.
    Les frozenset (non-sérialisables en JSON) sont convertis en
    paires [tagA, tagB] avec leur bonus associé.

    Response 200 :
    {
      "synergies": [
        { "tags": [tagA, tagB], "bonus": float },
        ...
      ]
    }
    """
    data = [
        {"tags": list(key), "bonus": bonus}
        for key, bonus in SYNERGY_TABLE.items()
    ]
    return jsonify({"synergies": data})


@app.route("/buy", methods=["POST"])
def buy():
    """
    POST /buy
    Achète un article du marché et l'ajoute à l'inventaire ou au manager.

    Body JSON :
    { "offer_id": int }

    Response 200 :
    { "success": bool, "message": str, "budget": int, "type": str }
    """
    data     = request.get_json()
    offer_id = data.get("offer_id")

    if offer_id is None:
        return jsonify({"success": False, "message": "offer_id manquant"}), 400

    store = game_state["store"]
    offer = next((o for o in store.current_offers if o["offer_id"] == offer_id), None)

    if offer is None:
        return jsonify({"success": False, "message": "Offre introuvable"}), 404

    if not store.can_afford(offer["prix"]):
        return jsonify({"success": False, "message": "Budget insuffisant",
                        "budget": store.budget}), 400

    if offer["type"] == "item":
        item = store.buy_item(offer_id)
        if item:
            game_state["inventory"].append(item)
            return jsonify({
                "success": True,
                "message": f"'{item.nom}' ajouté à l'inventaire",
                "budget":  store.budget,
                "type":    "item",
                "item":    item.to_dict(),
            })

    elif offer["type"] == "container":
        container = store.buy_container(offer_id)
        if container:
            mgr = game_state["manager"]
            placed = mgr.add_container(container)
            if placed:
                return jsonify({
                    "success":   True,
                    "message":   f"'{container.nom}' placé sur la grille",
                    "budget":    store.budget,
                    "type":      "container",
                    "container": container.to_dict(),
                })
            else:
                # Rembourser si la grille est pleine
                store.budget += container.prix
                return jsonify({"success": False, "message": "Grille pleine, container non placé",
                                "budget": store.budget}), 400

    return jsonify({"success": False, "message": "Erreur inconnue"}), 500


@app.route("/place", methods=["POST"])
def place_item():
    """
    POST /place
    Place un item de l'inventaire sur la grille.

    Body JSON :
    { "item_id": int, "x": int, "y": int, "rotation": int (0/90/180/270) }

    Response 200 :
    { "success": bool, "message": str, "score": dict }
    """
    data     = request.get_json()
    item_id  = data.get("item_id")
    x        = data.get("x", 0)
    y        = data.get("y", 0)
    rotation = data.get("rotation", 0)

    # Trouver l'item dans l'inventaire
    inv  = game_state["inventory"]
    item = next((it for it in inv if it.id == item_id), None)

    if item is None:
        return jsonify({"success": False, "message": "Item non trouvé dans l'inventaire"}), 404

    # Appliquer la rotation :
    # - item.rotation = rotation d'origine stockée en mémoire (ex: 270°)
    # - 'rotation' reçu = delta ajouté par l'utilisateur depuis le JS (0°, 90°, 180°, 270°)
    # - La forme JSON envoyée au JS est déjà pré-tournée (to_dict → get_rotated_shape),
    #   donc le JS repart de 0° et n'envoie que le delta. On additionne les deux.
    original_rotation = item.rotation
    item.rotation = (item.rotation + rotation) % 360

    mgr = game_state["manager"]
    if mgr.place_item(item, x, y):
        game_state["inventory"].remove(item)
        score_data = mgr.calculate_score()
        return jsonify({
            "success": True,
            "message": f"'{item.nom}' placé en ({x},{y})",
            "score":   int(score_data["total"]),
        })
    else:
        # Annuler la rotation si le placement échoue
        item.rotation = original_rotation
        return jsonify({
            "success": False,
            "message": "Position invalide (chevauchement ou hors container)",
        }), 400


@app.route("/calculate", methods=["POST"])
def calculate():
    """
    POST /calculate
    Calcule et retourne le score de puissance de la configuration actuelle.

    Body JSON : {} (utilise la grille courante)
    Optionnel : { "grid_override": [[...]] } pour tester une config spécifique

    Response 200 :
    {
      "total": float,
      "base_power": float,
      "synergy_bonus": float,
      "details": [...],
      "synergies": [...]
    }
    """
    mgr   = game_state["manager"]
    score = mgr.calculate_score()
    return jsonify(score)


@app.route("/optimize", methods=["POST"])
def optimize():
    """
    POST /optimize
    Lance N itérations du Recuit Simulé et retourne l'état mis à jour.
    Conçu pour être appelé en boucle depuis le frontend (animation).

    Body JSON :
    {
      "n_moves":     int  (défaut: 50)   - itérations par appel,
      "temperature": float (optionnel)   - forcer la température,
      "reset":       bool (optionnel)    - réinitialiser le SA,
    }

    Response 200 :
    {
      "iteration":    int,
      "temperature":  float,
      "score":        float,
      "best_score":   float,
      "base_power":   float,
      "synergy_bonus": float,
      "grid_state":   dict,
      "synergies":    list,
    }
    """
    data    = request.get_json() or {}
    n_moves = data.get("n_moves", 50)
    do_reset = data.get("reset", False)

    mgr = game_state["manager"]

    # Initialiser ou réinitialiser le moteur SA
    if game_state["sa_engine"] is None or do_reset:
        available_copy = [it for it in game_state["inventory"]]
        game_state["sa_engine"] = SimulatedAnnealing(mgr, available_copy)
        game_state["sa_history"] = []
        game_state["sa_running"] = True

    sa = game_state["sa_engine"]

    # Forcer la température si demandé
    if "temperature" in data:
        sa.T = float(data["temperature"])

    # Exécuter les itérations
    snapshot = sa.step(n_moves=n_moves)
    game_state["sa_history"].append(snapshot)

    # Stopper si la température est trop basse
    if sa.T < 0.5:
        game_state["sa_running"] = False
        sa.restore_best()

    score_data = mgr.calculate_score()

    return jsonify({
        **snapshot,
        "grid_state": mgr.get_grid_state(),
        "synergies":  score_data.get("synergies", []),
        "sa_running": game_state["sa_running"],
    })


@app.route("/reset", methods=["POST"])
def reset():
    """
    POST /reset
    Réinitialise complètement la partie.

    Response 200 :
    { "success": True, "message": str }
    """
    reset_game()
    return jsonify({"success": True, "message": "Partie réinitialisée"})


@app.route("/remove_item", methods=["POST"])
def remove_item():
    """
    POST /remove_item
    Retire un item de la grille et le remet en inventaire.

    Body JSON : { "item_id": int }
    """
    data    = request.get_json()
    item_id = data.get("item_id")

    mgr = game_state["manager"]
    if item_id in mgr.items_placed:
        item, x, y = mgr.items_placed[item_id]
        if mgr.remove_item(item_id):
            game_state["inventory"].append(item)
            return jsonify({"success": True, "score": mgr.calculate_score()})

    return jsonify({"success": False, "message": "Item non trouvé"}), 404


# ===================================================================
# POINT D'ENTRÉE
# ===================================================================

if __name__ == "__main__":
    print("============================================")
    print("|  Backpack Battles - Serveur Flask        |")
    print("|  http://localhost:5000                   |")
    print("============================================")
    app.run(debug=True, host="0.0.0.0", port=5000)