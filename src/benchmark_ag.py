"""
=====================================================================
  benchmark_ga.py - Comparaison AG Classique vs AG + ML
  Backpack Battles - M1 MIAGE
=====================================================================
  Génère un fichier JSON de résultats exploitable.
  
  Usage :
      python benchmark_ga.py
      -> Produit out/benchmark_ga_results.json
=====================================================================
"""

import json
import time
import statistics
import random
import os
import numpy as np

from models import (
    BackpackManager, Item, Container,
    ITEM_CATALOGUE, CONTAINER_CATALOGUE, Store
)
from ml_surrogate import NNSurrogate
from models_ga import GeneticAlgorithmML

# ── Reproductibilité et Paramètres ────────────────────────────────────────────
SEED          = 42
N_RUNS        = 8          # Nombre de répétitions par méthode
GENERATIONS   = 50         # Nombre de générations pour l'AG
POP_SIZE      = 20         # Taille de la population
TRAIN_SAMPLES = 5000       # Nombre d'exemples pour entraîner le modèle ML

random.seed(SEED)
np.random.seed(SEED)

# ==============================================================================
# HELPERS (Inspirés de vos expériences et benchmarks existants)
# ==============================================================================

def build_fresh_manager() -> BackpackManager:
    """Crée un BackpackManager propre avec 3 containers fixes."""
    mgr = BackpackManager()
    for cat in CONTAINER_CATALOGUE[:3]:
        c = Container(id=0, nom=cat["nom"], prix=cat["prix"],
                      largeur=cat["largeur"], hauteur=cat["hauteur"])
        mgr.add_container(c)
    return mgr

def build_items(n: int = 15) -> list:
    """Sélectionne aléatoirement n items du catalogue pour constituer l'inventaire."""
    inventory = []
    for _ in range(n):
        template = random.choice(ITEM_CATALOGUE)
        item_id = random.randint(1000, 99999)
        inventory.append(Item.from_catalogue(template, item_id))
    return inventory

def generate_training_data(num_samples: int):
    """Génère le dataset d'entraînement pour le réseau de neurones (Surrogate)."""
    print(f"Génération de {num_samples} exemples d'entraînement...")
    X, y = [], []
    surrogate = NNSurrogate(len(ITEM_CATALOGUE))
    
    for _ in range(num_samples):
        store = Store()
        manager = BackpackManager()
        store.generate_market()
        
        container_offer = next((o for o in store.current_offers if o["type"] == "container"), None)
        if container_offer:
            container = store.buy_container(container_offer["offer_id"])
            if container:
                manager.add_container(container)
        
        items_in_bag = []
        for template in random.sample(ITEM_CATALOGUE, random.randint(3, 8)):
            item = Item.from_catalogue(template, random.randint(1000, 99999))
            pos = manager.valid_positions(item)
            if pos:
                nx, ny = random.choice(pos)
                manager.place_item(item, nx, ny)
                items_in_bag.append(item)
                
        features = surrogate.extract_features(items_in_bag)
        score = manager.calculate_score()["total"] 
        X.append(features)
        y.append(score)
        
    return np.array(X), np.array(y), surrogate

def run_ga_experiment(manager: BackpackManager, inventory: list, model: NNSurrogate, generations: int, pop_size: int) -> dict:
    """Exécute l'Algorithme Génétique et retourne les métriques."""
    ga = GeneticAlgorithmML(manager, inventory, surrogate_model=model, pop_size=pop_size)
    t0 = time.perf_counter()
    
    # Exécution de l'algorithme sur toutes les générations
    ga.step(generations=generations)
    
    elapsed = time.perf_counter() - t0
    
    # Extraction de l'historique de convergence
    scores_per_step = [h["best_score"] for h in ga.history]
    
    return {
        "final_score":      ga.best_score,
        "best_score":       ga.best_score,
        "elapsed_s":        elapsed,
        "scores_over_time": scores_per_step,
        "n_items_placed":   len(ga.best_manager.items_placed),
    }

def aggregate(runs: list) -> dict:
    """Calcule les statistiques agrégées sur plusieurs runs."""
    scores  = [r["best_score"]    for r in runs]
    times   = [r["elapsed_s"]     for r in runs]
    items_p = [r["n_items_placed"] for r in runs]
    return {
        "mean_score":  round(statistics.mean(scores), 3),
        "std_score":   round(statistics.stdev(scores) if len(scores) > 1 else 0.0, 3),
        "min_score":   round(min(scores), 3),
        "max_score":   round(max(scores), 3),
        "mean_time_s": round(statistics.mean(times), 4),
        "std_time_s":  round(statistics.stdev(times)  if len(times) > 1 else 0.0, 4),
        "mean_items":  round(statistics.mean(items_p), 2),
        "all_scores":  [round(s, 3) for s in scores],
        "all_times":   [round(t, 4) for t in times],
        "conv_curves": [r["scores_over_time"] for r in runs],
    }

# ==============================================================================
# PIPELINE PRINCIPAL
# ==============================================================================

def main():
    print("=" * 60)
    print("  BENCHMARK — Algorithme Génétique Classique vs AG + ML")
    print("=" * 60)

    # ── Phase 0 : Entraîner le réseau de neurones ─────────────────────────────
    print(f"\n[0/3] Entraînement du Surrogate Model (MLPRegressor)...")
    X_train, y_train, dummy_surrogate = generate_training_data(TRAIN_SAMPLES)
    trained_model = NNSurrogate(len(ITEM_CATALOGUE))
    trained_model.train(X_train, y_train)
    
    # Modèle "vide" (non entraîné) pour l'AG classique
    classic_model = NNSurrogate(len(ITEM_CATALOGUE))

    # ── Phase 1 : N_RUNS × AG Classique ──────────────────────────────────────
    print(f"\n[1/3] AG Classique — {N_RUNS} runs × {GENERATIONS} générations...")
    classic_runs = []
    for i in range(N_RUNS):
        mgr   = build_fresh_manager()
        items = build_items(15)  # Inventaire de 15 items aléatoires
        result = run_ga_experiment(mgr, items, classic_model, GENERATIONS, POP_SIZE)
        classic_runs.append(result)
        print(f"  Run {i+1:2d} | score={result['best_score']:.2f} | "
              f"time={result['elapsed_s']:.3f}s | items={result['n_items_placed']}")

    # ── Phase 2 : N_RUNS × AG + ML ───────────────────────────────────────────
    print(f"\n[2/3] AG + ML (Surrogate) — {N_RUNS} runs × {GENERATIONS} générations...")
    ml_runs = []
    for i in range(N_RUNS):
        mgr   = build_fresh_manager()
        items = build_items(15)
        # On utilise exactement les mêmes items pour une comparaison loyale
        result = run_ga_experiment(mgr, items, trained_model, GENERATIONS, POP_SIZE)
        ml_runs.append(result)
        print(f"  Run {i+1:2d} | score={result['best_score']:.2f} | "
              f"time={result['elapsed_s']:.3f}s | items={result['n_items_placed']}")

    # ── Phase 3 : Agréger et sauvegarder ─────────────────────────────────────
    print("\n[3/3] Agrégation des résultats...")
    classic_agg = aggregate(classic_runs)
    ml_agg      = aggregate(ml_runs)

    gain_score  = ml_agg["mean_score"] - classic_agg["mean_score"]
    gain_pct    = gain_score / (classic_agg["mean_score"] + 1e-9) * 100
    overhead    = ml_agg["mean_time_s"] - classic_agg["mean_time_s"]

    results = {
        "config": {
            "seed": SEED, "n_runs": N_RUNS, "generations": GENERATIONS,
            "pop_size": POP_SIZE, "train_samples": TRAIN_SAMPLES,
        },
        "classic_ga": classic_agg,
        "ml_ga":      ml_agg,
        "delta": {
            "score_gain":     round(gain_score, 3),
            "score_gain_pct": round(gain_pct, 2),
            "time_overhead_s": round(overhead, 4),
        },
    }

    # Création du dossier out/ s'il n'existe pas
    os.makedirs("out", exist_ok=True)
    out_path = "out/benchmark_ga_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*60}")
    print(f"  Score moyen AG Classique : {classic_agg['mean_score']:.2f} ± {classic_agg['std_score']:.2f}")
    print(f"  Score moyen AG + ML      : {ml_agg['mean_score']:.2f} ± {ml_agg['std_score']:.2f}")
    print(f"  Gain de score            : {gain_score:+.2f} ({gain_pct:+.1f}%)")
    print(f"  Temps moyen AG Classique : {classic_agg['mean_time_s']:.4f}s ± {classic_agg['std_time_s']:.4f}s")
    print(f"  Temps moyen AG + ML      : {ml_agg['mean_time_s']:.4f}s ± {ml_agg['std_time_s']:.4f}s")
    print(f"  Surcoût temps ML         : {overhead:+.4f}s / run")
    print(f"{'='*60}")
    print(f"\n  Résultats sauvegardés dans → {out_path}")

if __name__ == "__main__":
    main()