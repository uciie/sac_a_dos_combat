"""
=====================================================================
  benchmark_backpack.py - Comparaison SA Classique vs SA + NN
  Backpack Battles - M1 MIAGE
=====================================================================
  Génère un fichier JSON de résultats exploitable par le dashboard.
  
  Usage :
      python benchmark_backpack.py
      -> Produit benchmark_results.json
=====================================================================
"""

from __future__ import annotations
import json, time, copy, statistics, random
import numpy as np
import torch

from models import (
    BackpackManager, Item, Container,
    SimulatedAnnealing, ITEM_CATALOGUE, CONTAINER_CATALOGUE
)
from nn_backpack import (
    GridEncoder, BackpackScoreNet, ExperienceBuffer,
    NNGuidedSA, generate_training_data, NNTrainer
)

# ── Reproductibilité ──────────────────────────────────────────────────────────
SEED        = 42
N_RUNS      = 8          # Répétitions par méthode (≥8 pour la variance)
N_MOVES     = 400        # Itérations SA par run
K_CANDIDATES = 5         # Candidats évalués par le NN à chaque step
TRAIN_CONFIGS = 150      # Configs pour l'entraînement du NN

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

"""
Le script exécute 8 runs indépendants pour chaque méthode sur le même setup (3 containers, 8 items, 400 itérations SA), 
puis agrège score moyen/écart-type/min/max, temps d'exécution, et courbes de convergence step par step.
"""

# ==============================================================================
# HELPERS
# ==============================================================================

def build_fresh_manager() -> BackpackManager:
    """Crée un BackpackManager propre avec 3 containers fixes."""
    mgr = BackpackManager()
    for cat in CONTAINER_CATALOGUE[:3]:          # Sacoche, Ceinturon, Sac de Voyage
        c = Container(id=0, nom=cat["nom"], prix=cat["prix"],
                      largeur=cat["largeur"], hauteur=cat["hauteur"])
        mgr.add_container(c)
    return mgr


def build_items(n: int = 8) -> list[Item]:
    """Sélectionne n items du catalogue."""
    selection = ITEM_CATALOGUE[:n]
    return [Item.from_catalogue(d, i+1) for i, d in enumerate(selection)]


def run_classic_sa(manager: BackpackManager, items: list[Item], n_moves: int) -> dict:
    """Lance un run SA classique et retourne les métriques."""
    sa = SimulatedAnnealing(manager, items)
    scores_per_step = []
    t0 = time.perf_counter()

    steps = n_moves // 50
    for _ in range(steps):
        snap = sa.step(n_moves=50)
        scores_per_step.append(snap["score"])

    elapsed = time.perf_counter() - t0
    if sa.T < 0.5:
        sa.restore_best()

    final_score = manager.calculate_score()["total"]
    return {
        "final_score":      final_score,
        "best_score":       sa.best_score,
        "elapsed_s":        elapsed,
        "scores_over_time": scores_per_step,
        "n_items_placed":   len(manager.items_placed),
    }


def run_nn_sa(
    manager: BackpackManager,
    items:   list[Item],
    model:   BackpackScoreNet,
    encoder: GridEncoder,
    n_moves: int,
) -> dict:
    """Lance un run SA guidé par le NN et retourne les métriques."""
    sa = NNGuidedSA(manager, items, model, encoder, k_candidates=K_CANDIDATES)
    scores_per_step = []
    t0 = time.perf_counter()

    steps = n_moves // 50
    for _ in range(steps):
        snap = sa.step(n_moves=50)
        scores_per_step.append(snap["score"])

    elapsed = time.perf_counter() - t0
    if sa.T < 0.5:
        sa.restore_best()

    final_score = manager.calculate_score()["total"]
    return {
        "final_score":      final_score,
        "best_score":       sa.best_score,
        "elapsed_s":        elapsed,
        "scores_over_time": scores_per_step,
        "n_items_placed":   len(manager.items_placed),
    }


def aggregate(runs: list[dict]) -> dict:
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
    print("  BENCHMARK — SA Classique vs SA + Réseau de Neurones")
    print("=" * 60)

    # ── Phase 0 : Entraîner le réseau ─────────────────────────────────────────
    print(f"\n[0/3] Génération du dataset ({TRAIN_CONFIGS} configs) + entraînement NN...")
    encoder = GridEncoder()
    mgr_ref = build_fresh_manager()
    items_ref = build_items(8)

    buffer  = generate_training_data(
        manager_template=mgr_ref,
        items_available=items_ref,
        encoder=encoder,
        n_configs=TRAIN_CONFIGS,
        sa_steps_per_config=100,
    )
    model   = BackpackScoreNet(in_channels=encoder.INPUT_CHANNELS)
    trainer = NNTrainer(model, encoder)
    trainer.train(buffer, epochs=30, batch_size=32, patience=8)
    metrics_train = trainer.evaluate(buffer)
    trainer.save_model("out/backpack_nn.pt")
    
    print(f"[Train] R²={metrics_train['r2']:.4f} | MAE={metrics_train['mae']:.3f}")

    # ── Phase 1 : N_RUNS × SA Classique ──────────────────────────────────────
    print(f"\n[1/3] SA Classique — {N_RUNS} runs × {N_MOVES} moves...")
    classic_runs = []
    for i in range(N_RUNS):
        mgr   = build_fresh_manager()
        items = build_items(8)
        result = run_classic_sa(mgr, items, N_MOVES)
        classic_runs.append(result)
        print(f"  Run {i+1:2d} | score={result['best_score']:.2f} | "
              f"time={result['elapsed_s']:.3f}s | items={result['n_items_placed']}")

    # ── Phase 2 : N_RUNS × SA + NN ───────────────────────────────────────────
    print(f"\n[2/3] SA + NN (k={K_CANDIDATES}) — {N_RUNS} runs × {N_MOVES} moves...")
    nn_runs = []
    for i in range(N_RUNS):
        mgr   = build_fresh_manager()
        items = build_items(8)
        result = run_nn_sa(mgr, items, model, encoder, N_MOVES)
        nn_runs.append(result)
        print(f"  Run {i+1:2d} | score={result['best_score']:.2f} | "
              f"time={result['elapsed_s']:.3f}s | items={result['n_items_placed']}")

    # ── Phase 3 : Agréger et sauvegarder ─────────────────────────────────────
    print("\n[3/3] Agrégation des résultats...")
    classic_agg = aggregate(classic_runs)
    nn_agg      = aggregate(nn_runs)

    gain_score  = nn_agg["mean_score"] - classic_agg["mean_score"]
    gain_pct    = gain_score / (classic_agg["mean_score"] + 1e-9) * 100
    overhead    = nn_agg["mean_time_s"] - classic_agg["mean_time_s"]

    results = {
        "config": {
            "seed": SEED, "n_runs": N_RUNS, "n_moves": N_MOVES,
            "k_candidates": K_CANDIDATES, "train_configs": TRAIN_CONFIGS,
        },
        "nn_training": {
            "r2":  round(metrics_train["r2"], 4),
            "mse": round(metrics_train["mse"], 3),
            "mae": round(metrics_train["mae"], 3),
            "train_losses": trainer.train_losses,
            "val_losses":   trainer.val_losses,
        },
        "classic_sa": classic_agg,
        "nn_sa":      nn_agg,
        "delta": {
            "score_gain":     round(gain_score, 3),
            "score_gain_pct": round(gain_pct, 2),
            "time_overhead_s": round(overhead, 4),
        },
    }

    out_path = "out/benchmark_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*60}")
    print(f"  Score moyen SA        : {classic_agg['mean_score']:.2f} ± {classic_agg['std_score']:.2f}")
    print(f"  Score moyen SA + NN   : {nn_agg['mean_score']:.2f} ± {nn_agg['std_score']:.2f}")
    print(f"  Gain                  : {gain_score:+.2f} ({gain_pct:+.1f}%)")
    print(f"  Temps SA              : {classic_agg['mean_time_s']:.4f}s ± {classic_agg['std_time_s']:.4f}s")
    print(f"  Temps SA + NN         : {nn_agg['mean_time_s']:.4f}s ± {nn_agg['std_time_s']:.4f}s")
    print(f"  Surcoût NN            : {overhead:+.4f}s / run")
    print(f"{'='*60}")
    print(f"\n  Résultats → {out_path}")
    return results


if __name__ == "__main__":
    main()