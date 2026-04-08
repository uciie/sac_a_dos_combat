"""
=====================================================================
  plot_benchmark.py — Génération de tous les graphiques
  Backpack Battles - M1 MIAGE

  Lit benchmark_results.json (produit par benchmark_backpack.py)
  et génère tous les graphiques en PNG haute résolution.

  Usage :
      python benchmark_backpack.py   # d'abord
      python plot_benchmark.py       # ensuite
      → Dossier  plots/  avec 8 fichiers PNG

  Graphiques produits :
      01_convergence_moyenne.png
      02_boxplot_scores.png
      03_scores_par_run.png
      04_temps_execution.png
      05_correlation_score_temps.png
      06_courbe_perte_nn.png
      07_heatmap_gains.png
      08_radar_synthese.png
=====================================================================
"""

import json
import os
import statistics
import math

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MaxNLocator
from pathlib import Path

# ── Configuration visuelle ────────────────────────────────────────────────────

plt.rcParams.update({
    "figure.dpi":        150,
    "savefig.dpi":       150,
    "savefig.bbox":      "tight",
    "savefig.pad_inches": 0.25,
    "font.family":       "DejaVu Sans",
    "font.size":         11,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         True,
    "grid.alpha":        0.35,
    "grid.linewidth":    0.6,
    "axes.titlesize":    13,
    "axes.titleweight":  "bold",
    "axes.titlepad":     10,
    "axes.labelsize":    11,
    "legend.fontsize":   10,
    "legend.framealpha": 0.85,
    "lines.linewidth":   2,
})

# Palette cohérente avec le dashboard
C_SA  = "#185FA5"   # Bleu SA classique
C_NN  = "#0F6E56"   # Vert SA + NN
C_SA_L = "#B5D4F4"  # Bleu clair
C_NN_L = "#9FE1CB"  # Vert clair
C_WARN = "#BA7517"  # Ambre (gain)
C_GRAY = "#888780"

OUTPUT_DIR = Path("plots")
OUTPUT_DIR.mkdir(exist_ok=True)


# ── Utilitaires ───────────────────────────────────────────────────────────────

def load_data(path: str = "out/benchmark_results.json") -> dict:
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        print(f"[✓] Données chargées depuis {path}")
        return data
    else:
        print(f"[!] {path} introuvable — utilisation des données de démo")
        return 


def avg(lst):
    return sum(lst) / len(lst)


def save(fig, name: str):
    path = OUTPUT_DIR / name
    fig.savefig(path)
    plt.close(fig)
    print(f"  → {path}")


# ══════════════════════════════════════════════════════════════════════════════
# GRAPHIQUE 1 — Convergence moyenne avec enveloppe d'écart-type
# ══════════════════════════════════════════════════════════════════════════════

def plot_convergence(data: dict):
    sa = data["classic_sa"]
    nn = data["nn_sa"]
    cfg = data["config"]

    steps     = len(sa["conv_curves"][0])
    x_labels  = [(i + 1) * 50 for i in range(steps)]
    n_runs    = len(sa["conv_curves"])

    sa_curves = np.array(sa["conv_curves"])  # (n_runs, steps)
    nn_curves = np.array(nn["conv_curves"])

    sa_mean = sa_curves.mean(axis=0)
    nn_mean = nn_curves.mean(axis=0)
    sa_std  = sa_curves.std(axis=0)
    nn_std  = nn_curves.std(axis=0)

    fig, ax = plt.subplots(figsize=(9, 5))

    # Enveloppes ±1σ
    ax.fill_between(x_labels, sa_mean - sa_std, sa_mean + sa_std,
                    alpha=0.18, color=C_SA, label="_nolegend_")
    ax.fill_between(x_labels, nn_mean - nn_std, nn_mean + nn_std,
                    alpha=0.18, color=C_NN, label="_nolegend_")

    # Courbes individuelles (très transparentes)
    for c in sa_curves:
        ax.plot(x_labels, c, color=C_SA, alpha=0.07, linewidth=0.8)
    for c in nn_curves:
        ax.plot(x_labels, c, color=C_NN, alpha=0.07, linewidth=0.8)

    # Moyennes
    ax.plot(x_labels, sa_mean, color=C_SA, linewidth=2.4, label="SA classique (moyenne)")
    ax.plot(x_labels, nn_mean, color=C_NN, linewidth=2.4, label=f"SA + NN k={cfg['k_candidates']} (moyenne)",
            linestyle="--")

    ax.set_title("Convergence moyenne — SA classique vs SA + NN")
    ax.set_xlabel("Itérations")
    ax.set_ylabel("Score de puissance")
    ax.legend(loc="lower right")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=8))

    # Annotation gain final
    gain = nn_mean[-1] - sa_mean[-1]
    ax.annotate(f"+{gain:.1f} pts",
                xy=(x_labels[-1], nn_mean[-1]),
                xytext=(-60, 12), textcoords="offset points",
                fontsize=10, color=C_WARN, fontweight="bold",
                arrowprops=dict(arrowstyle="-", color=C_WARN, lw=1.2))

    save(fig, "01_convergence_moyenne.png")


# ══════════════════════════════════════════════════════════════════════════════
# GRAPHIQUE 2 — Boxplot des scores finaux
# ══════════════════════════════════════════════════════════════════════════════

def plot_boxplot(data: dict):
    sa_scores = data["classic_sa"]["all_scores"]
    nn_scores = data["nn_sa"]["all_scores"]

    fig, ax = plt.subplots(figsize=(7, 5))

    bp = ax.boxplot(
        [sa_scores, nn_scores],
        tick_labels=["SA classique", "SA + NN"],
        patch_artist=True,
        widths=0.45,
        medianprops=dict(color="white", linewidth=2.5),
        whiskerprops=dict(linewidth=1.4),
        capprops=dict(linewidth=1.4),
        flierprops=dict(marker="o", markersize=6, linestyle="none"),
    )
    bp["boxes"][0].set_facecolor(C_SA_L)
    bp["boxes"][0].set_edgecolor(C_SA)
    bp["boxes"][1].set_facecolor(C_NN_L)
    bp["boxes"][1].set_edgecolor(C_NN)
    bp["fliers"][0].set_markerfacecolor(C_SA)
    bp["fliers"][1].set_markerfacecolor(C_NN)

    # Points individuels (jittered)
    rng = np.random.default_rng(42)
    for i, (scores, color) in enumerate([(sa_scores, C_SA), (nn_scores, C_NN)], 1):
        jitter = rng.uniform(-0.08, 0.08, len(scores))
        ax.scatter([i + j for j in jitter], scores,
                   color=color, zorder=5, s=35, alpha=0.8)

    ax.set_title("Distribution des scores finaux (tous runs)")
    ax.set_ylabel("Score de puissance (meilleur trouvé)")

    # Annotation moyenne
    for i, (scores, color) in enumerate([(sa_scores, C_SA), (nn_scores, C_NN)], 1):
        m = statistics.mean(scores)
        ax.text(i + 0.32, m, f"μ={m:.1f}", va="center",
                fontsize=9.5, color=color, fontweight="bold")

    save(fig, "02_boxplot_scores.png")


# ══════════════════════════════════════════════════════════════════════════════
# GRAPHIQUE 3 — Scores par run (barres groupées)
# ══════════════════════════════════════════════════════════════════════════════

def plot_scores_per_run(data: dict):
    n_runs    = data["config"]["n_runs"]
    sa_scores = data["classic_sa"]["all_scores"]
    nn_scores = data["nn_sa"]["all_scores"]

    x      = np.arange(n_runs)
    width  = 0.38
    labels = [f"Run {i+1}" for i in range(n_runs)]

    fig, ax = plt.subplots(figsize=(10, 5))

    bars_sa = ax.bar(x - width/2, sa_scores, width, color=C_SA_L,
                     edgecolor=C_SA, linewidth=0.8, label="SA classique")
    bars_nn = ax.bar(x + width/2, nn_scores, width, color=C_NN_L,
                     edgecolor=C_NN, linewidth=0.8, label="SA + NN")

    # Delta labels
    for i, (s, n) in enumerate(zip(sa_scores, nn_scores)):
        delta = n - s
        color = C_NN if delta >= 0 else "#A32D2D"
        sign  = "+" if delta >= 0 else ""
        ax.text(i, max(s, n) + 1.2, f"{sign}{delta:.1f}",
                ha="center", va="bottom", fontsize=9, color=color, fontweight="bold")

    ax.set_title("Scores par run — comparaison individuelle")
    ax.set_ylabel("Score de puissance (meilleur trouvé)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.legend()

    # Lignes de moyenne
    ax.axhline(statistics.mean(sa_scores), color=C_SA, linewidth=1.2,
               linestyle=":", alpha=0.7)
    ax.axhline(statistics.mean(nn_scores), color=C_NN, linewidth=1.2,
               linestyle=":", alpha=0.7)

    save(fig, "03_scores_par_run.png")


# ══════════════════════════════════════════════════════════════════════════════
# GRAPHIQUE 4 — Temps d'exécution comparé
# ══════════════════════════════════════════════════════════════════════════════

def plot_times(data: dict):
    n_runs   = data["config"]["n_runs"]
    sa_times = [t * 1000 for t in data["classic_sa"]["all_times"]]  # en ms
    nn_times = [t * 1000 for t in data["nn_sa"]["all_times"]]

    x     = np.arange(n_runs)
    width = 0.38
    labels = [f"R{i+1}" for i in range(n_runs)]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # ── Sous-graphe gauche : barres groupées ────────────────────────────
    ax = axes[0]
    ax.bar(x - width/2, sa_times, width, color=C_SA_L, edgecolor=C_SA,
           linewidth=0.8, label="SA classique")
    ax.bar(x + width/2, nn_times, width, color=C_NN_L, edgecolor=C_NN,
           linewidth=0.8, label="SA + NN")

    ax.set_title("Temps d'exécution par run")
    ax.set_ylabel("Durée (ms)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.legend()

    overhead_ms = data["delta"]["time_overhead_s"] * 1000
    ax.text(0.98, 0.96, f"Surcoût moyen NN\n+{overhead_ms:.0f} ms / run",
            transform=ax.transAxes, ha="right", va="top",
            fontsize=9.5, color=C_WARN,
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                      edgecolor=C_WARN, alpha=0.8))

    # ── Sous-graphe droit : scatter SA vs NN ────────────────────────────
    ax2 = axes[1]
    ax2.scatter(sa_times, nn_times, color=C_NN, s=60, zorder=5, edgecolors=C_NN_L)
    all_times = sa_times + nn_times
    lo, hi = min(all_times) * 0.95, max(all_times) * 1.05
    ax2.plot([lo, hi], [lo, hi], color=C_GRAY, linewidth=1.2,
             linestyle="--", alpha=0.6, label="Égalité")
    ax2.fill_between([lo, hi], [lo, hi], [hi, hi], alpha=0.06, color=C_NN)

    for i, (s, n) in enumerate(zip(sa_times, nn_times)):
        ax2.annotate(f"R{i+1}", (s, n), textcoords="offset points",
                     xytext=(5, 3), fontsize=8, color=C_GRAY)

    ax2.set_xlabel("SA classique (ms)")
    ax2.set_ylabel("SA + NN (ms)")
    ax2.set_title("Corrélation des temps d'exécution")
    ax2.legend(fontsize=9)
    ax2.set_xlim(lo, hi)
    ax2.set_ylim(lo, hi)

    fig.suptitle("Analyse des temps de calcul", fontsize=14, fontweight="bold", y=1.01)
    save(fig, "04_temps_execution.png")


# ══════════════════════════════════════════════════════════════════════════════
# GRAPHIQUE 5 — Corrélation score / temps pour chaque méthode
# ══════════════════════════════════════════════════════════════════════════════

def plot_score_time_scatter(data: dict):
    sa = data["classic_sa"]
    nn = data["nn_sa"]

    fig, ax = plt.subplots(figsize=(7, 5))

    ax.scatter([t*1000 for t in sa["all_times"]], sa["all_scores"],
               color=C_SA, s=70, zorder=5, label="SA classique", marker="o")
    ax.scatter([t*1000 for t in nn["all_times"]], nn["all_scores"],
               color=C_NN, s=70, zorder=5, label="SA + NN", marker="s")

    # Ellipses indicatives de dispersion
    for scores, times, color in [
        (sa["all_scores"], [t*1000 for t in sa["all_times"]], C_SA),
        (nn["all_scores"], [t*1000 for t in nn["all_times"]], C_NN),
    ]:
        mx, my = statistics.mean(times), statistics.mean(scores)
        sx = statistics.stdev(times) * 2 if len(times) > 1 else 1
        sy = statistics.stdev(scores) * 2 if len(scores) > 1 else 1
        ellipse = mpatches.Ellipse((mx, my), sx, sy,
                                   fill=False, edgecolor=color,
                                   linewidth=1.2, linestyle="--", alpha=0.5)
        ax.add_patch(ellipse)

    ax.set_xlabel("Durée du run (ms)")
    ax.set_ylabel("Score de puissance (meilleur trouvé)")
    ax.set_title("Qualité de solution vs temps de calcul")
    ax.legend()

    save(fig, "05_correlation_score_temps.png")


# ══════════════════════════════════════════════════════════════════════════════
# GRAPHIQUE 6 — Courbe de perte du réseau de neurones
# ══════════════════════════════════════════════════════════════════════════════

def plot_nn_loss(data: dict):
    tr   = data["nn_training"]
    train = tr["train_losses"]
    val   = tr["val_losses"]
    epochs = list(range(1, len(train) + 1))

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # ── Gauche : courbes train / val ────────────────────────────────────
    ax = axes[0]
    ax.plot(epochs, train, color=C_SA, label="Train MSE", linewidth=2)
    ax.plot(epochs, val,   color=C_NN, label="Val MSE",
            linewidth=2, linestyle="--")

    # Meilleure val
    best_epoch = int(np.argmin(val)) + 1
    best_val   = min(val)
    ax.axvline(best_epoch, color=C_WARN, linewidth=1.2, linestyle=":",
               label=f"Meilleure val (ép. {best_epoch})")
    ax.scatter([best_epoch], [best_val], color=C_WARN, s=60, zorder=6)

    ax.set_title("Courbe de perte — entraînement NN")
    ax.set_xlabel("Époque")
    ax.set_ylabel("MSE")
    ax.legend()

    # ── Droite : métriques finales + texte ─────────────────────────────
    ax2 = axes[1]
    ax2.axis("off")

    r2  = tr["r2"]
    mse = tr["mse"]
    mae = tr["mae"]

    # Tableau récapitulatif
    table_data = [
        ["Métrique", "Valeur", "Seuil cible", "Statut"],
        ["R²",       f"{r2:.3f}",  "≥ 0.70",  "✓ OK" if r2 >= 0.70 else "✗ Insuffisant"],
        ["MSE",      f"{mse:.2f}", "—",        "—"],
        ["MAE",      f"{mae:.2f}", "—",        "—"],
        ["Époques",  str(len(train)), "≤ 50", "✓ OK" if len(train) <= 50 else "Arrêt précoce"],
    ]
    col_widths = [0.22, 0.18, 0.22, 0.28]
    x_starts   = [0.05, 0.27, 0.45, 0.67]
    y_start    = 0.88
    row_h      = 0.14

    for row_idx, row in enumerate(table_data):
        y = y_start - row_idx * row_h
        for col_idx, cell in enumerate(row):
            x = x_starts[col_idx]
            bold = row_idx == 0
            color = "black"
            if row_idx > 0 and col_idx == 3:
                color = C_NN if "✓" in cell else ("#A32D2D" if "✗" in cell else C_GRAY)
            ax2.text(x, y, cell,
                     transform=ax2.transAxes,
                     fontsize=10.5 if bold else 10,
                     fontweight="bold" if bold else "normal",
                     color=color,
                     va="top")
        # Ligne séparatrice après en-tête
        if row_idx == 0:
            ax2.plot([0.03, 0.97], [y - 0.04, y - 0.04],
                         color=C_GRAY, linewidth=0.8,
                         transform=ax2.transAxes, clip_on=False)

    ax2.set_title("Résumé de l'entraînement", fontweight="bold", fontsize=12)

    save(fig, "06_courbe_perte_nn.png")


# ══════════════════════════════════════════════════════════════════════════════
# GRAPHIQUE 7 — Heatmap des gains par run (delta score + delta temps)
# ══════════════════════════════════════════════════════════════════════════════

def plot_heatmap_gains(data: dict):
    sa = data["classic_sa"]
    nn = data["nn_sa"]
    n  = data["config"]["n_runs"]

    delta_scores = [nn["all_scores"][i] - sa["all_scores"][i] for i in range(n)]
    delta_times  = [(nn["all_times"][i] - sa["all_times"][i]) * 1000 for i in range(n)]

    # Matrice 2 × n_runs
    matrix = np.array([delta_scores, delta_times])  # (2, n_runs)

    fig, ax = plt.subplots(figsize=(10, 3.5))

    cmap_score = plt.cm.RdYlGn
    cmap_time  = plt.cm.RdYlGn_r   # inversé : rouge = long, vert = court

    # Normalisation indépendante par ligne
    vmax_score = max(abs(v) for v in delta_scores) + 0.01
    vmax_time  = max(abs(v) for v in delta_times)  + 0.01

    im_data = np.zeros((2, n))
    im_data[0] = np.array(delta_scores)
    im_data[1] = np.array(delta_times)

    # On dessine manuellement case par case pour deux cmaps
    for col in range(n):
        norm_s = (delta_scores[col] + vmax_score) / (2 * vmax_score)
        norm_t = (delta_times[col]  - (-vmax_time)) / (2 * vmax_time)  # plus court = mieux
        color_s = cmap_score(norm_s)
        color_t = cmap_time(norm_t)   # moins de temps = vert
        ax.add_patch(plt.Rectangle((col, 0.5), 1, 1,
                                   facecolor=color_s, edgecolor="white", linewidth=1.5))
        ax.add_patch(plt.Rectangle((col, -0.5), 1, 1,
                                   facecolor=color_t, edgecolor="white", linewidth=1.5))

        # Texte score
        sign_s = "+" if delta_scores[col] >= 0 else ""
        ax.text(col + 0.5, 1.0, f"{sign_s}{delta_scores[col]:.1f}",
                ha="center", va="center", fontsize=10, fontweight="bold",
                color="black")
        # Texte temps
        sign_t = "+" if delta_times[col] >= 0 else ""
        ax.text(col + 0.5, 0.0, f"{sign_t}{delta_times[col]:.0f} ms",
                ha="center", va="center", fontsize=9.5,
                color="black")

    ax.set_xlim(0, n)
    ax.set_ylim(-0.5, 1.5)
    ax.set_xticks([i + 0.5 for i in range(n)])
    ax.set_xticklabels([f"Run {i+1}" for i in range(n)], fontsize=10)
    ax.set_yticks([0.0, 1.0])
    ax.set_yticklabels(["Δ Temps (ms)", "Δ Score"], fontsize=10.5, fontweight="bold")
    ax.set_title("Gains du SA+NN par rapport au SA classique — vert = meilleur, rouge = pire")
    ax.spines[:].set_visible(False)
    ax.tick_params(length=0)
    ax.grid(False)

    save(fig, "07_heatmap_gains.png")


# ══════════════════════════════════════════════════════════════════════════════
# GRAPHIQUE 8 — Radar de synthèse (5 dimensions)
# ══════════════════════════════════════════════════════════════════════════════

def plot_radar(data: dict):
    sa  = data["classic_sa"]
    nn  = data["nn_sa"]
    tr  = data["nn_training"]

    # 5 axes : score moyen, score max, stabilité (1/cv), vitesse (1/temps), qualité NN
    # Tout normalisé sur [0, 1] par rapport au maximum des deux méthodes
    sa_cv = sa["std_score"] / (sa["mean_score"] + 1e-9)
    nn_cv = nn["std_score"] / (nn["mean_score"] + 1e-9)
    max_cv = max(sa_cv, nn_cv) + 1e-9

    max_speed = max(sa["mean_time_s"], nn["mean_time_s"]) + 1e-9

    sa_vals = [
        sa["mean_score"] / (max(sa["mean_score"], nn["mean_score"]) + 1e-9),
        sa["max_score"]  / (max(sa["max_score"],  nn["max_score"])  + 1e-9),
        (1 - sa_cv / max_cv),          # stabilité : plus petit cv = meilleur
        1 - sa["mean_time_s"] / max_speed,  # vitesse : moins de temps = meilleur
        0.5,                            # NN quality n/a pour SA pur
    ]
    nn_vals = [
        nn["mean_score"] / (max(sa["mean_score"], nn["mean_score"]) + 1e-9),
        nn["max_score"]  / (max(sa["max_score"],  nn["max_score"])  + 1e-9),
        (1 - nn_cv / max_cv),
        1 - nn["mean_time_s"] / max_speed,
        min(tr["r2"], 1.0),            # qualité du modèle NN
    ]

    categories = [
        "Score moyen",
        "Score max",
        "Stabilité\n(1 - CV)",
        "Vitesse\n(inverse temps)",
        "Qualité NN\n(R²)",
    ]
    N = len(categories)
    angles = [2 * math.pi * i / N for i in range(N)] + [0]

    sa_vals = sa_vals + [sa_vals[0]]
    nn_vals = nn_vals + [nn_vals[0]]

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))

    ax.plot(angles, sa_vals, color=C_SA, linewidth=2, linestyle="-",
            label="SA classique")
    ax.fill(angles, sa_vals, color=C_SA, alpha=0.18)
    ax.plot(angles, nn_vals, color=C_NN, linewidth=2, linestyle="--",
            label="SA + NN")
    ax.fill(angles, nn_vals, color=C_NN, alpha=0.18)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=10)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0.25", "0.50", "0.75", "1.00"], fontsize=8, color=C_GRAY)
    ax.set_title("Synthèse multi-critères\n(normalisé [0,1], plus grand = meilleur)",
                 fontsize=13, fontweight="bold", pad=20)
    ax.legend(loc="lower right", bbox_to_anchor=(1.25, -0.05))

    save(fig, "08_radar_synthese.png")


# ══════════════════════════════════════════════════════════════════════════════
# GRAPHIQUE BONUS — Panneau synthèse 2×2 (résumé exécutif)
# ══════════════════════════════════════════════════════════════════════════════

def plot_summary_panel(data: dict):
    """Vue d'ensemble 2×2 pour le rapport."""
    sa = data["classic_sa"]
    nn = data["nn_sa"]

    fig = plt.figure(figsize=(14, 10))
    gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.4, wspace=0.35)

    # ── A: Convergence ───────────────────────────────────────────────────
    ax_a = fig.add_subplot(gs[0, 0])
    steps    = len(sa["conv_curves"][0])
    x_labels = [(i+1)*50 for i in range(steps)]
    sa_curves = np.array(sa["conv_curves"])
    nn_curves = np.array(nn["conv_curves"])
    sa_mean = sa_curves.mean(axis=0)
    nn_mean = nn_curves.mean(axis=0)
    sa_std  = sa_curves.std(axis=0)
    nn_std  = nn_curves.std(axis=0)

    ax_a.fill_between(x_labels, sa_mean - sa_std, sa_mean + sa_std, alpha=0.15, color=C_SA)
    ax_a.fill_between(x_labels, nn_mean - nn_std, nn_mean + nn_std, alpha=0.15, color=C_NN)
    ax_a.plot(x_labels, sa_mean, color=C_SA, linewidth=2)
    ax_a.plot(x_labels, nn_mean, color=C_NN, linewidth=2, linestyle="--")
    ax_a.set_title("A — Convergence moyenne", loc="left")
    ax_a.set_xlabel("Itérations")
    ax_a.set_ylabel("Score")
    ax_a.legend(["SA classique", "SA + NN"], fontsize=9)

    # ── B: Boxplot ───────────────────────────────────────────────────────
    ax_b = fig.add_subplot(gs[0, 1])
    bp = ax_b.boxplot([sa["all_scores"], nn["all_scores"]],
                      tick_labels=["SA", "SA+NN"], patch_artist=True,
                      widths=0.45, medianprops=dict(color="white", lw=2))
    bp["boxes"][0].set(facecolor=C_SA_L, edgecolor=C_SA)
    bp["boxes"][1].set(facecolor=C_NN_L, edgecolor=C_NN)
    ax_b.set_title("B — Distribution des scores", loc="left")
    ax_b.set_ylabel("Score (meilleur run)")

    # ── C: Scores & déltas ───────────────────────────────────────────────
    ax_c = fig.add_subplot(gs[1, 0])
    n = data["config"]["n_runs"]
    x = np.arange(n)
    w = 0.36
    ax_c.bar(x - w/2, sa["all_scores"], w, color=C_SA_L, edgecolor=C_SA, lw=0.8)
    ax_c.bar(x + w/2, nn["all_scores"], w, color=C_NN_L, edgecolor=C_NN, lw=0.8)
    ax_c.set_title("C — Scores par run", loc="left")
    ax_c.set_xticks(x)
    ax_c.set_xticklabels([f"R{i+1}" for i in range(n)], fontsize=9)
    ax_c.set_ylabel("Score")
    ax_c.legend(["SA classique", "SA + NN"], fontsize=9)

    # ── D: Temps ─────────────────────────────────────────────────────────
    ax_d = fig.add_subplot(gs[1, 1])
    sa_ms = [t*1000 for t in sa["all_times"]]
    nn_ms = [t*1000 for t in nn["all_times"]]
    ax_d.bar(x - w/2, sa_ms, w, color=C_SA_L, edgecolor=C_SA, lw=0.8)
    ax_d.bar(x + w/2, nn_ms, w, color=C_NN_L, edgecolor=C_NN, lw=0.8)
    ax_d.set_title("D — Temps d'exécution (ms)", loc="left")
    ax_d.set_xticks(x)
    ax_d.set_xticklabels([f"R{i+1}" for i in range(n)], fontsize=9)
    ax_d.set_ylabel("Durée (ms)")
    ax_d.legend(["SA classique", "SA + NN"], fontsize=9)

    delta_ms = data["delta"]["time_overhead_s"] * 1000
    gain_pct = data["delta"]["score_gain_pct"]
    fig.suptitle(
        f"Synthèse benchmark — Gain score : +{gain_pct:.1f}%  |  "
        f"Surcoût temps : +{delta_ms:.0f} ms / run",
        fontsize=13, fontweight="bold"
    )

    save(fig, "00_synthese_2x2.png")


# ══════════════════════════════════════════════════════════════════════════════
# POINT D'ENTRÉE
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 55)
    print("  plot_benchmark.py — Génération des graphiques")
    print("=" * 55)

    data = load_data("out/benchmark_results.json")

    print(f"\n[Config] {data['config']['n_runs']} runs × "
          f"{data['config']['n_moves']} itérations, "
          f"k={data['config']['k_candidates']}")

    plots = [
        ("Panneau synthèse 2×2",               plot_summary_panel),
        ("Convergence moyenne",                 plot_convergence),
        ("Boxplot des scores",                  plot_boxplot),
        ("Scores par run",                      plot_scores_per_run),
        ("Temps d'exécution",                   plot_times),
        ("Corrélation score / temps",           plot_score_time_scatter),
        ("Courbe de perte NN",                  plot_nn_loss),
        ("Heatmap des gains",                   plot_heatmap_gains),
        ("Radar de synthèse",                   plot_radar),
    ]

    print(f"\n[Génération] {len(plots)} graphiques → dossier '{OUTPUT_DIR}/'\n")
    for name, fn in plots:
        print(f"  {name}...")
        fn(data)

    print(f"\n[✓] {len(plots)} fichiers PNG dans '{OUTPUT_DIR}/'")
    files = sorted(OUTPUT_DIR.glob("*.png"))
    for f in files:
        size_kb = f.stat().st_size // 1024
        print(f"    {f.name}  ({size_kb} Ko)")


if __name__ == "__main__":
    main()