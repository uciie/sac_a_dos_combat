"""
=====================================================================
  plot_benchmark_ga.py - Génération des graphiques pour l'AG
  Backpack Battles - M1 MIAGE
=====================================================================
  Lit les résultats dans out/benchmark_ga_results.json et produit 
  les visualisations dans le dossier plots/.
=====================================================================
"""

import json
import os
import numpy as np
import matplotlib.pyplot as plt

# ── Configuration visuelle ────────────────────────────────────────────────────
plt.style.use('seaborn-v0_8-whitegrid')
COLORS = {'classic': '#1f77b4', 'ml': '#ff7f0e'}  # Bleu et Orange
LABELS = {'classic': 'AG Classique', 'ml': 'AG + ML (Surrogate)'}

def load_data(filepath="out/benchmark_ga_results.json"):
    """Charge le fichier JSON contenant les résultats du benchmark."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Erreur : Le fichier {filepath} est introuvable.")
        print("Veuillez d'abord exécuter 'python src/benchmark_ga.py'.")
        exit(1)

def ensure_plot_dir(dir_name="plots"):
    """Crée le dossier de destination pour les graphiques s'il n'existe pas."""
    os.makedirs(dir_name, exist_ok=True)

# ── Fonctions de tracé ────────────────────────────────────────────────────────

def plot_convergence(data, out_path="plots/ga_01_convergence.png"):
    """Trace la courbe de convergence moyenne au fil des générations."""
    plt.figure(figsize=(10, 6))
    
    for key in ['classic_ga', 'ml_ga']:
        color_key = 'classic' if key == 'classic_ga' else 'ml'
        curves = data[key]['conv_curves']
        
        # Convertir en tableau numpy pour calculer la moyenne et l'écart-type facilement
        arr = np.array(curves)
        mean_curve = np.mean(arr, axis=0)
        std_curve = np.std(arr, axis=0)
        generations = np.arange(len(mean_curve))
        
        # Tracer la ligne moyenne
        plt.plot(generations, mean_curve, label=LABELS[color_key], color=COLORS[color_key], linewidth=2)
        # Ajouter une zone ombrée pour l'écart-type
        plt.fill_between(generations, mean_curve - std_curve, mean_curve + std_curve, 
                         color=COLORS[color_key], alpha=0.2)

    plt.title("Convergence moyenne : AG Classique vs AG + ML", fontsize=14, fontweight='bold')
    plt.xlabel("Générations", fontsize=12)
    plt.ylabel("Score (Puissance de Combat)", fontsize=12)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"  -> Sauvegardé : {out_path}")

def plot_score_boxplot(data, out_path="plots/ga_02_boxplot_scores.png"):
    """Trace un diagramme en boîte des scores finaux."""
    plt.figure(figsize=(8, 6))
    
    scores_data = [data['classic_ga']['all_scores'], data['ml_ga']['all_scores']]
    
    box = plt.boxplot(scores_data, labels=[LABELS['classic'], LABELS['ml']], 
                      patch_artist=True, widths=0.5)
    
    # Colorier les boîtes
    colors = [COLORS['classic'], COLORS['ml']]
    for patch, color in zip(box['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    # Styling des médianes
    for median in box['medians']:
        median.set(color='black', linewidth=2)

    plt.title("Distribution des meilleurs scores finaux", fontsize=14, fontweight='bold')
    plt.ylabel("Score total", fontsize=12)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"  -> Sauvegardé : {out_path}")

def plot_time_execution(data, out_path="plots/ga_03_temps_execution.png"):
    """Trace un diagramme en boîte des temps d'exécution."""
    plt.figure(figsize=(8, 6))
    
    times_data = [data['classic_ga']['all_times'], data['ml_ga']['all_times']]
    
    box = plt.boxplot(times_data, labels=[LABELS['classic'], LABELS['ml']], 
                      patch_artist=True, widths=0.5)
    
    colors = [COLORS['classic'], COLORS['ml']]
    for patch, color in zip(box['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    for median in box['medians']:
        median.set(color='black', linewidth=2)

    plt.title("Temps d'exécution par exécution (Run)", fontsize=14, fontweight='bold')
    plt.ylabel("Temps (secondes)", fontsize=12)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"  -> Sauvegardé : {out_path}")

def plot_scatter_time_score(data, out_path="plots/ga_04_correlation_score_temps.png"):
    """Trace un nuage de points comparant le score et le temps pour chaque run."""
    plt.figure(figsize=(10, 6))
    
    plt.scatter(data['classic_ga']['all_times'], data['classic_ga']['all_scores'], 
                color=COLORS['classic'], label=LABELS['classic'], s=100, alpha=0.8, edgecolors='w')
    
    plt.scatter(data['ml_ga']['all_times'], data['ml_ga']['all_scores'], 
                color=COLORS['ml'], label=LABELS['ml'], s=100, alpha=0.8, edgecolors='w')

    plt.title("Compromis Temps vs Score", fontsize=14, fontweight='bold')
    plt.xlabel("Temps d'exécution (secondes)", fontsize=12)
    plt.ylabel("Score final", fontsize=12)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"  -> Sauvegardé : {out_path}")

# ==============================================================================
# PIPELINE PRINCIPAL
# ==============================================================================

def main():
    print("=" * 60)
    print("  GÉNÉRATION DES GRAPHIQUES - BENCHMARK AG")
    print("=" * 60)
    
    ensure_plot_dir("plots")
    
    print("\nChargement des données...")
    data = load_data("out/benchmark_ga_results.json")
    
    print("\nCréation des visualisations...")
    plot_convergence(data)
    plot_score_boxplot(data)
    plot_time_execution(data)
    plot_scatter_time_score(data)
    
    print("\n" + "=" * 60)
    print("  Terminé ! Tous les graphiques sont dans le dossier 'plots/'.")
    print("  Vous pouvez les utiliser pour la partie Expérimentations de votre rapport.")
    print("=" * 60)

if __name__ == "__main__":
    main()