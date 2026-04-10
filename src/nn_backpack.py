"""
=====================================================================
  nn_backpack.py - Réseau de Neurones pour Backpack Battles
  M1 MIAGE - Optimisation Combinatoire

  Ce module implémente :
    1. GridEncoder      -> Transforme un BackpackManager en tenseur
    2. BackpackScoreNet -> Architecture CNN + Dense (PyTorch)
    3. ExperienceBuffer -> Buffer de replay pour l'apprentissage supervisé
    4. NNTrainer        -> Boucle d'entraînement + évaluation
    5. NNGuidedSA       -> Recuit Simulé avec heuristique neuronale

  Usage rapide :
    encoder = GridEncoder()
    model   = BackpackScoreNet(encoder.INPUT_CHANNELS)
    trainer = NNTrainer(model, encoder)
    sa      = NNGuidedSA(manager, items, model, encoder)
=====================================================================
"""

from __future__ import annotations
import copy
import math
import random
import os
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# Import local du projet Backpack Battles
from models import (
    BackpackManager, Item, GRID_SIZE, SYNERGY_TABLE, ITEM_CATALOGUE
)


# ==============================================================================
# 0. CONSTANTES
# ==============================================================================

# Liste ordonnée et stable de tous les tags présents dans le jeu
ALL_TAGS: list[str] = sorted({
    tag
    for item_def in ITEM_CATALOGUE
    for tag in item_def.get("tags", [])
})

# Nombre de canaux du tenseur d'entrée :
#   Canal 0    : masque des cases occupées (0/1)
#   Canal 1    : puissance normalisée de l'item sur chaque case
#   Canal 2    : masque des containers actifs (zones autorisées)
#   Canal 3..N : one-hot encodage de chaque tag (len(ALL_TAGS) canaux)
N_TAG_CHANNELS: int = len(ALL_TAGS)
BASE_CHANNELS:  int = 3   # occupé, puissance, container


# ==============================================================================
# 1. ENCODEUR (Pré-traitement)
# ==============================================================================

class GridEncoder:
    """
    Transforme un BackpackManager en tenseur PyTorch prêt pour le réseau.

    Sortie : tenseur de forme  (C, GRID_SIZE, GRID_SIZE)
    avec C = BASE_CHANNELS + N_TAG_CHANNELS

    Canal 0 — occupancy  : 1.0 si la case contient un item, 0.0 sinon
    Canal 1 — power_map  : puissance_base normalisée (divisée par MAX_POWER)
    Canal 2 — container  : 1.0 si la case appartient à un container acheté
    Canal 3+ — tag_i     : 1.0 si l'item sur cette case possède le tag i

    Exemple :
        encoder = GridEncoder()
        tensor  = encoder.encode(manager)   # shape (C, 10, 10)
    """

    # Valeur de normalisation : puissance max observable dans le catalogue
    MAX_POWER: float = max(d["puissance"] for d in ITEM_CATALOGUE) + 1e-6

    @property
    def INPUT_CHANNELS(self) -> int:
        return BASE_CHANNELS + N_TAG_CHANNELS

    def encode(self, manager: BackpackManager) -> torch.Tensor:
        """
        Encode l'état courant de manager en tenseur (C, H, W).

        :param manager: BackpackManager - état de jeu courant
        :return: torch.FloatTensor de forme (INPUT_CHANNELS, GRID_SIZE, GRID_SIZE)
        """
        C = self.INPUT_CHANNELS
        tensor = np.zeros((C, GRID_SIZE, GRID_SIZE), dtype=np.float32)

        # -- Canal 2 : zones autorisées par les containers --------------
        tensor[2] = (manager.container_grid > 0).astype(np.float32)

        # -- Canaux 0, 1, 3+ : informations par item placé --------------
        for item_id, (item, x, y) in manager.items_placed.items():
            shape  = item.get_rotated_shape()
            power  = item.puissance_base / self.MAX_POWER
            tag_mask = self._tag_vector(item.tags)

            for dr in range(shape.shape[0]):
                for dc in range(shape.shape[1]):
                    if shape[dr, dc] == 1:
                        r, c = y + dr, x + dc
                        if 0 <= r < GRID_SIZE and 0 <= c < GRID_SIZE:
                            tensor[0, r, c] = 1.0         # occupancy
                            tensor[1, r, c] = power        # puissance normalisée
                            # canaux tags
                            for t_idx, val in enumerate(tag_mask):
                                tensor[BASE_CHANNELS + t_idx, r, c] = val

        return torch.from_numpy(tensor)  # (C, H, W) float32

    def _tag_vector(self, tags: list[str]) -> list[float]:
        """
        Encode une liste de tags en vecteur binaire de longueur N_TAG_CHANNELS.
        """
        vec = [0.0] * N_TAG_CHANNELS
        for tag in tags:
            if tag in ALL_TAGS:
                vec[ALL_TAGS.index(tag)] = 1.0
        return vec

    def encode_batch(self, managers: list[BackpackManager]) -> torch.Tensor:
        """
        Encode une liste de managers en batch : (B, C, H, W).
        Pratique pour l'inférence groupée dans le SA guidé.
        """
        tensors = [self.encode(m) for m in managers]
        return torch.stack(tensors)   # (B, C, H, W)


# ==============================================================================
# 2. ARCHITECTURE DU RÉSEAU
# ==============================================================================

class BackpackScoreNet(nn.Module):
    """
    Réseau de neurones hybride CNN + MLP pour prédire le score d'une
    configuration de sac à dos.

    Architecture :
        ┌-----------------------------------------------------┐
        │ Entrée : (C, 10, 10) - C canaux encodés             │
        ├--------------------------------┐                    │
        │ Conv2D(C, 32, 3*3, pad=1)      │ → (32, 10, 10)     │
        │ BatchNorm2d(32) + ReLU         │                    │
        │ Conv2D(32, 64, 3*3, pad=1)     │ → (64, 10, 10)     │
        │ BatchNorm2d(64) + ReLU         │                    │
        │ Conv2D(64, 64, 3*3, pad=1)     │ → (64, 10, 10)     │
        │ BatchNorm2d(64) + ReLU         │                    │
        │ AdaptiveAvgPool2d(4, 4)        │ → (64, 4, 4) = 1024│
        ├--------------------------------┘                    │
        │ Flatten → 1024                                       │
        │ Linear(1024, 256) + ReLU                            │
        │ Dropout(0.3)                                        │
        │ Linear(256, 64) + ReLU                              │
        │ Linear(64, 1) → score scalaire                      │
        └-----------------------------------------------------┘

    Note : Softplus en sortie pour garantir un score ≥ 0.
    """

    def __init__(self, in_channels: int):
        super().__init__()

        # -- Bloc convolutif ----------------------------------------------
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.AdaptiveAvgPool2d((4, 4)),   # → (64, 4, 4) = 1024 features
        )

        # -- Tête de régression -----------------------------------------
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
            nn.Softplus(),   # garantit score ≥ 0
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param x: (B, C, H, W) float32
        :return:  (B, 1) float32 - scores prédits
        """
        features = self.conv_block(x)
        score    = self.head(features)
        return score

    def predict_score(self, tensor: torch.Tensor, device: str = "cpu") -> float:
        """
        Prédit le score d'un unique tenseur (C, H, W).
        Pratique pour l'intégration dans le Recuit Simulé.
        """
        self.eval()
        with torch.no_grad():
            x = tensor.unsqueeze(0).to(device)  # (1, C, H, W)
            return self.forward(x).item()


# ==============================================================================
# 3. DATASET ET BUFFER D'EXPÉRIENCE
# ==============================================================================

class ExperienceBuffer(Dataset):
    """
    Buffer de type "Replay Memory" qui stocke les transitions
    (état_grille, score_réel) générées par le Recuit Simulé.

    Chaque entrée est un couple :
        tensor : (C, H, W) float32  - encodage de l'état
        score  : float              - score calculé exactement

    Exemples d'utilisation :
        buffer = ExperienceBuffer(max_size=10_000)
        buffer.push(encoder.encode(manager), exact_score)
        loader = DataLoader(buffer, batch_size=64, shuffle=True)
    """

    def __init__(self, max_size: int = 10_000):
        self.max_size = max_size
        self._tensors: list[torch.Tensor] = []
        self._scores:  list[float]        = []

    def push(self, tensor: torch.Tensor, score: float):
        """Ajoute une paire (état, score). Éjecte les plus anciens si plein."""
        if len(self._tensors) >= self.max_size:
            self._tensors.pop(0)
            self._scores.pop(0)
        self._tensors.append(tensor.clone())
        self._scores.append(float(score))

    def push_batch(self, tensors: list[torch.Tensor], scores: list[float]):
        for t, s in zip(tensors, scores):
            self.push(t, s)

    def __len__(self) -> int:
        return len(self._tensors)

    def __getitem__(self, idx: int):
        return self._tensors[idx], torch.tensor(self._scores[idx], dtype=torch.float32)

    def save(self, path: str):
        """Sauvegarde le buffer sur disque (tenseurs + scores)."""
        torch.save({"tensors": self._tensors, "scores": self._scores}, path)
        print(f"[Buffer] {len(self)} expériences sauvegardées → {path}")

    @classmethod
    def load(cls, path: str, max_size: int = 10_000) -> "ExperienceBuffer":
        """Recharge un buffer sauvegardé."""
        buf  = cls(max_size=max_size)
        data = torch.load(path, weights_only=False)
        buf._tensors = data["tensors"]
        buf._scores  = data["scores"]
        print(f"[Buffer] {len(buf)} expériences chargées depuis {path}")
        return buf


# ==============================================================================
# 4. GÉNÉRATEUR DE DONNÉES (depuis le Recuit Simulé)
# ==============================================================================

def generate_training_data(
    manager_template: BackpackManager,
    items_available: list[Item],
    encoder: GridEncoder,
    n_configs: int = 500,
    sa_steps_per_config: int = 200,
) -> ExperienceBuffer:
    """
    Génère un dataset d'entraînement en faisant tourner des sessions
    aléatoires de Recuit Simulé et en enregistrant chaque état + score exact.

    Stratégie :
      - Pour chaque config, on clone le manager de base,
        on exécute des mouvements aléatoires,
        on calcule le score réel et on stocke la paire.

    :param manager_template: BackpackManager de référence (containers achetés)
    :param items_available:  Liste d'items disponibles pour le SA
    :param encoder:          GridEncoder
    :param n_configs:        Nombre de configurations à générer
    :param sa_steps_per_config: Nombre de mouvements aléatoires par config
    :return: ExperienceBuffer rempli
    """
    from models import SimulatedAnnealing

    buffer = ExperienceBuffer(max_size=n_configs * 3)
    print(f"[DataGen] Génération de {n_configs} configurations...")

    for config_idx in range(n_configs):
        # Clone indépendant pour chaque run
        mgr_copy   = manager_template.clone()
        items_copy = [copy.deepcopy(it) for it in items_available]

        sa = SimulatedAnnealing(mgr_copy, items_copy)

        # Plusieurs checkpoints par run pour maximiser la diversité
        checkpoints = max(1, sa_steps_per_config // 50)
        for _ in range(checkpoints):
            sa.step(n_moves=50)
            score_data = mgr_copy.calculate_score()
            tensor     = encoder.encode(mgr_copy)
            buffer.push(tensor, score_data["total"])

        if (config_idx + 1) % 50 == 0:
            print(f"  [{config_idx+1}/{n_configs}] buffer size = {len(buffer)}")

    print(f"[DataGen] Terminé - {len(buffer)} exemples collectés.")
    return buffer


# ==============================================================================
# 5. ENTRAÎNEMENT
# ==============================================================================

class NNTrainer:
    """
    Gère l'entraînement supervisé de BackpackScoreNet.

    Perte : MSELoss  (régression de score)
    Optimiseur : AdamW avec scheduler cosine
    Validation : split 80/20 automatique

    Exemple :
        trainer = NNTrainer(model, encoder)
        trainer.train(buffer, epochs=50)
        trainer.save_model("backpack_nn.pt")
    """

    def __init__(
        self,
        model:   BackpackScoreNet,
        encoder: GridEncoder,
        lr:      float = 1e-3,
        device:  str   = "cpu",
    ):
        self.model   = model.to(device)
        self.encoder = encoder
        self.device  = device
        self.criterion = nn.MSELoss()
        self.optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        self.train_losses: list[float] = []
        self.val_losses:   list[float] = []

    def train(
        self,
        buffer:     ExperienceBuffer,
        epochs:     int   = 50,
        batch_size: int   = 64,
        val_split:  float = 0.2,
        patience:   int   = 10,
    ):
        """
        Boucle d'entraînement complète avec early stopping.

        :param buffer:     ExperienceBuffer avec les exemples (état, score)
        :param epochs:     Nombre maximum d'époques
        :param batch_size: Taille des mini-batches
        :param val_split:  Proportion du buffer réservée à la validation
        :param patience:   Nombre d'époques sans amélioration avant arrêt
        """
        # -- Split train / validation ------------------------------------
        n_total = len(buffer)
        n_val   = int(n_total * val_split)
        n_train = n_total - n_val
        train_set, val_set = torch.utils.data.random_split(
            buffer, [n_train, n_val],
            generator=torch.Generator().manual_seed(42)
        )

        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                                  num_workers=0, pin_memory=False)
        val_loader   = DataLoader(val_set,   batch_size=batch_size, shuffle=False,
                                  num_workers=0, pin_memory=False)

        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=epochs, eta_min=1e-5
        )

        best_val_loss  = float("inf")
        patience_count = 0

        print(f"\n[Train] {n_train} train / {n_val} val | {epochs} epochs max")
        print("-" * 55)

        for epoch in range(1, epochs + 1):
            # -- Phase d'entraînement ------------------------------------
            self.model.train()
            train_loss = 0.0
            for tensors, scores in train_loader:
                tensors = tensors.to(self.device)
                scores  = scores.to(self.device).unsqueeze(1)   # (B, 1)

                self.optimizer.zero_grad()
                preds = self.model(tensors)
                loss  = self.criterion(preds, scores)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
                train_loss += loss.item() * len(tensors)

            train_loss /= n_train

            # -- Phase de validation -------------------------------------
            self.model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for tensors, scores in val_loader:
                    tensors = tensors.to(self.device)
                    scores  = scores.to(self.device).unsqueeze(1)
                    preds   = self.model(tensors)
                    val_loss += self.criterion(preds, scores).item() * len(tensors)
            val_loss /= n_val

            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            scheduler.step()

            # -- Affichage et early stopping -----------------------------
            if epoch % 5 == 0 or epoch == 1:
                lr_now = self.optimizer.param_groups[0]["lr"]
                print(f"  Époque {epoch:3d}/{epochs} | "
                      f"Train MSE={train_loss:8.3f} | "
                      f"Val MSE={val_loss:8.3f} | "
                      f"LR={lr_now:.2e}")

            if val_loss < best_val_loss:
                best_val_loss  = val_loss
                patience_count = 0
                # Sauvegarder le meilleur poids en mémoire
                self._best_weights = copy.deepcopy(self.model.state_dict())
            else:
                patience_count += 1
                if patience_count >= patience:
                    print(f"\n[Train] Early stopping à l'époque {epoch} "
                          f"(patience={patience})")
                    break

        # Restaurer les meilleurs poids
        if hasattr(self, "_best_weights"):
            self.model.load_state_dict(self._best_weights)
            print(f"[Train] Meilleurs poids restaurés (val MSE={best_val_loss:.3f})")

    def evaluate(self, buffer: ExperienceBuffer, batch_size: int = 64) -> dict:
        """
        Calcule MSE, MAE et R² sur tout le buffer.

        :return: dict avec clés mse, mae, r2
        """
        loader = DataLoader(buffer, batch_size=batch_size, shuffle=False)
        all_preds, all_targets = [], []

        self.model.eval()
        with torch.no_grad():
            for tensors, scores in loader:
                preds = self.model(tensors.to(self.device)).squeeze(1).cpu()
                all_preds.append(preds)
                all_targets.append(scores)

        preds   = torch.cat(all_preds)
        targets = torch.cat(all_targets)

        mse = nn.MSELoss()(preds, targets).item()
        mae = nn.L1Loss()(preds, targets).item()

        ss_res = ((targets - preds) ** 2).sum().item()
        ss_tot = ((targets - targets.mean()) ** 2).sum().item()
        r2     = 1.0 - ss_res / (ss_tot + 1e-8)

        print(f"[Eval] MSE={mse:.3f} | MAE={mae:.3f} | R²={r2:.4f}")
        return {"mse": mse, "mae": mae, "r2": r2}

    def save_model(self, path: str):
        """Sauvegarde le modèle et les métadonnées d'entraînement."""
        torch.save({
            "model_state":   self.model.state_dict(),
            "in_channels":   self.model.conv_block[0].in_channels,
            "train_losses":  self.train_losses,
            "val_losses":    self.val_losses,
            "all_tags":      ALL_TAGS,
        }, path)
        print(f"[Trainer] Modèle sauvegardé → {path}")

    @staticmethod
    def load_model(path: str, device: str = "cpu") -> BackpackScoreNet:
        """Recharge un modèle sauvegardé."""
        ckpt = torch.load(path, map_location=device, weights_only=False)
        model = BackpackScoreNet(in_channels=ckpt["in_channels"])
        model.load_state_dict(ckpt["model_state"])
        model.to(device)
        model.eval()
        print(f"[Trainer] Modèle chargé depuis {path}")
        return model


# ==============================================================================
# 6. RECUIT SIMULÉ GUIDÉ PAR LE RÉSEAU
# ==============================================================================

class NNGuidedSA:
    """
    Recuit Simulé où la sélection des candidats utilise le réseau de neurones
    comme heuristique préliminaire (filtre rapide), et le calcul exact ne sert
    qu'au critère de Metropolis final.

    Stratégie "Top-K Scoring" :
      1. À chaque itération, générer K candidats voisins (états grilles)
      2. Prédire leur score avec le réseau (rapide, pas de synergies)
      3. Ne garder que le meilleur candidat prédit
      4. Appliquer le critère de Metropolis avec le SCORE EXACT (fiable)

    L'effet est de concentrer les évaluations exactes sur les mouvements
    à fort potentiel, réduisant le nombre d'explorations non-prometteuses.

    Paramètres :
        manager         : BackpackManager courant
        items_available : Items non encore placés
        model           : BackpackScoreNet entraîné
        encoder         : GridEncoder
        T0              : Température initiale
        alpha           : Coefficient de refroidissement
        k_candidates    : Nombre de candidats évalués par le réseau par étape
    """

    def __init__(
        self,
        manager:         BackpackManager,
        items_available: list[Item],
        model:           BackpackScoreNet,
        encoder:         GridEncoder,
        T0:              float = 1000.0,
        alpha:           float = 0.995,
        k_candidates:    int   = 5,
        device:          str   = "cpu",
    ):
        self.manager   = manager
        self.available = list(items_available)
        self.model     = model.eval()
        self.encoder   = encoder
        self.T         = T0
        self.alpha     = alpha
        self.k         = k_candidates
        self.device    = device

        self.iteration  = 0
        self.best_score = manager.calculate_score()["total"]
        self.best_state = manager.clone()
        self.history:   list[dict] = []

    # -- Interface principale -------------------------------------------------

    def step(self, n_moves: int = 50) -> dict:
        """
        Effectue n_moves itérations guidées et retourne un snapshot.
        Compatible avec l'interface de SimulatedAnnealing.step().
        """
        current_score = self.manager.calculate_score()["total"]

        for _ in range(n_moves):
            delta = self._guided_step(current_score)
            current_score += delta

            # Mettre à jour le meilleur score et état si nécessaire
            if current_score > self.best_score:
                self.best_score = current_score
                self.best_state = self.manager.clone()
            
            self.T *= self.alpha
            self.iteration += 1

        score_data = self.manager.calculate_score()
        snapshot = {
            "iteration":     self.iteration,
            "temperature":   round(self.T, 2),
            "score":         score_data["total"],
            "best_score":    round(self.best_score, 2),
            "base_power":    score_data["base_power"],
            "synergy_bonus": score_data["synergy_bonus"],
        }
        self.history.append(snapshot)
        return snapshot

    def restore_best(self):
        """Restaure le meilleur état trouvé."""
        self.manager.grid            = self.best_state.grid.copy()
        self.manager.container_grid  = self.best_state.container_grid.copy()
        self.manager.items_placed    = copy.deepcopy(self.best_state.items_placed)
        self.manager.containers_owned = copy.deepcopy(self.best_state.containers_owned)

    # -- Logique interne ------------------------------------------------------

    def _guided_step(self, current_score: float) -> float:
        """
        Génère K candidats, les prédit avec le réseau,
        sélectionne le meilleur, puis applique Metropolis avec score exact.
        """
        ops = ["MOVE", "ROTATE", "SWAP", "ADD", "REMOVE"]
        candidates: list[tuple[BackpackManager, list[Item], str]] = []

        # -- Étape 1 : générer K états candidats ----------------------
        for _ in range(self.k):
            op      = random.choice(ops)
            mgr_try = self.manager.clone()
            avl_try = list(self.available)
            success = self._apply_op(mgr_try, avl_try, op)
            if success:
                candidates.append((mgr_try, avl_try, op))

        if not candidates:
            return 0.0

        # -- Étape 2 : prédire les scores via le réseau ----------------
        if len(candidates) == 1:
            best_candidate = candidates[0]
        else:
            with torch.no_grad():
                tensors = torch.stack([
                    self.encoder.encode(c[0]) for c in candidates
                ]).to(self.device)                              # (K, C, H, W)
                preds = self.model(tensors).squeeze(1).cpu()   # (K,)
                best_idx       = int(preds.argmax().item())
                best_candidate = candidates[best_idx]

        best_mgr, best_avl, _ = best_candidate

        # -- Étape 3 : score exact du meilleur candidat ----------------
        new_score = best_mgr.calculate_score()["total"]
        delta     = new_score - current_score

        # -- Étape 4 : critère de Metropolis ---------------------------
        if delta >= 0 or random.random() < math.exp(delta / max(self.T, 0.01)):
            # Accepter le mouvement → mettre à jour l'état principal
            self.manager.grid             = best_mgr.grid.copy()
            self.manager.container_grid   = best_mgr.container_grid.copy()
            self.manager.items_placed     = copy.deepcopy(best_mgr.items_placed)
            self.manager.containers_owned = copy.deepcopy(best_mgr.containers_owned)
            self.available = best_avl
            return delta
        else:
            return 0.0

    @staticmethod
    def _apply_op(
        mgr:       BackpackManager,
        available: list[Item],
        op:        str,
    ) -> bool:
        """
        Applique un opérateur de voisinage sur une COPIE (mgr, available).
        Retourne True si l'opération a pu être effectuée.
        """
        if op == "MOVE" and mgr.items_placed:
            iid = random.choice(list(mgr.items_placed.keys()))
            item, *_ = mgr.items_placed[iid]
            mgr.remove_item(iid)
            pos = mgr.valid_positions(item)
            if pos:
                mgr.place_item(item, *random.choice(pos))
                return True

        elif op == "ROTATE" and mgr.items_placed:
            iid = random.choice(list(mgr.items_placed.keys()))
            item, *_ = mgr.items_placed[iid]
            mgr.remove_item(iid)
            rotated = item.rotate_cw()
            pos = mgr.valid_positions(rotated)
            if pos:
                mgr.place_item(rotated, *random.choice(pos))
                return True

        elif op == "SWAP" and len(mgr.items_placed) >= 2:
            ids = random.sample(list(mgr.items_placed.keys()), 2)
            item_a, xa, ya = mgr.items_placed[ids[0]]
            item_b, xb, yb = mgr.items_placed[ids[1]]
            mgr.remove_item(ids[0])
            mgr.remove_item(ids[1])
            ok_a = mgr.place_item(item_a, xb, yb)
            ok_b = mgr.place_item(item_b, xa, ya)
            if ok_a and ok_b:
                return True
            # Rollback du swap partiel
            if ok_a:  mgr.remove_item(ids[0])
            if ok_b:  mgr.remove_item(ids[1])
            mgr.place_item(item_a, xa, ya)
            mgr.place_item(item_b, xb, yb)
            return False

        elif op == "ADD" and available:
            item = random.choice(available)
            pos  = mgr.valid_positions(item)
            if pos:
                mgr.place_item(item, *random.choice(pos))
                available.remove(item)
                return True

        elif op == "REMOVE" and mgr.items_placed:
            iid = random.choice(list(mgr.items_placed.keys()))
            item, *_ = mgr.items_placed[iid]
            mgr.remove_item(iid)
            available.append(item)
            return True

        return False


# ==============================================================================
# 7. POINT D'ENTRÉE : DEMO / ENTRAÎNEMENT AUTONOME
# ==============================================================================

def demo_pipeline():
    """
    Démonstration complète du pipeline :
      1. Créer un manager de test avec quelques containers et items
      2. Générer des données d'entraînement via le SA classique
      3. Entraîner le réseau
      4. Comparer SA classique vs SA guidé par le réseau
    """
    print("=" * 60)
    print("  BACKPACK BATTLES — Pipeline Réseau de Neurones")
    print("=" * 60)

    from models import (
        Container, Item, CONTAINER_CATALOGUE, ITEM_CATALOGUE, SimulatedAnnealing
    )

    # -- 1. Préparer l'environnement de jeu --------------------------
    print("\n[Setup] Création du manager de test...")
    mgr = BackpackManager()

    # Ajouter 2 containers (Sac de Voyage 3*3 et Ceint uron 3*2)
    for  cat in CONTAINER_CATALOGUE[:2]:
        c = Container(id=0, nom=cat["nom"], prix=cat["prix"],
                      largeur=cat["largeur"], hauteur=cat["hauteur"])
        mgr.add_container(c)

    # Créer 6 items variés
    items_pool = [
        Item.from_catalogue(ITEM_CATALOGUE[i], item_id=i+1)
        for i in range(6)
    ]

    # -- 2. Générer le dataset ----------------------------------------
    print("\n[DataGen] Génération du dataset d'entraînement...")
    encoder = GridEncoder()
    buffer  = generate_training_data(
        manager_template=mgr,
        items_available=items_pool,
        encoder=encoder,
        n_configs=100,          # Réduire pour la démo (500+ en production)
        sa_steps_per_config=100,
    )

    # -- 3. Entraîner le réseau ---------------------------------------
    print(f"\n[Train] Entraînement du réseau ({encoder.INPUT_CHANNELS} canaux d'entrée)...")
    model   = BackpackScoreNet(in_channels=encoder.INPUT_CHANNELS)
    trainer = NNTrainer(model, encoder)
    trainer.train(buffer, epochs=30, batch_size=32, patience=8)
    trainer.evaluate(buffer)

    # Sauvegarder le modèle
    model_path = "src/out/backpack_nn.pt"
    trainer.save_model(model_path)

    # -- 4. Comparer les deux approches -------------------------------
    print("\n[Compare] SA classique vs SA guidé par NN (200 itérations chacun)")

    # Réinitialiser pour un test équitable
    mgr_classic = BackpackManager()
    mgr_nn      = BackpackManager()
    for cat in CONTAINER_CATALOGUE[:2]:
        for mgr_ref in [mgr_classic, mgr_nn]:
            c = Container(id=0, nom=cat["nom"], prix=cat["prix"],
                          largeur=cat["largeur"], hauteur=cat["hauteur"])
            mgr_ref.add_container(c)

    items_classic = [Item.from_catalogue(ITEM_CATALOGUE[i], i+1) for i in range(6)]
    items_nn      = [Item.from_catalogue(ITEM_CATALOGUE[i], i+1) for i in range(6)]

    # SA Classique
    sa_classic = SimulatedAnnealing(mgr_classic, items_classic)
    for _ in range(4):
        sa_classic.step(n_moves=50)
    score_classic = mgr_classic.calculate_score()["total"]
    print(f"  SA Classique  → score final = {score_classic:.2f}")

    # SA Guidé
    sa_nn = NNGuidedSA(mgr_nn, items_nn, model, encoder, k_candidates=5)
    for _ in range(4):
        sa_nn.step(n_moves=50)
    score_nn = mgr_nn.calculate_score()["total"]
    print(f"  SA + NN (k=5) → score final = {score_nn:.2f}")

    print("\n[Demo] Pipeline terminé avec succès ✓")
    return model, encoder, trainer


if __name__ == "__main__":
    # Vérifier la disponibilité de PyTorch
    print(f"PyTorch {torch.__version__} | CUDA: {torch.cuda.is_available()}")
    demo_pipeline()