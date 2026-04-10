import sys
import os
# Cette ligne dit à Python : "Considère le dossier 'src' comme un dossier principal pour chercher les imports"
sys.path.append(os.path.abspath("src"))

import numpy as np
import matplotlib.pyplot as plt
import time
import random

# Maintenant, on enlève "src." partout ici aussi !
from models import BackpackManager, Store, ITEM_CATALOGUE, Item
from ml_surrogate import NNSurrogate
from models_ga import GeneticAlgorithmML

def generate_training_data(num_samples=1000):
    """Génère des configurations aléatoires de sacs pour entraîner le ML"""
    print("Génération du dataset d'entraînement...")
    X, y = [], []
    surrogate = NNSurrogate(len(ITEM_CATALOGUE))
    
    for i in range(num_samples):
        store = Store()
        manager = BackpackManager()
        
        # 1. On génère le marché AVANT de chercher les offres !
        store.generate_market()
        
        # 2. On cherche l'offre qui est un container (garanti par Store)
        container_offer = next(o for o in store.current_offers if o["type"] == "container")
        container = store.buy_container(container_offer["offer_id"])
        if container:
            manager.add_container(container)
        
        # 3. Placer des objets au hasard pour créer un exemple d'apprentissage
        items_in_bag = []
        for template in random.sample(ITEM_CATALOGUE, random.randint(3, 8)):
            # On crée un véritable objet Item à partir du catalogue
            item_id = random.randint(1000, 99999) # ID fictif
            item = Item.from_catalogue(template, item_id)
            
            # On tente de le placer aléatoirement
            pos = manager.valid_positions(item)
            if pos:
                nx, ny = random.choice(pos)
                manager.place_item(item, nx, ny)
                items_in_bag.append(item)
                
        # 4. On extrait les features (les objets présents) et le vrai score
        features = surrogate.extract_features(items_in_bag)
        score = manager.calculate_score()["total"] 
        X.append(features)
        y.append(score)
        
    return np.array(X), np.array(y), surrogate

def run_experiment():
    # 1. Entraîner le Modèle
    X_train, y_train, nn_model = generate_training_data(10000)
    nn_model.train(X_train, y_train)
    nn_model.save("src/out/nn_surrogate.pkl")
    
    # 2. Configurer une partie test avec un vrai sac et un inventaire
    print("\nPréparation du sac et de l'inventaire pour la course...")
    store = Store()
    store.generate_market()
    manager = BackpackManager() 
    
    # Acheter un grand container pour le test
    container_offer = next(o for o in store.current_offers if o["type"] == "container")
    manager.add_container(store.buy_container(container_offer["offer_id"]))
    
    # Remplir l'inventaire avec 15 objets aléatoires pour que l'AG ait du choix
    inventory = []
    for _ in range(15):
        template = random.choice(ITEM_CATALOGUE)
        item_id = random.randint(1000, 99999)
        inventory.append(Item.from_catalogue(template, item_id))
    
    print(f"L'inventaire contient {len(inventory)} objets. Que le meilleur gagne !")
    print("\n--- Course : AG Classique vs AG + Réseau de Neurones ---")
    
    # AG Classique (Réseau non entraîné ou ignoré)
    ga_classic = GeneticAlgorithmML(manager, inventory.copy(), NNSurrogate(len(ITEM_CATALOGUE)))
    start = time.time()
    ga_classic.step(generations=50)
    time_classic = time.time() - start
    
    # AG + ML
    ga_ml = GeneticAlgorithmML(manager, inventory.copy(), nn_model)
    start = time.time()
    ga_ml.step(generations=50)
    time_ml = time.time() - start
    
    print(f"AG Classique : Score max = {ga_classic.best_score:.2f} (Temps: {time_classic:.2f}s)")
    print(f"AG + ML (NN): Score max = {ga_ml.best_score:.2f} (Temps: {time_ml:.2f}s)")

if __name__ == "__main__":
    run_experiment()