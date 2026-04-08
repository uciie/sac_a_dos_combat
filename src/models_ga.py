import random
import copy
from models import BackpackManager, Item
from ml_surrogate import NNSurrogate 

class GeneticAlgorithmML:
    def __init__(self, initial_manager, inventory, surrogate_model: NNSurrogate, pop_size=20):
        self.population = [initial_manager.clone() for _ in range(pop_size)]
        self.inventory = inventory
        self.surrogate = surrogate_model
        self.best_manager = initial_manager.clone()
        self.best_score = initial_manager.calculate_score()["total"]
        self.history = []
        self.iteration = 0

    def mutate(self, manager):
        """Applique une mutation aléatoire (Ajout, Retrait, Déplacement)"""
        mutant = manager.clone()
        op = random.choice(["ADD", "MOVE", "ROTATE"])
        
        if op == "ADD" and self.inventory:
            item = random.choice(self.inventory)
            pos = mutant.valid_positions(item)
            if pos:
                nx, ny = random.choice(pos)
                mutant.place_item(item, nx, ny)
        
        elif op == "MOVE" and mutant.items_placed:
            iid = random.choice(list(mutant.items_placed.keys()))
            item, ox, oy = mutant.items_placed[iid]
            mutant.remove_item(iid)
            pos = mutant.valid_positions(item)
            if pos:
                nx, ny = random.choice(pos)
                mutant.place_item(item, nx, ny)
                
        # (Vous pouvez ajouter le Crossover ici pour mixer 2 sacs)
        return mutant

    def step(self, generations=10):
        """Fait évoluer la population sur plusieurs générations"""
        for _ in range(generations):
            self.iteration += 1
            new_population = []
            
            # 1. Générer beaucoup de mutants (ex: 5 mutants par individu)
            candidates = []
            for indiv in self.population:
                for _ in range(5):
                    candidates.append(self.mutate(indiv))
                    
            # 2. FILTRE MACHINE LEARNING (L'hybridation !)
            # On prédit le potentiel des candidats avec le Réseau de Neurones
            if self.surrogate.is_trained:
                # On trie les candidats par leur score PRÉDIT (très rapide)
                candidates.sort(key=lambda c: self.surrogate.predict_score([i[0] for i in c.items_placed.values()]), reverse=True)
            
            # On ne garde que les meilleurs (taille de la population) pour l'évaluation réelle
            next_gen = candidates[:len(self.population)]
            
            # 3. Évaluation réelle et sélection
            for cand in next_gen:
                score = cand.calculate_score()["total"]
                if score > self.best_score:
                    self.best_score = score
                    self.best_manager = cand.clone()
            
            self.population = next_gen
            
            self.history.append({
                "iteration": self.iteration,
                "temperature": 1000, # Fake temperature (sera ajustée dans app.py)
                "score": self.best_score,
                "best_score": self.best_score
            })
            
        return self.history[-1]