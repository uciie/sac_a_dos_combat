import numpy as np
from sklearn.neural_network import MLPRegressor
import pickle

class NNSurrogate:
    def __init__(self, item_catalogue_size):
        # Réseau de neurones : 2 couches cachées de 32 et 16 neurones
        self.model = MLPRegressor(hidden_layer_sizes=(32, 16), max_iter=1000)
        self.item_catalogue_size = item_catalogue_size
        self.is_trained = False

    def extract_features(self, items_list):
        """
        Transforme une liste d'objets en un vecteur d'entiers (comptage).
        Ex: [0, 2, 1, 0, ...] signifie 0 objet type 1, 2 objets type 2, etc.
        """
        # On suppose que l'ID dans le catalogue correspond à l'index pour simplifier
        features = np.zeros(self.item_catalogue_size)
        for item in items_list:
            # Hashage simple basé sur le nom pour l'exemple (ou utiliser un ID de catalogue fixe)
            idx = hash(item.nom) % self.item_catalogue_size
            features[idx] += 1
        return features

    def train(self, X_train, y_train):
        """Entraîne le réseau de neurones sur un jeu de données généré"""
        if len(X_train) > 0:
            self.model.fit(X_train, y_train)
            self.is_trained = True
            print(f"Réseau de neurones entraîné sur {len(X_train)} exemples. Score R2 : {self.model.score(X_train, y_train):.2f}")

    def predict_score(self, items_list):
        """Prédit le score d'une combinaison d'objets en une fraction de milliseconde"""
        if not self.is_trained:
            return 0.0
        features = self.extract_features(items_list).reshape(1, -1)
        return self.model.predict(features)[0]

    def save(self, filename="nn_surrogate.pkl"):
        with open(filename, 'wb') as f:
            pickle.dump((self.model, self.is_trained), f)

    def load(self, filename="nn_surrogate.pkl"):
        try:
            with open(filename, 'rb') as f:
                self.model, self.is_trained = pickle.load(f)
        except FileNotFoundError:
            print("Aucun modèle pré-entraîné trouvé.")