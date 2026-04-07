import numpy as np 
from sklearn.neural_network import MLPregressor 
import pickle 

class NNSurrogate: 
    def __init__(self,item_catalogue_size):
        # Réseau de neurones : 2 couches cachées de 32 et 16 neurones
        self.model = MLPRegressor(hidden_layer_sizes=(32,16),max_iter=1000)
        self.item_catalogue_size = item_catalogue_size
        self.is_trained = False 

    def extract_features(self,items_list): 
        """
        Transforme une liste d'objets en un vecteur d'entiers (comptage).
        Ex: [0, 2, 1, 0, ...] signifie 0 objet type 1, 2 objets type 2, etc.
        """
        # On suppose que l'ID dans le catalogue correspond à l'index pour simplifier
        features = np.zeros(self.item_catalogue_size)
        for item in items_list: 
            