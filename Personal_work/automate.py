import pandas as pd
import numpy as np
import optuna
from sklearn.linear_model import LogisticRegression # Logistic Regression
from sklearn.model_selection import cross_val_score # split into two sample : validation and training

#### Définir la fonction objective
def find_best_hyperparameter_logistic(X, y, nb_fold:int, metric:str, direction:str):
    '''
    Return the best hyperameter with coefficient and score
    
    Attribut
    --------
    X : array
        feature of dataframe
    y : array
        target of dataframe
    metric : str
        metric to use for scoring
    direction : str
    
    Return
    ------
    model 
        return the best model with coefficient and score
    '''
    def objective(trial):
        
        l1_ratio = trial.suggest_uniform('l1_ratio', 0, 1)  # Exploration de l'hyperparamètre l1_ratio entre 0 et 1
    
        # Créer le modèle de régression logistique avec penalty='elasticnet'
        model = LogisticRegression(penalty='elasticnet', l1_ratio=l1_ratio, solver='saga')
    
        # Calculer le score de validation croisée (utilisation de la précision pour l'exemple)
        score = cross_val_score(model, X, y, cv=nb_fold, scoring=metric).mean()
        return score
    
    #### Créer un objet Optuna pour l'optimisation
    study = optuna.create_study(direction=direction)  # On cherche à maximiser la précision
    
    #### Retourne le modèle optimale
    return study