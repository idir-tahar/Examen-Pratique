import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics

import warnings
warnings.filterwarnings('ignore')

data = {
    'Taille(cm)': [182, 180, 170, 180, 152, 168, 165, 175],
    'Poids(kg)': [81.6, 86.2, 77.1, 74.8, 45.4, 68.0, 59.0, 68.0],
    'Pointure(cm)': [30, 28, 30, 25, 15, 20, 18, 23],
    'Genre': ['masculin', 'masculin', 'masculin', 'masculin', 'feminin', 'feminin', 'feminin', 'feminin'],
}

df = pd.DataFrame(data)

X = df.iloc[:, lambda df: [1, 2, 3]]
y = df.iloc[:, 0]

from sklearn.model_selection import train_test_split

#decomposer les donnees predicteurs en training/testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=44)

# faire une fonction qui retourne les metriques necessaires sous forme d'un
# dictionnaire

def eval_metrics(actual, pred):
    accuracy = metrics.accuracy_score(y_test, y_pred)
    precision = metrics.precision_score(y_test, y_pred, average='micro')
    recall = metrics.recall_score(y_test, y_pred, average='micro')
    F1 = 2 * ((precision * recall) / (precision + recall))
    return {'accuracy':accuracy, 'precision': precision, 'recall': recall, 'F1': F1}

# mettre les parametres a changer sous forme de dictionnaire
parameters = {
    'max_depth': [1,2,3],
    'min_samples_split': [2,3,4,5]
}

# avec une 'nested loop' on va lier chaque element du 'max_depth' avec chaque element du 
# 'min_samples_split'

# definir un dictionnaire
metriques = {}

for depth in parameters['max_depth']:
	for min_samples in parameters['min_samples_split']:
		with mlflow.start_run():
            # running the usual flow
            dtc = DecisionTreeClassifier(random_state=42, max_depth=depth, min_samples_split=min_samples)
            dtc.fit(X_train, y_train)
            y_pred = dtc.predict(X_test)
            # recuperer tous les metriques sous forme d'in dictionnaire
            all_metrics = eval_metrics(X_test, y_pred)
			# remplir le dictionnaire de toutes les metriques
			metriques[max_depth][min_samples] =  all_metrics
			
			# ecrire toutes les metriques dans un fichier texte:
			with open("metrics.txt", 'w') as outfile:
        		outfile.write(metriques)















