

# -*- coding: utf-8 -*-
"""
Created on Sat Jan 26 03:16:52 2019

@author: formateurit
"""

import numpy as np 
import pandas as pd 

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

dataset = pd.read_csv("cancer_du_sein-wisconsin.csv") 

X = dataset.iloc[:,-10:-1].values
target = dataset['Classe'].values

#gestion des  valeurs nulls
dataset.isnull().any()

from sklearn.preprocessing import Imputer

imptr = Imputer(missing_values= 'NaN',strategy = 'mean' ,axis = 0)

imptr.fit(X[:,5:6])
#Imputez toutes les valeurs manquantes dans X
X[:,5:6] = imptr.transform(X[:,5:6])



from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, target, test_size = 0.3, random_state = 42, stratify = target)

#APPLICATION DU LDA=========================================
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
lda = LDA(n_components = 1)
#Nombre de composants (<n_classes - 1) pour la réduction de la dimensionnalité.
X_train = lda.fit_transform(X_train, y_train)
X_test = lda.transform(X_test)

# ===========================================================










from sklearn.neighbors import KNeighborsClassifier


#Initialisation du classifieur kNN avec 3 voisins

knn_classifier = KNeighborsClassifier(n_neighbors=3)

# Adapter le classifieur aux données d'apprentissage

knn_classifier.fit(X_train, y_train)

y_pred = knn_classifier.predict(X_test)



# Extraire le score de précision des ensembles de test
knn_classifier.score(X_test, y_test)

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)

