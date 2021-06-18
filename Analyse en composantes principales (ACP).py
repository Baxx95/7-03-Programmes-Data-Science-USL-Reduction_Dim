# -*- coding: utf-8 -*-
"""
Created on Sun Feb 17 23:01:24 2019

@author: Formateur IT
"""
# -*- coding: utf-8 -*-
"""
Éditeur de Spyder

Ceci est un script temporaire.
"""

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

#APPLICATION DU PCA=========================================
from sklearn.decomposition import PCA
pca = PCA(n_components = 2)
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)

variance_explique = pca.explained_variance_ratio_

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


def affichage_region_dec(X, y, classifier, test_idx=None, resolution=0.02):
   markers = ('s', 'x', 'o', '^', 'v')
   colors = ('red', 'green', 'lightgreen', 'gray', 'cyan')
   cmap = ListedColormap(colors[:len(np.unique(y))])

   # trace la surface de décision
   x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
   x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
   xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
   np.arange(x2_min, x2_max, resolution))
   Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
   Z = Z.reshape(xx1.shape)
   plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
   plt.xlim(xx1.min(), xx1.max())
   plt.ylim(xx2.min(), xx2.max())

   X_test, y_test = X[test_idx, :], y[test_idx]
   for idx, cl in enumerate(np.unique(y)):
      plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1],
               alpha=0.8, c=cmap(idx),
               marker=markers[idx], label=cl)

   if test_idx:
      X_test, y_test = X[test_idx, :], y[test_idx]
      plt.scatter(X_test[:, 0], X_test[:, 1], c='',
               alpha=1.0, linewidth=1, marker='o',
               s=55, label='legend')
      
      
X_combine = np.vstack((X_train, X_test))
#Empilez les tableaux en séquence verticalement 
y_combine = np.hstack((y_train, y_test))
#Empilez les tableaux en séquence horizontalement

affichage_region_dec(X_combine,
                      y_combine, classifier=knn_classifier,
                      test_idx=range(105,150))
   
plt.xlabel('Épaisseur')
plt.ylabel('Taille_cellules_épithéliale')
plt.legend(loc='upper left')
plt.show()  





























