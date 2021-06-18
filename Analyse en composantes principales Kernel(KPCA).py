# -*- coding: utf-8 -*-
"""
Created on Thu Jan 24 22:48:44 2019

@author: Formateur IT
"""
# -*- coding: utf-8 -*-


# -*- coding: utf-8 -*-
"""
Created on Thu Jan 24 21:57:46 2019

@author: Formateur IT
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
import seaborn

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from matplotlib.colors import ListedColormap

dataset = pd.read_csv('prediction_de_fraud_2.csv')
X = dataset[['step','type','oldbalanceOrg','newbalanceOrig','oldbalanceDest','newbalanceDest','isFraud','isFlaggedFraud']].values
y = dataset['isFraud'].values

labEncr_X = LabelEncoder()
X[:,1] = labEncr_X.fit_transform(X[:,1])

# Fractionner le jeu de données entre les ensembles de formation et de test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

# Adaptation de la régression logistique à l'ensemble de formation

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Application du  Kernel PCA
from sklearn.decomposition import KernelPCA
kernelpca = KernelPCA(n_components = 2, kernel = 'rbf')
X_train = kernelpca.fit_transform(X_train)
X_test = kernelpca.transform(X_test)



from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(X_train, y_train)

# Prédire les résultats du test
y_pred = classifier.predict(X_test)



def affichage_region_dec(X, y, classifier, test_idx=None, resolution=0.02):
   markers = ('s', 'x', 'o', '^', 'v')
   colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
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
               s=55, label='test set')
   
   
           
X_combine = np.vstack((X_train, X_test))
#Empilez les tableaux en séquence verticalement 
y_combine = np.hstack((y_train, y_test))
#Empilez les tableaux en séquence horizontalement 

affichage_region_dec(X_combine,
                      y_combine, classifier=classifier,
                      test_idx=range(105,150))
   
plt.xlabel('oldbalanceOrg')
plt.ylabel('newbalanceOrig')
plt.legend(loc='upper left')
plt.show()     

#EVALUATION DU MODELE

classifier.score(X_test, y_test)
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

   
   
