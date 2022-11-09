import numpy as np
from tensorflow import keras
from tensorflow.keras import optimizers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

import random

#L'objectif ici est de déterminer le nombre de cases noires

#type de modèle qui permet de décrire le réseau couche par couche
#https://data-flair.training/blogs/keras-models/
modele = Sequential()

# Réseau : 
# C entrées : une par cases 
# première couche : p neurones
# seconde couche : q neurones
# troisième couche : 1 neurone

# activation = sigma

# config avec un nombre aléatoire de 1 parmi des 0
def toutes_configurations(C,n):
    liste = []
    for i in range(n):
        config=[]
        for j in range(C):
            config.append(random.randint(0,1))
        liste.append(config)
    return liste

# Test
C = 20
n=400
#liste = toutes_configurations(C,n)
#print(liste)


def nombre(config):  # nombre de 1
    return config.count(1)




# Architecture

p = 10
q = p

# Première couche : p neurones (entrée de dimension C)
modele.add(Dense(p, input_dim=C, activation='relu'))

# Seconde couche : q neurones
modele.add(Dense(q, activation='relu'))

# Couche de sortie  : 1 neurones
modele.add(Dense(1, activation='relu'))

# sgd = optimizers.SGD(lr=1)
modele.compile(loss='mean_squared_error', optimizer='adam')


# Données d'apprentissage
liste_X = toutes_configurations(C,n)
#random.shuffle(liste_X)
print("Taille des données", len(liste_X))
# liste_Y = [int(deux_cases_consecutives(c)) for c in liste_X]  # deux cases consécutives ?
# liste_Y = [rang_maximum(c) for c in liste_X]  # rang maximum de la liste
liste_Y = [nombre(c) for c in liste_X]  # hauteur maximum de la liste

# Données d'apprentissage
taille_train = len(liste_X)//2
X_train = np.array(liste_X[:taille_train])
Y_train = np.array(liste_Y[:taille_train])

# Données de test
X_test = np.array(liste_X[taille_train:])
Y_test = np.array(liste_Y[taille_train:])

# print(X_train)
# print(Y_train)
# print(X_test)
# print(Y_test)


# Descente de gradient
# modele.fit(X_train, Y_train, epochs=2000, batch_size=len(X_train), verbose = 1)
modele.fit(X_train, Y_train, epochs=4000, batch_size=100, verbose = 0)

print(modele.summary())


# Evaluation 

print('\n Evaluation sur les données de test')
loss_train = modele.evaluate(X_train, Y_train)
print("loss du set d'apprentissage", loss_train)
loss_test = modele.evaluate(X_test, Y_test)
print('loss du set de test', loss_test)


def evaluation():

    print("\nEvaluation p =", p)

    # A. Evaluation sur les données d'apprentissage
    nb_correct = 0
    for i in range(len(X_train)):

        entree = np.array([X_train[i]])
        sortie_attendue = np.array([Y_train[i]])[0]
        sortie_predite = modele.predict(entree,verbose=0)[0][0]
        # print(entree)
        # print(sortie_attendue)
        # print(modele.predict(entree,verbose=0))
        # print(modele.predict(entree,verbose=0)[0])
        # print(modele.predict(entree,verbose=0)[0][0])
        if round(sortie_predite) == sortie_attendue:
            nb_correct += 1

    print("Nb de données d'apprentissage :", len(X_train))
    print("Nb de données prédite correctement :", nb_correct)
    perc = nb_correct/len(X_train)*100
    print("Pourcentage de réussite :",round(perc))

    nb_correct = 0
    for i in range(len(X_test)):

        entree = np.array([X_test[i]])
        sortie_attendue = np.array([Y_test[i]])[0]
        sortie_predite = modele.predict(entree,verbose=0)[0][0]
        # print(entree)
        # print(sortie_attendue)
        # print(sortie_predite)
        if round(sortie_predite) == sortie_attendue:
            nb_correct += 1

    print("Nb de données dans le test :", len(X_test))
    print("Nb de données prédite correctement :", nb_correct)
    perc = nb_correct/len(X_train)*100
    print("Pourcentage de réussite :",round(perc))

    return

#evaluation()