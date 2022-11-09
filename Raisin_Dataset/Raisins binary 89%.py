import os
import xlrd
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
import numpy as np
from tensorflow import keras
from tensorflow.keras import optimizers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from matplotlib import pyplot as plt
import random
#https://www.codeforests.com/2020/05/25/use-xlrd-library-to-read-excel-file/

"""
For a binary classification problem, a Binomial probability distribution is used. This is achieved using a network with a single node in the output layer that predicts the probability of an example belonging to class 1.
For a multi-class classification problem, a Multinomial probability is used. This is achieved using a network with one node for each class in the output layer and the sum of the predicted probabilities equals one.
Ici j'ai fait une multi classe alors que la binary est plus adaptée, comme ça je peux adapter facilement ce code à un autre dataset
"""

#type de modèle qui permet de décrire le réseau couche par couche
#https://data-flair.training/blogs/keras-models/
modele = Sequential()

# Architecture

# Première couche : p neurones (entrée de dimension C)
modele.add(Dense(10, input_dim=7, activation='linear'))
# Seconde couche : q neurones
modele.add(Dense(10, activation='linear'))

# Couche de sortie  : 2 neurones parce qu'il y a 2 classes de raisins possibles.
#softmax est particulièrement adaptée pour les neurones de sortie dans un classement multi-classe.
modele.add(Dense(1, activation='linear'))

modele.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
#modele.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
#modele.compile(loss='mean_squared_error', optimizer='adam')


# Données d'apprentissage
workbook = xlrd.open_workbook(r"C:/Users/jeanb/OneDrive/Documents/Python/Algo génétiques et réseaux de neurones/Raisin_Dataset/Raisin_Dataset - scaled.xls")
#getting the first sheet
sheet = workbook.sheet_by_index(0)

row_count = sheet.nrows
col_count = sheet.ncols
print("Taille des données",row_count,col_count)

liste_X=[]
liste_Y=[]
total=[sheet.row_values(i, start_colx=0, end_colx=8) for i in range(1,sheet.nrows)]
random.shuffle(total)

for data in total:
	liste_X.append(data[:-1])
	liste_Y.append(data[-1])


# Données d'apprentissage
taille_train=int(sheet.nrows * 0.67)
X_train = np.array(liste_X[:taille_train])
Y_train = np.array(liste_Y[:taille_train])

# Données de test
X_test = np.array(liste_X[taille_train:])
Y_test = np.array(liste_Y[taille_train:])

#print(X_train)
#print(Y_train)
#print(X_test)
#print(Y_test)
#print(len(Y_test))


# Descente de gradient
# modele.fit(X_train, Y_train, epochs=2000, batch_size=len(X_train), verbose = 1)
modele.fit(X_train, Y_train, epochs=4000, batch_size=100, verbose = 1)

print(modele.summary())


# Evaluation 

print('\n Evaluation sur les données de test')
loss_train = modele.evaluate(X_train, Y_train)
print("loss du set d'apprentissage", loss_train)
loss_test = modele.evaluate(X_test, Y_test)
print('loss du set de test', loss_test)

#not correct for the moment
def evaluation():

    # A. Evaluation sur les données d'apprentissage
    nb_correct = 0
    for i in range(len(X_train)):

        entree = np.array([X_train[i]])
        sortie_attendue = np.array([Y_train[i]])[0]
        sortie_predite = modele.predict(entree,verbose=3)[0][0]
        print(sortie_attendue)
        print(sortie_predite)
        print('/')
        # print(entree)
        # print(modele.predict(entree,verbose=0))
        # print(modele.predict(entree,verbose=0)[0])
        # print(modele.predict(entree,verbose=0)[0][0])
        if round(sortie_predite) == sortie_attendue:
            nb_correct += 1

    print("Nb de données d'apprentissage :", len(X_train))
    print("Nb de données prédite correctement :", nb_correct)
    perc = nb_correct/len(X_train)*100
    print("Pourcentage de réussite :",round(perc),"%")

    nb_correct = 0
    for i in range(len(X_test)):

        entree = np.array([X_test[i]])
        sortie_attendue = np.array([Y_test[i]])[0]
        sortie_predite = modele.predict(entree,verbose=3)[0][0]
        # print(entree)
        # print(sortie_attendue)
        #print(sortie_predite)
        if round(sortie_predite) == sortie_attendue:
            nb_correct += 1

    print("Nb de données dans le test :", len(X_test))
    print("Nb de données prédite correctement :", nb_correct)
    perc = nb_correct/len(X_test)*100
    print("Pourcentage de réussite :",round(perc),"%")

    return

evaluation()