# -*- coding: utf-8 -*-
"""
multiplos_avaliacao_cancer.py: Avaliação da performance de classificador na base de cancer de mama

@author: Prof. Hugo de Paula
@contact: hugo@pucminas.br

"""


from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

import numpy as np
import matplotlib.pyplot as plt


cancer = load_breast_cancer()
print("cancer.keys(): {}".format(cancer.keys()))
print("Contagem de registros por classe:\n{}".format(
      {n: v for n, v in zip(cancer.target_names, np.bincount(cancer.target))}))
print("Atributos:\n{}".format(cancer.feature_names))


X_train, X_test, y_train, y_test = train_test_split(
    cancer.data, cancer.target, stratify=cancer.target, random_state=66)

training_accuracy = []
test_accuracy = []

neighbors_settings = range(1, 11)

for n_neighbors in neighbors_settings:
    clf = KNeighborsClassifier(n_neighbors=n_neighbors)
    clf.fit(X_train, y_train)
    training_accuracy.append(clf.score(X_train, y_train))
    test_accuracy.append(clf.score(X_test, y_test))
    
plt.figure(figsize=(6, 4))
plt.plot(neighbors_settings, training_accuracy, label="acurácia de treinamento")
plt.plot(neighbors_settings, test_accuracy, label="acurácia de teste")
plt.ylabel("Acurácia")
plt.xlabel("Num de vizinhos")
plt.legend()




logreg = LogisticRegression().fit(X_train, y_train)
print("Regressão Logística C=1")
print("Acurácia na base de treinamento: {:.2f}".format(logreg.score(X_train, y_train)))
print("Acurácia na base de teste: {:.2f}".format(logreg.score(X_test, y_test)))

logreg100 = LogisticRegression(C=100).fit(X_train, y_train)
print("Regressão Logística C=100")
print("Acurácia na base de treinamento: {:.2f}".format(logreg100.score(X_train, y_train)))
print("Acurácia na base de teste: {:.2f}".format(logreg100.score(X_test, y_test)))

logreg001 = LogisticRegression(C=0.01).fit(X_train, y_train)
print("Regressão Logística C=0.01")
print("Acurácia na base de treinamento: {:.2f}".format(logreg001.score(X_train, y_train)))
print("Acurácia na base de teste: {:.2f}".format(logreg001.score(X_test, y_test)))

plt.figure(figsize=(6, 4))
plt.plot(logreg.coef_.T, 'o', label="C=1")
plt.plot(logreg100.coef_.T, '^', label="C=100")
plt.plot(logreg001.coef_.T, 'v', label="C=0.01")
plt.xticks(range(cancer.data.shape[1]), cancer.feature_names, rotation=90)
plt.hlines(0, 0, cancer.data.shape[1])
plt.ylim(-5, 5)
plt.xlabel("Coeficiente")
plt.ylabel("Magnitude")
plt.legend()


forest = RandomForestClassifier(n_estimators=100, random_state=0)
forest.fit(X_train, y_train)
print("Random Forest")
print("Acurácia na base de treinamento: {:.2f}".format(forest.score(X_train, y_train)))
print("Acurácia na base de teste: {:.2f}".format(forest.score(X_test, y_test)))

gbrt = GradientBoostingClassifier(random_state=0)
gbrt.fit(X_train, y_train)
print("Gradient Boosting")
print("Acurácia na base de treinamento: {:.2f}".format(gbrt.score(X_train, y_train)))
print("Acurácia na base de teste: {:.2f}".format(gbrt.score(X_test, y_test)))

svc = SVC(C=1000)
svc.fit(X_train, y_train)
print("SVC")
print("Acurácia na base de treinamento: {:.2f}".format(svc.score(X_train, y_train)))
print("Acurácia na base de teste: {:.2f}".format(svc.score(X_test, y_test)))


mlp = MLPClassifier(max_iter=1000, random_state=0)
mlp.fit(X_train, y_train)
print("Multi-Layer Perceptron")
print("Acurácia na base de treinamento: {:.2f}".format(mlp.score(X_train, y_train)))
print("Acurácia na base de teste: {:.2f}".format(mlp.score(X_test, y_test)))




mean_on_train = X_train.mean(axis=0)
std_on_train = X_train.std(axis=0)

X_train_scaled = (X_train - mean_on_train) / std_on_train
X_test_scaled = (X_test - mean_on_train) / std_on_train

mlp = MLPClassifier(max_iter=1000, random_state=0)
mlp.fit(X_train_scaled, y_train)
print("Multi-Layer Perceptron com normalizacao")
print("Acurácia na base de treinamento: {:.2f}".format(mlp.score(X_train_scaled, y_train)))
print("Acurácia na base de teste: {:.2f}".format(mlp.score(X_test_scaled, y_test)))

svc = SVC(C=1000)
svc.fit(X_train_scaled, y_train)
print("SVC com normalizacao")
print("Acurácia na base de treinamento: {:.2f}".format(svc.score(X_train_scaled, y_train)))
print("Acurácia na base de teste: {:.2f}".format(svc.score(X_test_scaled, y_test)))
























