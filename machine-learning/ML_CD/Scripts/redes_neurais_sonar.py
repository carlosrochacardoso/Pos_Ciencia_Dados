# -*- coding: utf-8 -*-
"""
redes_neurais_sonar.py: Avaliação da performance de classificador MLP

Avaliar a performance da classificação da base de dados sonar com 
redes neurais artificiais.

@author: Prof. Hugo de Paula
@contact: hugo@pucminas.br


"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder
import numpy as np


np.set_printoptions(precision=2)

# Carrega a base de dados sonar.
sonar = pd.read_excel('../Datasets/sonar.xlsx', sheetname=0) 

X = sonar.iloc[:,0:(sonar.shape[1] - 1)]

le = LabelEncoder()
y = le.fit_transform(sonar.iloc[:,(sonar.shape[1] - 1)])

class_names = le.classes_


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=0)



print("----------------------------------------")
print("MLP com uma camadas ocultas")

# Rede Perceptron Multicamadas (MLP):  Configuração default otimizando a função log-loss
# uma camada oculta com 100 neurônios.

mlp = MLPClassifier(solver='lbfgs', random_state=0).fit(X_train, y_train)
mlp.fit(X_train, y_train)
y_pred = mlp.predict(X_test)


print("Camadas da rede: {}".format(mlp.n_layers_))
print("Neurônios na camada oculta: {}".format(mlp.hidden_layer_sizes))
print("Neurônios na camada de saída: {}".format(mlp.n_outputs_))
print("Pesos na camada de entrada: {}".format(mlp.coefs_[0].shape))
print("Pesos na camada oculta: {}".format(mlp.coefs_[1].shape))

print("Acurácia da base de treinamento: {:.2f}".format(mlp.score(X_train, y_train)))
print("Acurácia da base de teste: {:.2f}".format(mlp.score(X_test, y_test)))


print(classification_report(y_test, y_pred, target_names=class_names))

# Calcula a matriz de confusão
cnf_matrix = confusion_matrix(y_test, y_pred)
print(cnf_matrix)



print("----------------------------------------")
print("MLP com duas camadas ocultas")


mlp = MLPClassifier(solver='lbfgs', random_state=0, hidden_layer_sizes=[100, 60]).fit(X_train, y_train)
mlp.fit(X_train, y_train)
y_pred = mlp.predict(X_test)


print("Camadas da rede: {}".format(mlp.n_layers_))
print("Neurônios na camada oculta: {}".format(mlp.hidden_layer_sizes))
print("Neurônios na camada de saída: {}".format(mlp.n_outputs_))
print("Pesos na camada de entrada: {}".format(mlp.coefs_[0].shape))
print("Pesos na camada oculta: {}".format(mlp.coefs_[1].shape))

print("Acurácia da base de treinamento: {:.2f}".format(mlp.score(X_train, y_train)))
print("Acurácia da base de teste: {:.2f}".format(mlp.score(X_test, y_test)))


print(classification_report(y_test, y_prev, target_names=class_names))

# Calcula a matriz de confusão
cnf_matrix = confusion_matrix(y_test, y_prev)
print(cnf_matrix)


# ajustamento dos dados

# Calcula a média e o desvio padrão de cada atributo da base de treinamento
mean_on_train = X_train.mean(axis=0)
std_on_train = X_train.std(axis=0)

# Normaliza os atributos pela norma Z = (X - média) / desvio padrão
# afterwards, mean=0 and std=1
X_train_scaled = (X_train - mean_on_train) / std_on_train
# usa a esma transformação nos dados de teste
X_test_scaled = (X_test - mean_on_train) / std_on_train


# A rede atinge o número máximo de iterações, mas não converge.
mlp = MLPClassifier(random_state=0)
mlp.fit(X_train_scaled, y_train)
print("Acurácia da base de treinamento: {:.2f}".format(mlp.score(X_train, y_train)))
print("Acurácia da base de teste: {:.2f}".format(mlp.score(X_test, y_test)))

# Vamos aumentar o número máximo de iterações
mlp = MLPClassifier(max_iter=1000, random_state=0)
mlp.fit(X_train_scaled, y_train)
print("Acurácia da base de treinamento: {:.2f}".format(mlp.score(X_train, y_train)))
print("Acurácia da base de teste: {:.2f}".format(mlp.score(X_test, y_test)))



