# -*- coding: utf-8 -*-
"""
regressao_linear_oleo.py: Avaliação da performance de previsao de consumo de oleo.

@author: Prof. Hugo de Paula
@contact: hugo@pucminas.br

"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso

# Define número de casas decimais
np.set_printoptions(precision=4)


# Carrega a base de dados de aprendizado
print("Carrega a base de dados de aprendizado")
oleo = pd.read_excel('../Datasets/Atividade - Regressao - Bases.xlsx', sheetname=0)
print("\nDimensões: {0}".format(oleo.shape))
print("\nCampos: {0}".format(oleo.keys()))
print(oleo.describe(), sep='\n')

X = oleo.iloc[:,0:(oleo.shape[1] - 1)]
y = oleo.iloc[:,(oleo.shape[1] - 1)]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=0)
#X_train = X_test = X
#y_train = y_test = y


print("\n-----------------------------------------------")

# Carrega a base de dados de previsao
print("Carrega a base de dados de previsao")
previsao = pd.read_excel('../Datasets/Atividade - Regressao - Bases.xlsx', sheetname=1)
print("\nDimensões: {0}".format(previsao.shape))
print("\nCampos: {0}".format(previsao.keys()))
print(previsao.describe(), sep='\n')




# Cria um modelo de regressão linear
lnr = LinearRegression().fit(X_train, y_train)

print("\nRegressao Linear")
print("Acurácia da base de treinamento: {:.2f}".format(lnr.score(X_train, y_train)))
print("Acurácia da base de testes: {:.2f}".format(lnr.score(X_test, y_test)))
print("w: {}  b: {}".format(lnr.coef_, lnr.intercept_))



#Cria um modelo de regressão com 3 vizinhos. Os valores default são:
# KNeighborsRegressor(algorithm='auto', leaf_size=30, metric='minkowski',
# metric_params=None, n_jobs=1, n_neighbors=3, p=2, weights='uniform')

knreg = KNeighborsRegressor(n_neighbors=3)
knreg.fit(X_train, y_train)

print("\nRegressao K-NN")
print("Acurácia da base de treinamento: {:.2f}".format(knreg.score(X_train, y_train)))
print("Acurácia da base de testes: {:.2f}".format(knreg.score(X_test, y_test)))


print("\nDeploy do modelo")

y_prev = lnr.predict(previsao)


previsao["Aquecimento_oleo"] = y_prev


print("Previsão total de venda de óleo: {:.2f}".format(y_prev.sum()))
print("Consumo médio de óleo por residência: {:.2f}".format(y_prev.mean()))
print("Consumo médio de óleo por número de residentes: \n{}"
          .format(previsao[["Num_ocupantes","Aquecimento_oleo"]].groupby('Num_ocupantes').mean()))


# Regularização e feature selection

ridge = Ridge(alpha=1).fit(X_train, y_train)
print("Acurácia da base de treinamento: {:.2f}".format(ridge.score(X_train, y_train)))
print("Acurácia da base de testes: {:.2f}".format(ridge.score(X_test, y_test)))


lasso = Lasso(alpha=1).fit(X_train, y_train)
print("Acurácia da base de treinamento: {:.2f}".format(lasso.score(X_train, y_train)))
print("Acurácia da base de testes: {:.2f}".format(lasso.score(X_test, y_test)))




print("Regressão Linear: w: {}  b: {}".format(lnr.coef_, lnr.intercept_))
print("Ridge: w: {}  b: {}".format(ridge.coef_, ridge.intercept_))
print("Número de atributos usados: {}".format(np.sum(ridge.coef_ != 0)))

print("Lasso: w: {}  b: {}".format(lasso.coef_, lasso.intercept_))
print("Número de atributos usados: {}".format(np.sum(lasso.coef_ != 0)))



