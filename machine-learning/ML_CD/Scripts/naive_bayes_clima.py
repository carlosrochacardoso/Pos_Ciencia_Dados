# -*- coding: utf-8 -*-
"""
clima_nb.py: Código para estudo de árvore de decisão

@author: Prof. Hugo de Paula
@contact: hugo@pucminas.br


"""

import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.feature_extraction import DictVectorizer
from sklearn.naive_bayes import BernoulliNB


clima_nominal = pd.read_excel('../Datasets/clima.xlsx', sheetname=0) 

X_dict = clima_nominal.iloc[:,0:4].T.to_dict().values()
vect = DictVectorizer(sparse=False)
X_train = vect.fit_transform(X_dict)

y_train = clima_nominal.iloc[:,4]

nb_nominal = BernoulliNB()
nb_nominal = nb_nominal.fit(X_train, y_train)
print("Acurácia:", nb_nominal.score(X_train, y_train))

y_pred = nb_nominal.predict(X_train)
print("Acurácia de previsão:", accuracy_score(y_train, y_pred))
print(classification_report(y_train, y_pred))

cnf_matrix = confusion_matrix(y_train, y_pred)

print(cnf_matrix)


lin_nomes = ['N', 'S']
novo_dataframe = pd.DataFrame(cnf_matrix, index=lin_nomes, columns=lin_nomes)
print(novo_dataframe)








