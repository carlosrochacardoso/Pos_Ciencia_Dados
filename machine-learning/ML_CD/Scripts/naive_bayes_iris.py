# -*- coding: utf-8 -*-
"""
iris_nb.py: Classificação do dataset Iris da UCI pelo Gaussian Naive Bayes.

@author: Prof. Hugo de Paula
@contact: hugo@pucminas.br


"""

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, classification_report
import pandas as pd

iris = load_iris()

X_train, X_test, y_train, y_test = train_test_split(
                                iris['data'], iris['target'], random_state=0)


gnb = GaussianNB()
y_pred = gnb.fit(X_train, y_train).predict(X_test)

print("Número de erros de classificação {0} de {1}"
      .format((y_test != y_pred).sum(), iris.data.shape[0]))

ac = gnb.score(X_test, y_test)

print("\nAcurácia do modelo: {0:.2f}%\n".format(100*ac))

print(classification_report(y_test, y_pred, target_names=iris.target_names))

cnf_matrix = confusion_matrix(y_test, y_pred)


lin_nomes = ['Setosa', 'Versicolor', 'Virginica']
novo_dataframe = pd.DataFrame(cnf_matrix, index=lin_nomes, columns=lin_nomes)
print(novo_dataframe)





