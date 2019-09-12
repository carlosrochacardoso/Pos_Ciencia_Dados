# -*- coding: utf-8 -*-
"""
sonar_dt.py: Código para estudo de árvore de decisão.

@author: Prof. Hugo de Paula
@contact: hugo@pucminas.br


"""

import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn import tree


sonar = pd.read_excel('../Datasets/sonar.xlsx', sheetname=0) 
print("\nDimensões: {0}".format(sonar.shape))
print("\nCampos: {0}".format(sonar.keys()))
print(sonar.describe(), sep='\n')

X_train = sonar.iloc[:,0:(sonar.shape[1] - 1)]

le = LabelEncoder()
y_train = le.fit_transform(sonar.iloc[:,(sonar.shape[1] - 1)])



sonar_tree = DecisionTreeClassifier(random_state=0)
sonar_tree = sonar_tree.fit(X_train, y_train)
print("Acurácia:", sonar_tree.score(X_train, y_train))

Train_predict = sonar_tree.predict(X_train)
print("Acurácia de previsão:", accuracy_score(y_train, Train_predict))
print(classification_report(y_train, Train_predict))

with open("sonar.dot", 'w') as f:
     f = tree.export_graphviz(sonar_tree, out_file=f)
     
# dot -Tpdf sonar.dot -o sonar.pdf

