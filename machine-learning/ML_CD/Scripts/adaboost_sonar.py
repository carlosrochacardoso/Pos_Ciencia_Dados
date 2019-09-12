# -*- coding: utf-8 -*-
"""
sonar_ab.py: Avaliação da performance de classificação com ensembles: Adaboost e Random Forest.

Avaliar a performance da classificação da base de dados sonar com os métodos de 
combinação de classificadores.

@author: Prof. Hugo de Paula
@contact: hugo@pucminas.br

"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.preprocessing import LabelEncoder

# Load a common dataset, fit a decision tree to it
sonar = pd.read_excel('../Datasets/sonar.xlsx', sheetname=0) 

X = sonar.iloc[:,0:(sonar.shape[1] - 1)]

le = LabelEncoder()
y = le.fit_transform(sonar.iloc[:,(sonar.shape[1] - 1)])

class_names = le.classes_
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)


clf = tree.DecisionTreeClassifier()
clf = clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print("Classificador Árvore de Decisão:\n")
print(classification_report(y_test, y_pred, target_names=class_names))


# Random forest com 10 arvores
clr = RandomForestClassifier(n_estimators=10)
clr = clf.fit(X_train, y_train)
y_pred = clr.predict(X_test)
print("Classificador Random Forest:\n RandomForestClassifier(n_estimators=10)\n")
print(classification_report(y_test, y_pred, target_names=class_names))

# Random forest com heurísticas extremas
cle = ExtraTreesClassifier(n_estimators=10)
cle = cle.fit(X_train, y_train)
y_pred = cle.predict(X_test)
print("Classificador Extreme Tree:\n ExtraTreesClassifier(n_estimators=10)\n")
print(classification_report(y_test, y_pred, target_names=class_names))



# Adaboost com árvores mínimas
ada = AdaBoostClassifier(tree.DecisionTreeClassifier(max_depth=1), 
                         algorithm="SAMME", n_estimators=200)
ada.fit(X_train, y_train)
y_pred = ada.predict(X_test)
print("Classificador AdaBoost:\n AdaBoostClassifier(tree.DecisionTreeClassifier(max_depth=1), algorithm=\"SAMME\", n_estimators=200)\n")
print(classification_report(y_test, y_pred, target_names=class_names))


