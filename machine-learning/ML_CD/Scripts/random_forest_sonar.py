# -*- coding: utf-8 -*-
"""
random_forest_sonar.py: Avaliação da performance de classificação com Random Forest.

Avaliar a performance da classificação da base de dados sonar com os métodos de 
combinação de classificadores.

@author: Prof. Hugo de Paula
@contact: hugo@pucminas.br


"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.preprocessing import LabelEncoder
#from IPython.display import Image

# Load a common dataset, fit a decision tree to it
sonar = pd.read_excel('../Datasets/sonar.xlsx', sheetname=0) 

X = sonar.iloc[:,0:(sonar.shape[1] - 1)]

le = LabelEncoder()
y = le.fit_transform(sonar.iloc[:,(sonar.shape[1] - 1)])

class_names = le.classes_
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)


# Arvore de decisao
clf = tree.DecisionTreeClassifier()
clf = clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred, target_names=class_names))


# Random forest com 10 arvores
clr = RandomForestClassifier(n_estimators=10)
clr = clr.fit(X_train, y_train)
y_pred = clr.predict(X_test)
print(classification_report(y_test, y_pred, target_names=class_names))

# Random forest com heurísticas extremas
cle = ExtraTreesClassifier(n_estimators=10)
cle = cle.fit(X_train, y_train)
y_pred = cle.predict(X_test)
print(classification_report(y_test, y_pred, target_names=class_names))


