# -*- coding: utf-8 -*-
"""
iris_rf.py: Avaliação da performance de classificação com Random Forest.

Avaliar a performance da classificação da base de dados iris com os métodos de 
combinação de classificadores.

@author: Prof. Hugo de Paula
@contact: hugo@pucminas.br


"""

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
#from IPython.display import Image
import pydotplus

iris = load_iris()
X, y = iris.data, iris.target
class_names = iris.target_names
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)


clf = tree.DecisionTreeClassifier()
clf = clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred, target_names=class_names))


dot_data = tree.export_graphviz(clf, 
              out_file=None,
              feature_names=iris.feature_names,
              class_names=iris.target_names,
              filled=True, rounded=True)
graph = pydotplus.graph_from_dot_data(dot_data)
#Image(graph.create_png("iris-dt.png"))



# Random forest com 10 arvores
clr = RandomForestClassifier(n_estimators=20)
clr = clr.fit(X_train, y_train)
y_pred = clr.predict(X_test)
print(classification_report(y_test, y_pred, target_names=class_names))

# Random forest com heurísticas extremas
cle = ExtraTreesClassifier(n_estimators=20)
cle = cle.fit(X_train, y_train)
y_pred = cle.predict(X_test)
print(classification_report(y_test, y_pred, target_names=class_names))


