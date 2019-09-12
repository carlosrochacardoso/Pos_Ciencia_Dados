# -*- coding: utf-8 -*-
"""
arvore_decisao_clima.py: Código para estudo de árvore de decisão

@author: Prof. Hugo de Paula
@contact: hugo@pucminas.br


"""

import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn import tree

# WEATHER.NOMINAL, Weka
# http://storm.cis.fordham.edu/~gweiss/data-mining/datasets.html
# DESCRIÇÃO DOS ATRIBUTOS:
# Aparência {Ensolarado, Nublado, Chuvoso}
# Temperatura {Quente, Moderado, Frio}
# Umidade {Alta, Normal}
# Vento {Fraco, Forte}
# Jogar {Sim, Não}

clima_nominal = pd.read_excel('../Datasets/clima.xlsx', sheet_name=0) 
print("\nDimensões: {0}".format(clima_nominal.shape))
print("\nCampos: {0}".format(clima_nominal.keys()))
print(clima_nominal.describe(), sep='\n')

X_dict = clima_nominal.iloc[:,0:4].T.to_dict().values()
vect = DictVectorizer(sparse=False)
X_train = vect.fit_transform(X_dict)

le = LabelEncoder()
y_train = le.fit_transform(clima_nominal.iloc[:,4])


# Exibe o dado convertido em dicionario.
print(X_dict)

# Exibe a estrutura do dado convertido em binário.
print("Shape do dado de treinamento: {0}".format(X_train.shape))


tree_clima = DecisionTreeClassifier(random_state=0, criterion='entropy')
tree_clima = tree_clima.fit(X_train, y_train)
print("Acurácia:", tree_clima.score(X_train, y_train))

y_pred = tree_clima.predict(X_train)
print("Acurácia de previsão:", accuracy_score(y_train, y_pred))
print(classification_report(y_train, y_pred))

cnf_matrix = confusion_matrix(y_train, y_pred)
cnf_table = pd.DataFrame(data=cnf_matrix, index=["Jogar=Não", "Jogar=Sim"], columns=["Jogar(prev)=Não", "Prev. Jogar(prev)=Sim"])
print(cnf_table)

with open("nominal.dot", 'w') as f:
     f = tree.export_graphviz(tree_clima, out_file=f, 
                              feature_names=vect.feature_names_, 
                              class_names=["Jogar=Não", "Jogar=Sim"])
     
# dot -Tpdf nominal.dot -o nominal.pdf
