# -*- coding: utf-8 -*-
"""
transformacao_dados_multiplos.py: Código para análise exploratória dos dados

@author: Prof. Hugo de Paula
@contact: hugo@pucminas.br


"""

import numpy as np
import pandas as pd

# Base de dados Titanic, Kaggle:
# https://www.kaggle.com/c/titanic/data
# DESCRIÇÃO DOS ATRIBUTOS:
# survival        Sobrevivente
#                 (0 = Não; 1 = Sim)
# pclass          Classe do passageiro
#                 (1 = 1a classe; 2 = 2a classe; 3 = 3a classe)
# name            Nome
# sex             Sexo
# age             Idade
# sibsp           Número de irmãos/conjuges à bordo
# parch           Número de pais/filhos à bordo
# ticket          Número da passagem
# fare            Tarifa do passageiro
# cabin           Cabine
# embarked        Porto de embarque
#                 (C = Cherbourg; Q = Queenstown; S = Southampton)

# Carregamento dos dados
titanic_train = pd.read_csv("../Datasets/Titanic/train.csv") 
print("\nDimensões de Titanic: {0}\n".format(titanic_train.shape))
print("\nCampos de Titanic: {0}\n".format(titanic_train.keys()))
print("\nTipos dos dados: {0}\n".format(titanic_train.dtypes))

# Explorar os dados (início/cabeça (head) e fim/caula (tail))
print(titanic_train.head(5))

# Estatística descritiva dos dados
descricao_titanic = titanic_train.describe()
print(descricao_titanic)

# Os atributos não numéricos são descartados. Para se ter uma visão dos 
# atributos categóricos.

categorical = titanic_train.dtypes[titanic_train.dtypes == "object"].index
print("\n", categorical, sep='\n')

print("\n", titanic_train[categorical].describe(), sep='\n')

###############################################################################
# Remoção de atributos irrelevantes
# Por exemplo, passengerId é apenas uma chave primária para identificar um
# passageiro, mas não é relevante para o problema. Portanto, pode ser removido.

del titanic_train["PassengerId"]

# O atributo survival é nosso objetivo (ou label)
# atributos que descrevem os passageiros ou os agrupam em categorias são úteis.
# Por isso, os atributos Pclass, Sex, Age, SibSp, Parch, Fare and Embarked 
# parecem ser interessantes e devem ser mantidos.
# Analise agora os atributos:  Name, Ticket e Cabin.

# Explorando os atributos Name, Ticket e Cabin.
print("\nAnálise do atributo Name:", sorted(titanic_train["Name"])[0:15], sep='\n')
print(titanic_train["Name"].describe())

print("\nAnálise do atributo Ticket:", sorted(titanic_train["Ticket"])[0:15], sep='\n')
print(titanic_train["Ticket"].describe())

print("\nAnálise do atributo Cabin:", titanic_train["Cabin"][0:15], sep='\n')
print(titanic_train["Cabin"].describe())

# Após análise:
# 1 - Atributo Name náo é útil para previsão, mas pode ser útil para 
# identificação dos registros ou pós-processamento 
# (por exemplo, extrair o último nome).
# 2 - Atributo Ticket não identifica o registro e nem descreve o passageiro,
# por isso, deve ser removido.
# 3 - Atributo Cabin não identifica bem os passageiro, mas pode ser útil 
# utilizarmos o padrão letra+numero para descrever os passageiros pelo andar 
# do local da cabine.

del titanic_train["Ticket"]



###############################################################################
# Transformação de variáveis.
# Variáveis categóricas codificadas numericamente possuem baixa legibilidade.
# Portanto, podem ser candidatas a serem recodificadas.

new_survived = pd.Categorical(titanic_train["Survived"])
new_survived = new_survived.rename_categories(["Morreu","Sobreviveu"])              

print("\nAnálise do atributo Survived modificado:",new_survived.describe(), sep='\n')

new_Pclass = pd.Categorical(titanic_train["Pclass"],
                           ordered=True)

new_Pclass = new_Pclass.rename_categories(["1aClasse","2aClasse","3aClasse"])     

print("\nAnálise do atributo Pclass modificado:", new_Pclass.describe(), sep='\n')

# No caso de Survided, como esse é o atributo objetivo de uma competição 
# do Kaggle, ela não deveria ser modificada. Entretanto, o atributo Pclass
# pode ficar mais claro se alterarmos sua codificação.

titanic_train["Pclass"] = new_Pclass

# Retornando ao atributo Cabin, 
# parece que o padrão letra+número indica que uma cabine pertence a algum
# andar, ou nível. Veja os valores únicos

print("\nValores únicos do atributo Cabin:",titanic_train["Cabin"].unique(), sep='\n')

# Podemos agrupar o atributo Cabin pela letra inicial da cabine.

# Covnerte o dado para String
char_cabin = titanic_train["Cabin"].astype(str) # Convert data to str

# Pega apenas a primeira letra
new_Cabin = np.array([cabin[0] for cabin in char_cabin])

new_Cabin = pd.Categorical(new_Cabin)

titanic_train["Cabin"] = new_Cabin

print("\nAnálise do atributo Cabin modificado:", new_Cabin.describe(), sep='\n')

# Pela descrição do novo atributo Cabin, pode-se notar que 2/3 dos registros 
# estão com o número da cabine ausente. Isso pode indicar que o atributo não 
# será muito útil para classificação, ou indicar alguma particularidade na forma
# de coleta do dado, que pode ajudar na classificação. Por exemplo, apenas os
# sobreviventes teriam indicado o número da cabine.


###############################################################################
# Valores omissos, extremos ou inconsistentes.
 
# Para detectar valores omissos (isNull())

exemplo = pd.Series([1,None,3,None,7,8])

print("\nValores omissos:", exemplo.isnull(), sep='\n')

# Valores omissos em atributos numéricos
print("\nAnálise do atributo Age:")
print(titanic_train["Age"].describe())

omissos = np.where(titanic_train["Age"].isnull() == True)

print("\nValores omissos no atributo Age:", omissos, sep='\n')

print("\nQuantidade de valores omissos no atributo Age:", 
          len(omissos[0]), sep='\n')

# Possibilidades
# 1 - substituir por zeros
# 2 - Substituir por um valor médio ou mediano.
# 3 - Estimar valores usando modelos estatísticos ou preditivos
# 4 - Particionar a base em registros completos e registros sem Age

# Vamos analisar o dado Age construindo um histograma
titanic_train.hist(column='Age',    # Coluna a ser plotada
                   figsize=(9,6),   # Tamanho do gráfico
                   bins=20)         # Numero de colunas do histogram

mediana = np.median([el for el in titanic_train["Age"] if (np.isnan(el) == False)])
print("\nMediana o atributo Age:", mediana, sep='\n')


new_age_var = np.where(titanic_train["Age"].isnull(), # condição
                       mediana,                       # Valor se verdadeiro
                       titanic_train["Age"])          # Valor se falso

titanic_train["Age"] = new_age_var                   
print("\nAnálise do novo atributo Age:")
print(titanic_train["Age"].describe())

titanic_train.hist(column='Age',    # Coluna a ser plotada
                   figsize=(9,6),   # Tamanho do gráfico
                   bins=20)         # Numero de colunas do histogram

                   
# Detectando outliers (valores estremos)
titanic_train["Fare"].plot(kind="box", figsize=(9,9))

index = np.where(titanic_train["Fare"] == max(titanic_train["Fare"]) )

print("Registros com valores extremos:",titanic_train.loc[index], sep='\n')

###############################################################################
# Criando novos atributos
# Vamos criar uma nova variável Family, que irá unir, conjude e irmãos (SibSp)
# com pais e filhos (Parch)

titanic_train["Family"] = titanic_train["SibSp"] + titanic_train["Parch"]

# Encontrando quem tem a maior família À bordo
most_family = np.where(titanic_train["Family"] == max(titanic_train["Family"]))

print("\nA maior família à bordo:\n{0}"
          .format(titanic_train.ix[most_family]))

# Os atributos agora estão redundantes, ou muito correlacionados, como podemos
# ver com a função de correlação
print("\nÍndice de correlação entre os atributos SibSp e Family:\n{0}"
          .format(np.corrcoef(titanic_train['SibSp'],titanic_train['Family'])))

int_fields = titanic_train.dtypes[titanic_train.dtypes == "int64"].index
corr = np.corrcoef(titanic_train[int_fields].transpose())
correlacao = pd.DataFrame(data=corr, index=int_fields, columns=int_fields)

print("\nMatriz de correlação dos atributos inteiros:\n{0}"
          .format(correlacao))
