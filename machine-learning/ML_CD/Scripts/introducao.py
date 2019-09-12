# -*- coding: utf-8 -*-
"""
introducao.py: Teste dos principais módulos que deverão estar funcionando.

Apresenta a sintaxe básica o Python e algumas funções de manipulação de dados.

@author: Prof. Hugo de Paula
@contact: hugo@pucminas.br


Baseado no livro: Andreas C. Müller, Sarah Guido (2016)
Introduction to Machine Learning with Python: A Guide for Data Scientists 1st Edition

"""

import IPython
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import sparse
import sklearn

# Versões dos pacotes instalados na plataforma

print("IPython version: %s" % IPython.__version__)
print("matplotlib version: %s" % matplotlib.__version__)
print("numpy version: %s" % np.__version__)
print("pandas version: %s" % pd.__version__)
print("scikit-learn version: %s" % sklearn.__version__)

# Pacote numpy
# Cria uma matriz com duas linhas e três colunas

matriz = np.array([[1, 2, 3], [4, 5, 6]])
print("Matriz com Numpy.array:\n", matriz)

# Cria uma matriz diagonal contendo 1's e as demais posições contendo 0's.

diagonal = np.eye(4)
print("\nMatriz diagonal com Numpy.eye:\n{0}".format(diagonal))


# Converte um arranjo do Numpy para a representação de matriz esparsa
# no formato CSR (Compressed Row Storage) do Scipy. Apenas os valores 
# não nulos são armazenados.

matriz_esparsa = sparse.csr_matrix(diagonal)
print("\nMAtriz esparsa no formato Scipy.CSR:\n{0}\n".format(matriz_esparsa))

# Gera uma sequencia de números inteiros de 0 a 20, em intervalos de 0.2
x = np.arange(0, 20, 0.2)

# Aplica a função coseno ao vetor de inteiros gerado anteriormente
y = np.sin(x)

# Exibe um gráfico de linha simples de cor vermelha de y em função de x.
plt.plot(x, y, marker="x", color='#d62728')


# Cria um dataset de pessoas
data = {'Nome': ["Hugo", "Ana", "Bernardo", "Felipe"],
'Local' : ["Kyoto", "New York", "Orlando", "Santiago"],
'Idade' : [40, 39, 7, 4]
}

data_pandas = pd.DataFrame(data)
print("\n\nTabela de dados pandas:\n{0}\n".format(data_pandas))


# Construindo DataFrames a partir de arranjos
lin_nomes = ['A', 'B']
col_nomes = ['UM', 'DOIS', 'TRÊS']
novo_dataframe = pd.DataFrame(matriz, index=lin_nomes, columns=col_nomes)
print("\n\nTabela de dados pandas:\n{0}\n".format(novo_dataframe))



