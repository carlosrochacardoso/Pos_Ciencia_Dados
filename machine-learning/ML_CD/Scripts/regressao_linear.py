# -*- coding: utf-8 -*-
"""
regressao_linear.py: Apresentação de métodos de previsão.

Avaliar a performance de métodos de previsão por reressão linear e por K-NN.

@author: Prof. Hugo de Paula
@contact: hugo@pucminas.br


"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from matplotlib.colors import ListedColormap

def make_wave(n_samples=100):
    rnd = np.random.RandomState(42)
    x = rnd.uniform(-3, 3, size=n_samples)
    y_no_noise = (np.sin(4 * x) + x)
    y = (y_no_noise + rnd.normal(size=len(x))*1.3) / 2
    return x.reshape(-1, 1), y

def plot_linear_regression_wave(X, y, lnr):
    plt.figure(figsize=(8, 8))
    
    cm2 = ListedColormap(['#0000aa', '#ff6030'])
    line = np.linspace(-3, 3, 100).reshape(-1, 1)
    
    plt.plot(X, y, 'o', c=cm2(0))
    plt.plot(line, lnr.predict(line), c=cm2(4))
    ax = plt.gca()
    ax.spines['left'].set_position('center')
    ax.spines['right'].set_color('none')
    ax.spines['bottom'].set_position('center')
    ax.spines['top'].set_color('none')
    ax.set_ylim(-3, 3)
    ax.legend(["Modelo", "Dado de treinamento"], loc="best")
    ax.grid(True)
    ax.set_aspect('equal')

def plot_linear_regression_error(X, knerror, lrerror):
    plt.figure(figsize=(8, 8))
    
    cm2 = ListedColormap(['#0000aa', '#ff6030'])
    
    ylim = max(max(knerror),max(knerror)) + 0.1
    plt.bar(X, knerror)
    plt.plot(X,lrerror,'o', c=cm2(4))
    plt.title("Erro de previsão")
    ax = plt.gca()
    ax.set_ylim(0, ylim)
    ax.legend(["Regressão Linear", "K-NN"], loc="best")



# Produz uma base de dados artificial com 1000 amostras e adiciona ruído a ela.
X, y = make_wave(n_samples=200)

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

#Cria um modelo de regressão com 3 vizinhos. Os valores default são:
# KNeighborsRegressor(algorithm='auto', leaf_size=30, metric='minkowski',
# metric_params=None, n_jobs=1, n_neighbors=3, p=2, weights='uniform')

knreg = KNeighborsRegressor(n_neighbors=3)
knreg.fit(X_train, y_train)

y_knprev = knreg.predict(X_test)

print("Valores reais:\n{}".format(y_test))
print("Valores previstos:\n{}".format(y_knprev))
print("Acurácia da previsão: {:.2f}".format(knreg.score(X_test, y_test)))

print("--------------------------------------")
# Cria um modelo de regressão linear
lnr = LinearRegression().fit(X_train, y_train)

y_lrprev = lnr.predict(X_test)

print("Acurácia da base de treinamento: {:.2f}".format(lnr.score(X_train, y_train)))
print("Acurácia da base de testes: {:.2f}".format(lnr.score(X_test, y_test)))
print("w[0]: %f  b: %f" % (lnr.coef_[0], lnr.intercept_))

# Plota os resultados
plot_linear_regression_wave(X, y, lnr)

plot_linear_regression_wave(X, y, knreg)

