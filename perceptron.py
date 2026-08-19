# Utilizando Bias

import numpy as np

X = np.array([
    [-1, 0, 0],
    [-1, 0, 1],
    [-1, 1, 0],
    [-1, 1, 1]
])

Y = np.array([0, 1, 1, 1])          # Saídas esperadas (Tabela Verdade da porta OR)
W = np.array([-0.5, 0.3, -0.2])     # Pesos iniciais (incluindo o peso do bias)

alpha = 0.5
epocas = 4

# Função de ativação degrau
def degrau(u):
    return 1 if u >= 0 else 0

# Treinamento do perceptron
for epoca in range(epocas):

    print(f"Época {epoca + 1}")

    for i in range(len(X)):
        u = np.dot(W, X[i])         # Soma ponderada
        o = degrau(u)               # Saída do perceptron
        erro = Y[i] - o             # Cálculo do erro

        if erro != 0:
            W += alpha * erro * X[i]    # Atualização dos pesos
            print(f"Pesos atualizados: {W}")
        
        print(f"Entrada: {X[i][1:]}, Saída: {o}, Erro: {erro}, Pesos: {W}")

    print("-" * 50)

# Testando o perceptron treinado
for i in range(len(X)):
    u = np.dot(W, X[i])         # Soma ponderada
    o = degrau(u)               # Saída do perceptron
    print(f"Entrada: {X[i][1:]}, Saída final: {o}")