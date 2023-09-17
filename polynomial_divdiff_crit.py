'''Trabalho 4 de Métodos Computacionais
Polinômio pelo critério de diferenças divididas
Autor: Luiz Tarralo'''

import numpy as np

def divided_differences_interpolation(x, y, x_val):
    n = len(x) # número de pontos
    coef = np.zeros([n, n])
    coef[:, 0] = y # primeira coluna recebe os valores de y

    for j in range(1, n):
        for i in range(n - j):
            coef[i][j] = (coef[i + 1][j - 1] - coef[i][j - 1]) / (x[i + j] - x[i]) # calcula os coeficientes

    p = coef[0][0]
    equation = str(coef[0][0])
    for i in range(1, n):
        term = coef[0][i] # termo atual
        if term != 0:
            equation += " + "
            if term != 1:
                equation += str(term) # adiciona o coeficiente
            power_terms = []
            for j in range(i):
                power_terms.append("(x - " + str(x[j]) + ")") # adiciona os termos de potência
            equation += "*".join(power_terms) # junta os termos de potência
        p += term * np.prod(x_val - np.array(x[:i])) # calcula o valor do polinômio para x_val

    print("Equação polinomial: P(x) =", equation)
    return p

x = [0, 1, 2, 3] # pontos de x
y = [1, 1, 15, 61] # valores de y
x_val = 2.5 # valor de x para o qual queremos calcular P(x)
print("MÉTODO DE DIFERENÇAS DIVIDIDAS - POLINÔMIO INTERPOLADOR\n")

result = divided_differences_interpolation(x, y, x_val)

print("P(x) para x =", x_val, "é", result)