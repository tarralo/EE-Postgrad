'''TRABALHO 6 - MÉTODOS COMPUTACIONAIS
Método de Euler Modificado
LUIZ TARRALO - 2023.1'''

import numpy as np

def f(x, y):
    return -x / y

def euler_modified_method(x, y, h):
    n = len(x)
    for i in range(n - 1): y[i + 1] = y[i] + h * f(x[i] + h / 2, y[i] + h / 2 * f(x[i], y[i]))
    return y

x = np.linspace(0, 0.2, 4)
y = np.zeros(len(x))
y[0] = 1
h = 0.05

y = euler_modified_method(x, y, h)

print("y'(0.2) =", y[3])


