'''Gauss-Laguerre Integration
estimate the integral of function e^(-x)*cos(x) from 0 to infinity
using Gauss-Laguerre method for 1, 2 and 3 points
weight function: w(x) = e^(-x); f(x) = cos(x); I = sum(i=0,n)A_i*f(x_i)
Values of the abscissas and weights for n = 1, 2 and 3 given by Kiusalaas
n = 1: x0 = 0.585786, a0 = 0.853554
       x1 = 3.414214, a1 = 0.146447
n = 2: x0 = 0.415775, a0 = 0.711093
       x1 = 2.294280, a1 = 0.278518
       x2 = 6.289945, a2 = 0.010389
n = 3: x0 = 0.322548, a0 = 0.603154
       x1 = 1.745761, a1 = 0.357419
       x2 = 4.536620, a2 = 0.038887
       x3 = 9.395071, a3 = 0.000539 '''

import numpy as np
import matplotlib.pyplot as plt

def f(x):
    return np.cos(x)

def gauss_laguerre(n):
    if n == 1: x_i = (0.585786,3.414214); a_i = (0.853554,0.146447)
    if n == 2: x_i = (0.415775,2.294280,6.289945); a_i = (0.711093,0.278518,0.010389)
    if n == 3: x_i = (0.322548,1.745761,4.536620,9.395071); a_i = (0.603154,0.357419,0.038887,0.000539)
    gl = 0; i = 0
    while i <= n: 
        gl += a_i[i]*f(x_i[i])
        i += 1
    return gl

n = 3
print(f'\nMÉTODO DE GAUSS-LAGUERRE PARA {n} PONTOS\n')

print(f'I_{n-2} = {gauss_laguerre(1):.6f}')
print(f'I_{n-1} = {gauss_laguerre(2):.6f}')
print(f'I_{n} = {gauss_laguerre(3):.6f}\n')

def integrand(x):
    return np.exp(-x)*np.cos(x)

vintegrand = np.polynomial.laguerre.laggauss(100)[1]

x = np.linspace(0,10,100)
plt.plot(x,integrand(x), label='Analytic solution')
plt.plot(x,vintegrand, label='Scipy polynomial')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.legend(loc = 'best')
plt.grid(True)
plt.show()