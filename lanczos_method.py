'''Trabalho 4 de Métodos Computacionais
Polinômio pelo critério de diferenças divididas
Autor: Luiz Tarralo'''

import numpy as np
from scipy import linalg as la

def lanczos(A, v, n):
    '''Iterações de Lanczos para transformar uma matriz simétrica em uma matriz tridiagonal'''
    m = len(A) # número de linhas da matriz A
    Q = np.zeros((m, n)) # matriz Q de autovetores
    Q[:,0] = v/la.norm(v) # preenche primeira coluna de Q com v normalizado
    alpha = np.zeros(n) # diagonal principal da matriz tridiagonal
    beta = np.zeros(n-1) # diagonais secundárias da matriz tridiagonal
    for j in range(1,n): # j = 1, 2, ..., n-1
        w = np.dot(A, Q[:,j-1]) # w é o produto de A com a coluna anterior de Q
        alpha[j-1] = np.dot(w, Q[:,j-1]) # alfa é o produto de w com a coluna anterior de Q
        w = w - alpha[j-1]*Q[:,j-1] - beta[j-1]*Q[:,j-2] # w é a subtração de w com a soma dos produtos de alfa e beta com as colunas anteriores de Q
        beta[j-1] = la.norm(w) # beta é a norma de w
        Q[:,j] = w/beta[j-1]   # coluna j de Q é w normalizado
    T = np.diag(alpha) + np.diag(beta, k=1) + np.diag(beta, k=-1) # matriz tridiagonal
    print('MATRIZ TRIDIAGONALIZADA') 
    print(T)
    return la.eig(T) # eigenvalues and eigenvectors of T

def main():
    print('PROCESSO DE LANCZOS PARA REDUÇÃO DE MATRIZ SIMÉTRICA EM TRIDIAGONAL\nAUTOR: LUIZ TARRALO')
    A = np.array([[1,1,1,1],[1,2,2,2],[1,2,3,3],[1,2,3,4]]) # matriz simétrica A
    v = np.array([0.5, 0.5, 0.5, 0.5]) # vetor inicial v
    n = 4 # número de iterações
    eig_val, eig_vec = lanczos(A, v, n) # autovalores e autovetores de A
    print('Autovalores: \n', eig_val, '\n') # autovalores de A
    print('Autovetores: \n', eig_vec, '\n') # autovetores de A

if __name__ == '__main__':
    main()