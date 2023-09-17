'''Trabalho 1 - Métodos Numéricos
   Método Iterativo de Jacobi para sistemas lineares - forma: Ax = b
   Autor: Luiz Arthur Tarralo Passatuto'''

import numpy as np

def Jacobi(A, b, x0, tol, maxiter):
    
    n = len(b) # tamanho do vetor b

    x = np.zeros(n) # vetor solução

    iter = 0 # contador de iterações

    for iter in range(maxiter): # iterações

        for i in range(n): # calcula o valor de x[i]

            soma = 0 

            for j in range(n): # soma dos termos da linha i

                if j != i:

                    soma += A[i,j]*x0[j]

            x[i] = (b[i] - soma)/A[i,i] # calcula o valor de x[i]

        if norm(x-x0) < tol: # critério de parada

            if iter > 1:

                print(f" Convergiu em {iter} iterações.")

                return x
            
            else:
                    
                print(f" Convergiu em {iter} iteração.")
    
                return x
        
        iter += 1 # atualiza o contador de iterações
        
        x0 = x.copy() # atualiza o vetor x0

    print(f" Não convergiu em {maxiter} iterações.")

    return x

def norm(x): # retorna a norma da matriz 
    
    return np.sqrt(np.sum(x**2))

def main():

    A = np.array([[1, 2, -2, 1], 
                  [2, 5, -2, 3],
                  [-2, -2, 5, 3],
                  [1, 3, 3, 2]]) # matriz A
    
    b = np.array([4, 7, -1, 0]) # vetor b

    chute = x0 = np.array([0, 0, 0, 0]) # chute inicial

    tol = 1e-5 # tolerância

    maxiter = 1000 # número máximo de iterações

    x = Jacobi(A, b, x0, tol, maxiter) # chamada da função Jacobi

    x = np.nan_to_num(x, nan = 0) # substitui os valores NaN por 0
    
    sol = np.linalg.solve(A, b) # solução exata

    np.set_printoptions(precision = 4) # define a precisão dos números impressos em 4 casas decimais

    print(f" Vetor x obtido = {x}. \n Tolerância de {tol}. \n Chute inicial {chute}.")

    print(f" Solução exata = {sol}.")
 
main() 
