'''Trabalho 2 - Métodos Numéricos
   Processo de Newton-Raphson para sistemas não lineares
   2x1^2 - 4x1x2 + 2x2^2 = 0
   3x2^2 + 6x1 - x1^2 - 4x1x2 - 5 = 0
   chute inicial x1=x2=1
   tol = 1e-5
   PARA TESTAR A OUTRA FUNÇÃO, APENAS TIRE O # E COMENTE A OUTRA PARA EVITAR CONFLITO
   Autor: Luiz Arthur Tarralo Passatuto'''

import sympy as sp
import numpy as np

# define a função
def funcao(x1,x2):
    return np.array([2*x1**2 - 4*x1*x2 + 2*x2**2, 3*x2**2 + 6*x1 - x1**2 - 4*x1*x2 - 5]) # função proposta no trabalho
    #return np.array([x1*x2 - 1, x1**2 + x2**2 - 4]) # função comparativa do griffiths

# define a Jacobiana
def jacobiana(x):

    x1, x2 = sp.symbols('x1 x2') # define as variáveis simbólicas

    # calcula a derivada parcial de cada função em relação a cada variável
    df1dx1 = sp.diff(funcao(x1,x2)[0], x1) # calcula a derivada parcial de f1 em relação a x1
    df1dx2 = sp.diff(funcao(x1,x2)[0], x2) # calcula a derivada parcial de f1 em relação a x2
    df2dx1 = sp.diff(funcao(x1,x2)[1], x1) # calcula a derivada parcial de f2 em relação a x1
    df2dx2 = sp.diff(funcao(x1,x2)[1], x2) # calcula a derivada parcial de f2 em relação a x2

    # substitui os valores de x1 e x2 pelos valores de x
    df1dx1 = df1dx1.subs({x1:x[0],x2:x[1]}) # substitui x1 por x[0] e x2 por x[1]
    df1dx2 = df1dx2.subs({x1:x[0],x2:x[1]}) # substitui x1 por x[0] e x2 por x[1]
    df2dx1 = df2dx1.subs({x1:x[0],x2:x[1]}) # substitui x1 por x[0] e x2 por x[1]
    df2dx2 = df2dx2.subs({x1:x[0],x2:x[1]}) # substitui x1 por x[0] e x2 por x[1]

    # cria a matriz jacobiana 2x2
    j = np.zeros((2,2), dtype=float) # cria uma matriz 2x2 de zeros
    j[0,0] = df1dx1 # atribui o valor de df1dx1 à posição [0,0] da matriz jacobiana
    j[0,1] = df1dx2 # atribui o valor de df1dx2 à posição [0,1] da matriz jacobiana
    j[1,0] = df2dx1 # atribui o valor de df2dx1 à posição [1,0] da matriz jacobiana
    j[1,1] = df2dx2 # atribui o valor de df2dx2 à posição [1,1] da matriz jacobiana

    return j

# define a função do Processo de Newton-Raphson
def newtonRaphson(x0, tol, max_iter):

    erro = 1 # define o erro inicial

    #cria vetor para armazenar delta x1 e delta x2
    dx = np.zeros((2,1), dtype=float)

    x1, x2 = sp.symbols('x1 x2') # define as variáveis simbólicas
    print(f'\nPROCESSO DE NEWTON-RAPHSON PARA SISTEMAS NÃO-LINEARES\nFunções: {funcao(x1,x2)[0], funcao(x1,x2)[1]}\nChute inicial x1 = {x0[0]}, x2 = {x0[1]}\nTolerância = {tol}\nNúmero máximo de iterações = {max_iter}\n')

    for iter in range(max_iter):
        if erro < tol: 
            break
        else:
            j = jacobiana(x0)

            # checa se a matriz jacobiana é singular
            if np.linalg.det(j) == 0: break

            # calcula a inversa da matriz jacobiana
            j_inv = np.linalg.inv(j)

            # calcula as funções para o chute atual
            f = funcao(x0[0],x0[1])

            # calcula delta x1 e delta x2
            dx = np.dot(j_inv,-f.T)

            # calcula o novo chute
            x0[0] = x0[0] + dx[0]
            x0[1] = x0[1] + dx[1]

            # calcula o erro
            erro = np.linalg.norm(dx)

            # imprime o resultado
            print(f'Iteração {iter+1}: x1 = {x0[0]:.6f}, x2 = {x0[1]:.6f}, erro = {erro:.6e}.')

    if iter+1 == max_iter: print(f'O número máximo de {max_iter} iterações foi atingido, o sistema não convergiu.\n')
        
    else:
        if np.linalg.det(j) == 0: print(f'Como a matriz jacobiana é singular, o método não converge. \nJ={j} \nDeterminante(J)={np.linalg.det(j)}.\n')

        else:
            print(f'\nO sistema convergiu após {iter} iterações. \nRaízes encontradas: x1 = {x0[0]:.6f}, x2 = {x0[1]:.6f}.')
            print(f'Erro final = {erro:.6e}.\n')

def main():

    # define o chute inicial
    x0 = np.array([1,1]) # chute inicial trabalho
    #x0 = np.array([1.8,0.5]) # chute inicial griffiths

    # define a tolerância
    tol = 1e-5

    # define o número máximo de iterações
    max_iter = 100

    # chama a função do Processo de Newton-Raphson
    newtonRaphson(x0, tol, max_iter)

main()