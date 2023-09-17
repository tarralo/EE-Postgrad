'''Autor: Luiz Arthur Tarralo Passatuto
Minimização de uma função de 1 variável utilizando Algoritmos Genéticos'''

import numpy as np
import pandas as pd
import random
import math
import matplotlib.pyplot as plt
from PyQt5 import QtGui

# função principal
def main():

    pop_length = 50 # quantidade de indivíuos na população
    ind_length = 10 # quantidade de bits em cada indivíduo

    pc = 0.6 # probabilidade de cruzamento
    pm = 0.01 # probabilidade de mutação
    num_gen = 100 # número de gerações

    x_min = 0 # limite inferior do domínio da função
    x_max = 512 # limite superior do domínio da função

    pop = generate_pop(pop_length, ind_length) # gera a população inicial

    data_list = [] # cria uma lista para armazenar os dados da geração

    # dá início a evolução da população através das gerações
    
    for k in range(num_gen):

        fit_score = fitness(pop, ind_length, x_min, x_max) # calcula a aptidão de cada indivíduo

        best_fit = min(fit_score) # calcula a melhor aptidão da geração, ou seja, o valor mínimo

        best_ind = fit_score.index(best_fit) # encontra o índice do indivíduo com a melhor aptidão

        best_crom = pop[best_ind] # define o cromossomo do indivíduo com a melhor aptidão

        best_x = decode_ind(best_crom, ind_length, x_min, x_max) # decodifica o cromossomo do indivíduo com a melhor aptidão

        data = {'Geração': k + 1, 'Melhor Indivíduo': best_x, 'Fitness': best_fit} # cria um dicionário com os dados da geração
        
        data_list.append(data) # adiciona os dados da geração na lista

        new_pop = [best_crom] # define a nova população com o melhor indivíduo da geração anterior

        for _ in range(pop_length):

            old_ind_1 = roulette_whell(pop, fit_score) # seleciona o indivíduo 1 por roleta
            old_ind_2 = roulette_whell(pop, fit_score) # seleciona o indivíduo 2 por roleta

            new_ind_1, new_ind_2 = crossover(old_ind_1, old_ind_2, pc) # ocorre o cruzamento entre os indivíduos selecionados

            new_ind_1, new_ind_2 = mutation(new_ind_1, pm), mutation(new_ind_2, pm) # ocorre a mutação nos novos indivíduos

            new_pop.append(new_ind_1) # adiciona o novo indivíduo 1 na nova população
            new_pop.append(new_ind_2) # adiciona o novo indivíduo 2 na nova população

        pop = new_pop # a nova geração ocupa o lugar da geração anterior

    save_excel(data_list) # salva os dados da geração em um arquivo xlsx

    graph_plot(best_x) # plota o gráfico da função objetivo com a solução destacada

    return

# função objetivo (função a ser minimizada)
def obj_func(x):

    return -abs(x * math.sin(math.sqrt(abs(x)))) # retorna o valor da função objetivo

# função para gerar um indivíduo aleatório
def generate_ind(ind_length):

    return [random.randint(0,1) for _ in range(ind_length)] # gera um indivíduo aleatório

# função para compor nova geração 
def generate_pop(pop_length, ind_length):

    return [generate_ind(ind_length) for _ in range(pop_length)] # gera uma população de indivíduos aleatórios

# função para decodificar o indivíduo
def decode_ind(indiv, ind_length, x_min, x_max):
    
    indiv_str = ''.join(str(bit) for bit in indiv) # transforma a lista de bits em uma string

    x = int(indiv_str, 2) # transforma a string em um número inteiro, o número 2 indica que a string está em binário 

    return x_min + (x * ((x_max - x_min) / (2 ** ind_length - 1))) # transforma o número inteiro em um número real

# função para calcular a aptidão de cada indivíduo
def fitness(pop, ind_length, x_min, x_max):

    return [obj_func(decode_ind(indiv, ind_length, x_min, x_max)) for indiv in pop] # decodifica o indivíduo e calcula a aptidão do indivíduo

# função para seleção de indivíduos por roleta
def roulette_whell(pop, fit_score):
    
    pop_fit = sum(fit_score) # soma a aptidão de todos os indivíduos da população

    select_prob = [fit / pop_fit for fit in fit_score] # calcula a probabilidade de seleção de cada indivíduo

    return random.choices(pop, weights=select_prob)[0] # seleciona um indivíduo por roleta

# função para cruzamento e geração de novos indivíduos
def crossover(old_ind_1, old_ind_2, pc):

    r = random.random() # gera um número aleatório entre 0 e 1

    if r < pc: # se o número aleatório for menor que a taxa de cruzamento, então ocorre o cruzamento

        crossover = random.randint(1, len(old_ind_1) - 1) # escolhe um ponto de cruzamento aleatório

        new_ind_1 = old_ind_1[:crossover] + old_ind_2[crossover:] # gera o novo indivíduo 1

        new_ind_2 = old_ind_2[:crossover] + old_ind_1[crossover:] # gera o novo indivíduo 2

    else: # se o número aleatório for maior que a taxa de cruzamento, então os indivíduos não sofrem cruzamento

        new_ind_1 = old_ind_1 # o novo indivíduo 1 é igual ao indivíduo 1

        new_ind_2 = old_ind_2 # o novo indivíduo 2 é igual ao indivíduo 2

    return new_ind_1, new_ind_2

# função para ocorrer mutação em um indivíduo
def mutation(indiv, pm):

    for i in range(len(indiv)):

        r = random.random() # gera um número aleatório entre 0 e 1

        if r < pm: # se o número aleatório for menor que a taxa de mutação, então ocorre a mutação

            indiv[i] = 1 - indiv[i] # inverte o bit

    return indiv

# função para criação do arquivo de dados da geração em excel 
def save_excel(data_list):

    df = pd.DataFrame(data_list, columns = ['Geração', 'Melhor Indivíduo', 'Fitness']) # cria um dataframe com os dados da geração

    df.index += 1 # define o índice do dataframe

    df.to_excel('dados_geracao.xlsx', sheet_name = 'Dados das gerações', float_format = "%.4f", index = False) # salva os dados da geração em um arquivo xlsx

    return

# função para plotar o gráfico da função objetivo com os valores de x
def graph_plot(best_x):

    x_axis = range(512) # define o eixo x

    y_axis = [obj_func(x) for x in x_axis] # define o eixo y

    fig, ax = plt.subplots() # cria o gráfico e os eixos
    ax.plot(x_axis, y_axis) # plota o gráfico

    ax.scatter(best_x, obj_func(best_x), color='red') # plota o ponto da solução encontrada        

    # adiciona uma anotação no gráfico, com parâmetros de posição, texto, seta, caixa de texto e cor
    ax.annotate('Mínimo global:' + "\n" + 'x = ' + "{: .5f}".format(best_x)  + "\n" + 'f(x) = ' + "{: .2f}".format(obj_func(best_x)), \
                (best_x, obj_func(best_x)), xytext=(100, -350), arrowprops=dict(facecolor='black', shrink=0.05), \
                bbox=dict(boxstyle='round', fc='white', ec='black', lw=1, alpha=0.9)) 
    
    ax.set_xlabel('Eixo de x') # define o nome do eixo x
    ax.set_ylabel('Eixo de f(x)') # define o nome do eixo y

    function = 'f(x) = -|x * sen(√(|x|))|' # define a função

    ax.set_title(f'Gráfico de {function}') # define o título do gráfico
    ax.set_facecolor('white') # define a cor do fundo do gráfico

    win = plt.gcf().canvas.manager.window # define a janela do gráfico
    win.setWindowTitle('Minização de ' + function + ' por Algoritmo Genético') # define o título da janela do gráfico
    
    plt.grid(True, which = 'both', axis = 'both', linestyle = 'dashed') # adiciona uma grade no gráfico
    plt.minorticks_on() # adiciona as marcas menores no gráfico
    plt.savefig('graf_f(x).png', dpi=300, format = 'png', bbox_inches='tight') # salva o gráfico como uma imagem png
    plt.show() # mostra o gráfico

main()
