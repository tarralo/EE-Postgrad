'''ALGORITMO GENÉTICO DENTRO DE UMA INTERFACE GRÁFICA (GUI) PARA MAXIMIZAÇÃO DE UMA FUNÇÃO COM DUAS VARIÁVEIS 
   FUNÇÃO f(x,y) = 10 + x * sen(4x) + 3 * sen(2y)
   VARIÁVEIS x E y VARIAM ENTRE 0 E 4 E 0 E 2, RESPECTIVAMENTE
   COM PLOT 2D E 3D
   AUTOR: Luiz Tarralo'''

import random
import math
import tkinter as tk
from numpy import amax
from numpy import linspace
from numpy import meshgrid
from numpy import sin
from numpy import argmax
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from tkinter import ttk

PRIMARY_COLOR = "#3366CC"
SECONDARY_COLOR = "#DCDCDC"      

# define a função principal
def main(num_gen, ind_size, pop_length, pc, pm, select_type, frame_2d, frame_3d):

    # se na primeira execução, chama a função que cria a GUI
    if num_gen == None:
            
        gui()

    # chama a função que cria a população inicial
    pop = create_pop(pop_length, ind_size)

    # cria listas para armazenar best_fit, avg_fit, best_ind de cada geração
    best_fit = []
    avg_fit = []
    best_ind = []

    # cria listas para armazenar x, y e z de cada geração
    x_list = []
    y_list = []
    z_list = [] # z é a função objetivo

    # processo de evolução em loop para num_gen gerações
    for i in range(num_gen):

        # chama a função de decodificação 
        x, y = decode(pop, ind_size)

        # chama a função de avaliação
        z = [obj_f(x[j], y[j]) for j in range(pop_length)]

        # armazena o melhor fitness, o fitness médio e o melhor indivíduo da geração atual
        best_fit.append(amax(z))
        avg_fit.append(sum(z) / pop_length)
        best_ind.append(pop[z.index(amax(z))])

        # armazena x, y e z do melhor indivíduo da geração atual
        x_list.append(x[z.index(amax(z))])
        y_list.append(y[z.index(amax(z))])
        z_list.append(amax(z))

        if i != 0: # se não for a primeira geração

            # checa se o melhor indivíduo da geração atual é melhor que o melhor indivíduo da geração anterior
            # se for, armazena o melhor indivíduo da geração atual como elite   
            # se não, armazena o melhor indivíduo da geração anterior como elite
            # o elite é o indivíduo que será mantido na próxima geração
            if amax(z) > best_fit[i - 1]:
                elite = pop[z.index(amax(z))]

            else:
                elite = pop[z.index(best_fit[i - 1])]
        
        else: # se for a primeira geração, armazena o melhor indivíduo da geração atual como elite

            elite = pop[z.index(amax(z))]

        # chama a função de seleção dos pais
        parents = select_parents(pop, z, pop_length, select_type)

        # chama a função de crossover
        offspring = crossover(parents, pc, ind_size)

        # chama a função de mutação
        offspring = mutation(offspring, pm, ind_size)

        offspring.append(elite)

        # substitui os indivíduos menos aptos da população atual pelos filhos gerados
        pop = replace(pop, offspring, z)

    # chama a função que plota o gráfico 2D com duas curvas: melhor fitness e fitness médio por geração
    plot_2d(num_gen, best_fit, avg_fit, frame_2d)

    # chama a função que plota a superfície 3D da função objetivo, recebendo como parâmetros as listas x, y e z
    plot_3d(x_list, y_list, z_list, frame_3d)
    
    return 0

# define a função que cria a população inicial
def create_pop(pop_length, ind_size):
    
    return [create_ind(ind_size) for _ in range(pop_length)] # cria uma lista de indivíduos com tamanho pop_length

# define a função que cria um indivíduo
def create_ind(ind_size):

    return random.choices([0,1], k = ind_size) # cria uma lista de 0s e 1s com tamanho ind_size

# define a função que retorna a função objetivo
def obj_f(x, y):
    
    return 10 + x * math.sin(4 * x) + 3 * math.sin(2 * y)

# define a função de decodificação
def decode(pop, ind_size):

    x_size = int(ind_size / 2)
    y_size = int(ind_size / 2)

    # decodifica x para cada indivíduo com os limites de 0 e 4
    x = [int(''.join(map(str, pop[j][:x_size])), 2) * 4 / (2 ** x_size - 1) for j in range(len(pop))]

    # decodifica y para cada indivíduo com os limites de 0 e 2
    y = [int(''.join(map(str, pop[j][x_size:])), 2) * 2 / (2 ** y_size - 1) for j in range(len(pop))]
    
    return x, y

# define a função de seleção dos pais
def select_parents(pop, z, pop_length, select_type):

    # cria uma lista vazia para armazenar os pais
    parents = []

    # se select_type for roleta
    if select_type == 'Roleta': 

        total_fit = sum(z) # calcula o fitness total da população
        prob = [z[i] / total_fit for i in range(pop_length)] # calcula a probabilidade de cada indivíduo ser selecionado
        prob_acum = [sum(prob[:i + 1]) for i in range(pop_length)] # calcula a probabilidade acumulada de cada indivíduo ser selecionado
        
        # seleciona dois pais aleatoriamente com base na probabilidade acumulada, sem repetição de pais
        for _ in range(2):

            r = random.random() # gera um número aleatório entre 0 e 1

            for i in range(pop_length): # percorre a lista de probabilidade acumulada

                if r <= prob_acum[i]: # se o número aleatório for menor ou igual à probabilidade acumulada do indivíduo i

                    parents.append(pop[i]) # adiciona o indivíduo i à lista de pais

                    break

    # se select_type for torneio
    elif select_type == 'Torneio':

        pop_copy = pop.copy() # cria uma cópia da população

        # seleciona dois pais aleatoriamente com base no torneio, sem repetição de pais
        for _ in range(2):

            # seleciona três indivíduos aleatoriamente
            inds = random.sample(pop_copy,3)

            # pega o fitness de cada indivíduo
            fit_1 = z[pop.index(inds[0])]
            fit_2 = z[pop.index(inds[1])]
            fit_3 = z[pop.index(inds[2])]

            # seleciona o melhor indivíduo entre os três
            if fit_1 > fit_2 and fit_1 > fit_3:
                parents.append(inds[0])

            elif fit_2 > fit_1 and fit_2 > fit_3:
                parents.append(inds[1])

            else:
                parents.append(inds[2])

    return parents

# define a função de crossover
def crossover(parents, pc, ind_size):

    # cria uma lista vazia para armazenar os descendentes
    offspring = []

    # se o número aleatório for menor que a probabilidade de crossover, realiza o crossover
    if random.random() < pc:
            
            # seleciona um ponto de corte aleatório
            cut = random.randint(1, ind_size - 1)
    
            # cria os dois descendentes com base nos dois pais e no ponto de corte
            offspring1 = parents[0][:cut] + parents[1][cut:]
            offspring2 = parents[1][:cut] + parents[0][cut:]
    
            # adiciona os dois descendentes à lista de descendentes
            offspring.append(offspring1)
            offspring.append(offspring2)
    
    # se o número aleatório for maior que a probabilidade de crossover, os dois pais são os dois descendentes
    else:
        offspring = parents

    return offspring

# define a função de mutação
def mutation(offspring, pm, ind_size):

    # se um número aleatório for menor que a probabilidade de mutação, realiza a mutação do offspring 1 e 2
    if random.random() < pm:
            
        # seleciona um bit aleatório para ser mutado
        bit = random.randint(0, ind_size - 1)

        # inverte o bit selecionado
        if offspring[0][bit] == 0:
            offspring[0][bit] = 1

        else:
            offspring[0][bit] = 0

        if offspring[1][bit] == 0:
            offspring[1][bit] = 1

        else:
            offspring[1][bit] = 0

    return offspring

# define a função de substituição
def replace(pop, offspring, z):

    # cria uma cópia da população atual e ordena os fitness em ordem crescente
    sorted_pop = pop.copy()
    sorted_pop.sort(key=lambda ind: z[pop.index(ind)])

    # substitui os indivíduos menos aptos pelos dois descendentes
    for i in range(len(offspring)):
        sorted_pop[i] = offspring[i]

    return sorted_pop

# define a função que plota o gráfico 2D com duas curvas: melhor fitness e fitness médio por geração denro de frame_2d
def plot_2d(num_gen, best_fit, avg_fit, frame_2d):

    # cria uma lista com o número de gerações, que será o eixo x do gráfico
    x = [i for i in range(num_gen)]

    fig = Figure(figsize=(6, 5), dpi=70) # cria um objeto figura
    ax = fig.add_subplot(111) # cria um objeto eixo
    ax.plot(x, best_fit, label='Melhor fitness') # plota a curva do melhor fitness
    ax.plot(x, avg_fit, label='Fitness médio') # plota a curva do fitness médio
    ax.set_xlabel('Geração') # define o rótulo do eixo x
    ax.set_ylabel('Fitness') # define o rótulo do eixo y
    ax.legend() # adiciona a legenda
    ax.grid(True) # adiciona uma grade ao gráfico
    canvas = FigureCanvasTkAgg(fig, master = frame_2d)
    canvas.draw()
    canvas.get_tk_widget().grid(row=0, column=0, sticky='w')

# define a função que plota a superfície 3D da função objetivo com destaque para os pontos nas listas x, y e z dentro de frame_3d
def plot_3d(x_list, y_list, z_list, frame_3d):

    # cria uma lista com os valores de x e y para plotar a superfície 3D
    x = linspace(0, 4, 100)
    y = linspace(0, 2, 100)
    x, y = meshgrid(x, y)

    # calcula os valores da função objetivo 10 + x*sin(4*x) + 3*sin(2*y) para cada par de valores de x e y
    z = 10 + x*sin(4*x) + 3*sin(2*y)

    # cria a figura 3D dentro de frame_3d
    fig = Figure(figsize=(5, 5), dpi=150)
    ax = fig.add_subplot(111, projection='3d')

    # ajusta as margens ao redor do plot
    fig.subplots_adjust(left = 0.05, right = 0.95, bottom = 0.05, top = 0.95)

    # ajusta os espaços entre os subplots
    fig.subplots_adjust(wspace = 0.1, hspace = 0.1)

    # cria o canvas para plotar a figura 3D dentro de frame_3d
    frame_3d = FigureCanvasTkAgg(fig, master=frame_3d)
    frame_3d.draw()

    # plota a superfície 3D da função objetivo dentro de frame_3d
    ax.plot_surface(x, y, z, cmap='viridis', edgecolor='none')

    # Pega o índice do maior valor da lista z_list
    max_index = argmax(z_list)

    # plota os pontos nas listas x, y e z dentro de frame_3d
    ax.scatter(x_list, y_list, z_list, c='r', marker='o', s=100, edgecolor='k', linewidth=1, label = 'Melhores indivíduos')

    max_z = z_list[max_index]
    max_x = x_list[max_index]
    max_y = y_list[max_index]
    text = f'Máximo global encontrado pelo AG: \n (x,y,z) = ({max_x:.4f}, {max_y:.4f}, {max_z:.4f})'
    ax.text2D(0.1, 0.9, text, transform=ax.transAxes, fontsize=10, bbox=dict(facecolor='white', alpha=0.5))

    # define o título do gráfico 3D dentro de frame_3d
    ax.set_title('Superfície 3D da Função objetivo')

    # define os rótulos dos eixos x, y e z dentro de frame_3d
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')

    # define a legenda dentro de frame_3d
    ax.legend(loc='lower right')

    # atualiza o gráfico
    frame_3d.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

# define a função da interface gráfica
def gui():

    def callback(gen, pop, ind, pc, pm, sel): # define a função que será chamada quando o botão 'Executar' for clicado

        # obtém os valores dos widgets de entrada de dados
        num_gen = int(gen.get())
        pop_size = int(pop.get())
        ind_size = int(ind.get())
        prob_crossover = float(pc.get())
        prob_mutation = float(pm.get())
        selection = sel.get()

        # create a frame for the 2D graph on the left down corner of the window
        frame_2d = tk.Frame(window, width = 500, height = 400, bg = SECONDARY_COLOR, highlightbackground= 'black', highlightthickness=2)
        frame_2d.grid(row = 1, column = 0, padx = 10, pady = 10)

        # create a frame for the 3d graph on the right side of the window
        frame_3d = tk.Frame(window, width = 500, height = 500, bg = SECONDARY_COLOR, highlightbackground= 'black', highlightthickness=2)
        frame_3d.grid(row = 0, column = 1, padx = 10, pady = 10, rowspan = 2)

        # passa os dados e os frames para a função principal do algoritmo genético
        main(num_gen, pop_size, ind_size, prob_crossover, prob_mutation, selection, frame_2d, frame_3d)

    window = tk.Tk() # cria a janela
    window.title('Algoritmo Genético para Maximização de f(x,y) - Luiz Tarralo') # define o título da janela
   
    # tamanho da janela fullscreen
    window.geometry('1200x1000')
    # janela redimensionável
    window.resizable(True, True)

    window.columnconfigure(1, weight=1)
    window.configure(bg = PRIMARY_COLOR)

    # cria um frame para os widgets de entrada de dados no canto superior esquerdo da janela
    frame_w = tk.Frame(window, width = 250, height = 250, bg = 'green', highlightbackground= 'black', highlightthickness=2)
    frame_w.grid(row = 0, column = 0, padx = 10, pady = 10)

    # chama a função que cria os widgets para a entrada de dados
    create_widgets(frame_w, callback)

    # cria um botão para fechar a janela
    button_quit = tk.Button(window, text = 'Sair', font = ('Roboto Mono Regular', 10), command = window.destroy, relief = 'solid', bd = 1)
    button_quit.grid(row = 2, column = 0, padx = 10, pady = 10)
 
    # mantém a janela aberta
    window.mainloop()

# define a função que cria os widgets para a entrada de dados
def create_widgets(frame_w, callback):

    # cria um label para entrada do número de gerações
    label_gen = tk.Label(frame_w, text = 'NÚMERO DE GERAÇÕES', font = ('Roboto Mono Regular', 10), fg = 'green', bg = 'white', relief = 'solid', bd = 1)
    label_gen.grid(row = 0, column = 0, padx = 10, pady = 10, sticky = 'w')

    # cria um entry para entrada do número de gerações
    entry_gen = tk.Entry(frame_w, width = 10, font = ('Arial', 10), relief = 'solid', bd = 2)
    entry_gen.grid(row = 0, column = 1, padx = 10, pady = 10, sticky = 'w')

    # quando o usuário pressiona a tecla Enter, checa se o valor digitado é inteiro, maior que zero e menor que 1000
    # se não for um valor válido, exibe uma mensagem de erro e limpa o entry
    entry_gen.bind('<Return>', lambda event: check_gen(entry_gen))

    # cria um label para entrada do tamanho da população
    label_pop = tk.Label(frame_w, text = 'TAMANHO DA POPULAÇÃO', font = ('Roboto Mono Regular', 10), fg = 'green', bg = 'white', relief = 'solid', bd = 1)
    label_pop.grid(row = 1, column = 0, padx = 10, pady = 10, sticky = 'w')

    # cria um entry para entrada do tamanho da população
    entry_pop = tk.Entry(frame_w, width = 10, font = ('Arial', 10), relief = 'solid', bd = 2)
    entry_pop.grid(row = 1, column = 1, padx = 10, pady = 10, sticky = 'w')

    # quando o usuário pressiona a tecla Enter, checa se o valor digitado é inteiro, maior que zero e menor que 1000
    # se não for um valor válido, exibe uma mensagem de erro e limpa o entry
    entry_pop.bind('<Return>', lambda event: check_pop(entry_pop))

    # cria um label para entrada do tamanho dos indivíduos em bits
    label_ind = tk.Label(frame_w, text = 'COMPRIMENTO DOS INDIVÍDUOS (Nº DE BITS)', font = ('Roboto Mono Regular', 10), fg = 'green', bg = 'white', relief = 'solid', bd = 1)
    label_ind.grid(row = 2, column = 0, padx = 10, pady = 10, sticky = 'w')

    # cria um entry para entrada do tamanho dos indivíduos em bits
    entry_ind = tk.Entry(frame_w, width = 10, font = ('Arial', 10), relief = 'solid', bd = 2)
    entry_ind.grid(row = 2, column = 1, padx = 10, pady = 10, sticky = 'w')

    # quando o usuário pressiona a tecla Enter, checa se o valor digitado é inteiro, maior que zero e menor que 1000, e divisível por 3
    # se não for um valor válido, exibe uma mensagem de erro e limpa o entry
    entry_ind.bind('<Return>', lambda event: check_ind(entry_ind))

    # cria um label para entrada da probabilidade de crossover
    label_pc = tk.Label(frame_w, text = 'TAXA DE CROSSOVER/CRUZAMENTO', font = ('Roboto Mono Regular', 10), fg = 'green', bg = 'white', relief = 'solid', bd = 1)
    label_pc.grid(row = 3, column = 0, padx = 10, pady = 10, sticky = 'w')

    # cria um entry para entrada da probabilidade de crossover
    entry_pc = tk.Entry(frame_w, width = 10, font = ('Arial', 10), relief = 'solid', bd = 2)
    entry_pc.grid(row = 3, column = 1, padx = 10, pady = 10, sticky = 'w')

    # quando o usuário pressiona a tecla Enter, checa se o valor digitado é um número real entre 0 e 1
    # se não for um valor válido, exibe uma mensagem de erro e limpa o entry
    entry_pc.bind('<Return>', lambda event: check_pc(entry_pc))

    # cria um label para entrada da probabilidade de mutação
    label_pm = tk.Label(frame_w, text = 'TAXA DE MUTAÇÃO', font = ('Roboto Mono Regular', 10), fg = 'green', bg = 'white', relief = 'solid', bd = 1)
    label_pm.grid(row = 4, column = 0, padx = 10, pady = 10, sticky = 'w')

    # cria um entry para entrada da probabilidade de mutação
    entry_pm = tk.Entry(frame_w, width = 10, font = ('Arial', 10), relief = 'solid', bd = 2)
    entry_pm.grid(row = 4, column = 1, padx = 10, pady = 10, sticky = 'w')

    # quando o usuário pressiona a tecla Enter, checa se o valor digitado é um número real entre 0 e 1
    # se não for um valor válido, exibe uma mensagem de erro e limpa o entry
    entry_pm.bind('<Return>', lambda event: check_pm(entry_pm))

    # cria uma combobox para o tipo de seleção
    label_sel = tk.Label(frame_w, text = 'MÉTODO DE SELEÇÃO DOS PAIS', font = ('Roboto Mono Regular', 10), fg = 'green', bg = 'white', relief = 'solid', bd = 1)
    label_sel.grid(row = 5, column = 0, padx = 10, pady = 10, sticky = 'w')
    combo_sel = ttk.Combobox(frame_w, width = 10, font = ('Roboto Mono Regular', 10))
    combo_sel['values'] = ('Roleta', 'Torneio')
    combo_sel['state'] = 'readonly'
    combo_sel.current(0)
    combo_sel.grid(row = 5, column = 1, padx = 10, pady = 10, sticky = 'w')

    # cria um label para a função objetivo f(x,y) = 10 + x*sen(4x) + 3*sen(2y) e o intervalo de x e y
    # x = [0,4] e y = [0,2]
    # posiciona abaixo do botão 'Iniciar'
    label_obj = tk.Label(frame_w, text="FUNÇÃO OBJETIVO: \n f(x,y) = 10 + x.sen(4x) + 3.sen(2y) \n x = [0,4], y = [0,2]", font=('Roboto Mono Regular', 12), fg = 'orange', bg = 'green', justify = "center")
    label_obj.grid(row = 7, column = 0, columnspan = 2, padx = 10, pady = 5)

    # cria um botão que ao ser clicado retorna os valores digitados nos entries e combobox para a função callback
    button = tk.Button(frame_w, text = 'Executar', font = ('Roboto Mono Regular', 10), command = lambda: callback(entry_gen, entry_pop, entry_ind, entry_pc, entry_pm, combo_sel), relief = 'solid', bd = 1)
    button.grid(row = 6, column = 0, columnspan = 2, padx = 10, pady = 10)

# define a função que checa entry_gen e exibe uma mensagem de erro e limpa a entrada se o valor digitado não for inteiro, maior que zero e menor que 1000
def check_gen(entry):
    
        # tenta converter o valor digitado para inteiro
        try:
            gen = int(entry.get())
            # se o valor digitado for inteiro, checa se é maior que zero e menor que 10^6
            if gen <= 0 or gen > 999999:
                tk.messagebox.showerror('Erro', 'O valor deve ser um inteiro maior que zero e menor que 1.000.000.')
                entry.delete(0, tk.END)
            else: 
                return gen
    
        # se o valor digitado não for inteiro, exibe uma mensagem de erro e limpa a entrada
        except:
            tk.messagebox.showerror('Erro', 'O valor deve ser um inteiro maior que zero e menor que 1.000.000.')
            entry.delete(0, tk.END)

# define a função que checa entry_pop e exibe uma mensagem de erro e limpa a entrada se o valor digitado não for inteiro, maior que zero e menor que 1000
def check_pop(entry):
        
            # tenta converter o valor digitado para inteiro
            try:
                pop = int(entry.get())
                # se o valor digitado for inteiro, checa se é maior que zero e menor que 1000
                if pop <= 0 or pop > 999999:
                    tk.messagebox.showerror('Erro', 'O valor deve ser um inteiro maior que zero e menor que 1.000.000.')
                    entry.delete(0, tk.END)
                else:
                    return pop
        
            # se o valor digitado não for inteiro, exibe uma mensagem de erro e limpa a entrada
            except:
                tk.messagebox.showerror('Erro', 'O valor deve ser um inteiro maior que zero e menor que 1.000.000.')
                entry.delete(0, tk.END)

# define a função que checa entry_ind e exibe uma mensagem de erro e limpa a entrada se o valor digitado não for inteiro, maior que zero, menor que 1000 e divisível por 3
def check_ind(entry):
    
        # tenta converter o valor digitado para inteiro
        try:
            ind = int(entry.get())
            # se o valor digitado for inteiro, checa se é maior que zero, menor que 10^6 e divisível por 2
            if ind <= 0 or ind > 999999 or ind % 2 != 0:
                tk.messagebox.showerror('Erro', 'O valor deve ser um inteiro maior que zero, menor que 1.000.000 e divisível por 2.')
                entry.delete(0, tk.END)
            else:
                return ind
    
        # se o valor digitado não for inteiro, exibe uma mensagem de erro e limpa a entrada
        except:
            tk.messagebox.showerror('Erro', 'O valor deve ser um inteiro maior que zero, menor que 1.000.000 e divisível por 2.')
            entry.delete(0, tk.END)

# define a função que checa entry_pc e exibe uma mensagem de erro e limpa a entrada se o valor digitado não for um número real entre 0 e 1
def check_pc(entry):

    # tenta converter o valor digitado para float
    try:
        pc = float(entry.get())
        # se o valor digitado for float, checa se é maior ou igual a zero e menor ou igual a 1
        if pc <= 0 or pc > 1:
            tk.messagebox.showerror('Erro', 'O valor deve ser um número real entre 0 e 1, use ponto para casas decimais.')
            entry.delete(0, tk.END)
        else:
            return pc

    # se o valor digitado não for float, exibe uma mensagem de erro e limpa a entrada
    except:
        tk.messagebox.showerror('Erro', 'O valor deve ser um número real entre 0 e 1, use ponto para casas decimais.')
        entry.delete(0, tk.END)

# define a função que checa entry_pm e exibe uma mensagem de erro e limpa a entrada se o valor digitado não for um número real entre 0 e 1
def check_pm(entry):
    
        # tenta converter o valor digitado para float
        try:
            pm = float(entry.get())
            # se o valor digitado for float, checa se é maior ou igual a zero e menor ou igual a 1
            if pm <= 0 or pm > 1:
                tk.messagebox.showerror('Erro', 'O valor deve ser um número real entre 0 e 1, use ponto para casas decimais.')
                entry.delete(0, tk.END)
            else:
                return pm
    
        # se o valor digitado não for float, exibe uma mensagem de erro e limpa a entrada
        except:
            tk.messagebox.showerror('Erro', 'O valor deve ser um número real entre 0 e 1, use ponto para casas decimais.')
            entry.delete(0, tk.END)
    
num_gen, pop_size, ind_size, prob_crossover, prob_mutation, selection, frame_2d, frame_3d = None, None, None, None, None, None, None, None
main(num_gen, pop_size, ind_size, prob_crossover, prob_mutation, selection, frame_2d, frame_3d)



