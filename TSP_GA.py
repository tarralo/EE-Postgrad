'''Travelling Salesman Problem with Genetic Algorithm
Author: Luiz Tarralo'''

import random
import math
import numpy as np
import pandas as pd
import tkinter as tk
from pyproj import Transformer 
from shapely.geometry import Point
from tkinter import *
from tkinter import ttk
from tkinter import messagebox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

PRIMARY_COLOR = "yellow"
SECONDARY_COLOR = "black"
input_crs = 'epsg:4326' # CRS WGS 84 (latitude, longitude)
output_crs = 'epsg:3857' # CRS Mercator (coordenadas cartesianas)

def main(num_generations, population_size, pm, pc, tournament_size, sel, elt, points, frame_map):

    if num_generations == None: interface()
    # cria um dicionário com os pontos e seus respectivos índices
    if points != None: points_dict = {i: points[i] for i in range(0, len(points))}
    else: return 0
    # cria listas para armazenar os dados a serem plotados
    best_ind = []
    best_dist = []
    best_fit = []
    avg_dist = []
    avg_fit = []
    # cria a população inicial
    pop = init_pop(population_size, points)
    for i in range(num_generations):
        # chama a função que calcula a distância entre os pontos
        distances = calc_distances(pop, points_dict)
        # chama a função que calcula o fitness
        fitness = calc_fitness(distances)
        # reserva as informações do melhor indivíduo da geração
        best_ind.append(pop[np.argmax(fitness)]), best_dist.append(distances[np.argmax(fitness)]), best_fit.append(np.amax(fitness))
        # reserva as informações sobre média de distância e fitness da geração
        avg_dist.append(np.mean(distances)), avg_fit.append(np.mean(fitness))
        if elt == True: # se elitismo for True, reserva o melhor indivíduo 
            elite = []
            if i == 0: elite = best_ind[i].copy()
            else: 
                if best_fit[i] > best_fit[i-1]: elite = best_ind[i].copy()
                else: elite = best_ind[i-1].copy()
        # chama a função que seleciona os pais para o crossover de acordo com o método escolhido
        if sel == 'Roleta': parents = roulette(pop, fitness)
        else: parents = tournament(pop, fitness, tournament_size)
        # chama a função que realiza o crossover PMX
        offspring = crossover_pmx(parents, pc)
        # chama a função que realiza a mutação
        offspring = mutation(offspring, pm)
        if elt == True: offspring.append(elite) # adiciona o elite à descendência
        # chama a função que substitui os indivíduos menos aptos da população pelos descendentes
        pop = replace(pop, offspring, fitness)
    # chama a função que plota o mapa de pontos
    show_map(points_dict, best_ind, frame_map, None)
    # cria um botão abaixo do mapa de pontos que ao ser clicado chama a função que plota os gráficos
    button_map = tk.Button(frame_map, text = 'Mostrar gráficos', command = lambda: plot(best_dist, best_fit, avg_dist, avg_fit, num_generations))
    button_map.grid(row = 1, column = 0, padx=10, pady = 10, sticky = 'se')

def create_points(num_points):
    points = []
    while len(points) < num_points:
        new_point = (random.random(), random.random())
        if new_point not in points: points.append(new_point)
    return points

def init_pop(population_size, points):
    return [random.sample(range(len(points)), len(points)) for _ in range(population_size)]

def calc_distances(pop, points_dict):
    return [sum([math.sqrt((points_dict[ind[i+1]][0] - points_dict[ind[i]][0])**2 + (points_dict[ind[i+1]][1] - points_dict[ind[i]][1])**2) for i in range(len(ind) - 1)]) for ind in pop]

def calc_fitness(distances):
    fitness = distances.copy()
    if distances[0] > 100: 
        for i in range(len(distances)): fitness[i] = (1 / distances[i]) * 100
    else:
        for i in range(len(distances)): fitness[i] = 1 / distances[i]                  
    return fitness

def roulette(pop, fitness):
    total_fit = sum(fitness) # soma dos fitness
    prob = [fitness[i]/total_fit for i in range(len(pop))] # probabilidade de cada indivíduo ser escolhido
    cum_prob = [sum(prob[:i+1]) for i in range(len(pop))] # probabilidade acumulada
    return [pop[i] for r in [random.random() for _ in range(2)] for i, p in enumerate(cum_prob) if r <= p] # escolhe os pais de acordo com a probabilidade acumulada

def tournament(pop, fitness, tournament_size):
    parents = []
    while len(parents) < 2: # seleciona 2 pais
        tour_fit = []
        tour = random.sample(pop, tournament_size) # seleciona os indivíduos para o torneio
        for j in range(tournament_size): tour_fit.append(fitness[pop.index(tour[j])]) # adiciona o fitness do indivíduo ao vetor tour_fit
        parent = tour[np.argmax(tour_fit)] # seleciona o indivíduo com maior fitness
        parents.append(parent)
    return parents

def crossover_pmx(parents, pc):
    if random.random() < pc:
        a, b = sorted(random.sample(range(len(parents[0])),2))
        offspring1, offspring2 = parents[0][:], parents[1][:]
        offspring1[a:b], offspring2[a:b] = parents[1][a:b], parents[0][a:b]
        for i in range(len(offspring1)):
            if i < a or i >= b:
                while offspring1[i] in parents[1][a:b]: offspring1[i] = parents[0][a:b][parents[1][a:b].index(offspring1[i])] 
                while offspring2[i] in parents[0][a:b]: offspring2[i] = parents[1][a:b][parents[0][a:b].index(offspring2[i])]
        return [offspring1, offspring2]
    else: return parents

def mutation(offspring, pm):
    if random.random() < pm:
        offspring_new = offspring.copy()
        for i in range(len(offspring_new)):
            a, b = sorted(random.sample(range(len(offspring_new[i])),2))
            offspring_new[i][a], offspring_new[i][b] = offspring_new[i][b], offspring_new[i][a]
        return offspring_new
    else: return offspring

def replace(pop, offspring, fitness):
    worst_indices = np.array(fitness).argsort()[:len(offspring)] # índices dos indivíduos menos aptos
    for i, j in zip(worst_indices, offspring): pop[i] = j # substitui os indivíduos menos aptos pelos descendentes
    return pop

def plot(best_dist, best_fit, avg_dist, avg_fit, num_generations):
    # cria uma janela para mostrar os resultados 
    window_results = tk.Toplevel()
    window_results.title('Resultados obtidos pelo AG - Luiz Tarralo')
    window_results.geometry('1250x600')
    window_results.columnconfigure(0, weight = 1)
    window_results.configure(bg = 'white')
    # cria um frame para o gráfico da distância
    frame_dist = tk.Frame(window_results, width = 500, height = 500, bg = 'white', highlightbackground = SECONDARY_COLOR, highlightthickness = 1)
    frame_dist.grid(row=0,column=0,padx=10,pady=10, rowspan = 2, sticky='nw')
    frame_dist.config(width = 500, height = 750)
    # cria um frame para o gráfico do fitness
    frame_fit = tk.Frame(window_results, width = 500, height = 500, bg = 'white', highlightbackground = SECONDARY_COLOR, highlightthickness = 1)
    frame_fit.grid(row=0,column=1,padx=10,pady=10, rowspan = 2, sticky='ne')
    frame_fit.config(width = 500, height = 750)
    # define eixo x para os gráficos
    x = [i for i in range(num_generations)]
    xticks = np.arange(0, num_generations + 1, (num_generations/10))
    # define a figura que armazena o plot da distância
    fig_dist = Figure(figsize=(6,5),dpi=100)
    y_lim1_inf = np.amin(avg_dist) - 0.1 * np.amax(avg_dist)
    y_lim1_sup = np.amax(avg_dist) + 0.1 * np.amax(avg_dist)
    y_pass1 = float(f'{(y_lim1_sup - y_lim1_inf)/10:.3f}')
    y_ticks1 = np.arange(y_lim1_inf, y_lim1_sup + y_pass1, y_pass1)
    # cria o plot da distância
    ax1 = fig_dist.add_subplot(111)
    ax1.plot(x, best_dist, label='Melhor distância')
    ax1.plot(x, avg_dist, label='Média de distância')
    ax1.set_title('Distância')
    ax1.set_xlabel('Geração')
    ax1.set_ylabel('Distância')
    ax1.set_xticks(xticks)
    ax1.set_yticks(y_ticks1)
    pos_dist = ax1.get_position()
    ax1.set_position([pos_dist.x0 + pos_dist.width*0.1, pos_dist.y0 + pos_dist.height*0.3, pos_dist.width*0.9, pos_dist.height*0.7])
    ax1.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), fancybox = True, shadow = True, ncol=2)
    ax1.grid(linestyle='--', linewidth=0.5, which = 'major', axis = 'both', alpha = 0.5)
    # cria um label com a melhor distância e a média de distância
    label_dist = tk.Label(master = frame_dist, text = f'MELHOR DISTÂNCIA: {best_dist[-1]:.3f}\nMÉDIA DISTÂNCIA: {avg_dist[-1]:.3f}', bg = 'orange', fg = 'white')
    label_dist.pack(side = tk.BOTTOM)
    # cria o botão para salvar o gráfico
    button_dist = tk.Button(master = frame_dist, text = 'Salvar Gráfico', command = lambda: fig_dist.savefig('graf_dist.png', bbox_inches='tight'))
    button_dist.pack(side = tk.TOP)
    # cria o canvas para o plot da distância
    canvas_dist = FigureCanvasTkAgg(fig_dist, master = frame_dist)
    canvas_dist.draw()
    canvas_dist.get_tk_widget().pack(side = tk.BOTTOM)
    # define a figura que armazena o plot do fitness
    fig_fit = Figure(figsize=(6,5),dpi=100)
    y_lim2_inf = np.amin(avg_fit) - 0.1 * np.amax(best_fit)
    y_lim2_sup = np.amax(avg_fit) + 0.1 * np.amax(best_fit)
    y_pass2 = float(f'{(y_lim2_sup - y_lim2_inf) / 10:.3f}')
    y_ticks2 = np.arange(y_lim2_inf, y_lim2_sup + y_pass2, y_pass2)
    # cria o plot do fitness
    ax2 = fig_fit.add_subplot(111)
    ax2.plot(x, best_fit, label='Melhor fitness')
    ax2.plot(x, avg_fit, label='Média de fitness')
    ax2.set_title('Fitness')
    ax2.set_xlabel('Geração')
    ax2.set_ylabel('Fitness')
    ax2.set_xticks(xticks)
    ax2.set_yticks(y_ticks2)
    pos_fit = ax2.get_position()
    ax2.set_position([pos_fit.x0 + pos_fit.width*0.1, pos_fit.y0 + pos_fit.height*0.3, pos_fit.width*0.9, pos_fit.height*0.7])
    ax2.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), fancybox = True, shadow = True, ncol=2)
    ax2.grid(linestyle='--', linewidth=0.5, which = 'major', axis = 'both', alpha = 0.5)
    # cria um label com o melhor fitness e a média de fitness
    label_fit = tk.Label(master = frame_fit, text = f'MELHOR FITNESS: {best_fit[-1]:.3f}\nMÉDIA FITNESS: {avg_fit[-1]:.3f}', bg = 'blue', fg = 'white')
    label_fit.pack(side = tk.BOTTOM)
    # cria o botão para salvar o gráfico
    button_fit = tk.Button(master = frame_fit, text = 'Salvar Gráfico', command = lambda: fig_fit.savefig('graf_fit.png', bbox_inches='tight'))
    button_fit.pack(side = tk.TOP)
    # cria o canvas para o plot do fitness
    canvas_fit = FigureCanvasTkAgg(fig_fit, master = frame_fit)
    canvas_fit.draw()
    canvas_fit.get_tk_widget().pack(side = tk.BOTTOM)

def interface():
    def callback(gen, pop, tour, pc, pm, sel, elt, file, num_points): # define a função que será chamada quando o botão 'Iniciar AG' for pressionado
        num_generations = int(gen.get())
        population_size = int(pop.get())
        pm = float(pm.get())
        pc = float(pc.get())
        tournament_size = int(tour.get())
        sel = sel.get()
        elt = elt.get()
        file = file.get()
        points = None
        if file == True: 
            points = read_input()
            num_points = len(points)
        else: num_points = int(num_points.get())
        # cria um frame para mostrar o mapa de pontos
        frame_map = tk.Frame(window, width = 250, height = 250, bg = 'white', highlightbackground = 'black', highlightthickness = 1)
        frame_map.grid(row=0,column=1,padx=10,pady=10,sticky='ne')
        # chama a função que mostra o mapa de pontos e retorna o dicionário de pontos
        best_ind = None
        points = show_map(points, best_ind, frame_map, num_points)
        # cria um botão em frame_map que ao ser pressionado chama a função main
        if points != None:
            button_ag = tk.Button(frame_map, text = 'Iniciar AG', command = lambda: main(num_generations, population_size, pm, pc, tournament_size, sel, elt, points, frame_map))
            button_ag.grid(row = 1, column = 0, padx = 10, pady = 10, sticky = 'sw')
    window = tk.Tk() # cria a janela principal
    window.title('AG - Problema do Caixeiro Viajante - Luiz Tarralo') # define o título da janela
    window.geometry('1000x600') # define o tamanho da janela
    window.columnconfigure(1, weight = 1)
    window.rowconfigure(1, weight = 1)
    window.configure(bg = PRIMARY_COLOR)
    # cria um frame para os widgets de entrada de dados
    frame_input = tk.Frame(window, width = 250, height = 250, bg = 'white', highlightbackground = 'black', highlightthickness = 1)
    frame_input.grid(row=0,column=0,padx=10,pady=10,sticky='nw')
    # chama a função que cria os inputs 
    create_inputs(frame_input, callback)
    window.mainloop() # inicia o loop principal da janela

def create_inputs(frame_input, callback):
    # cria um label para entrada de dados do número de gerações
    label_gen = tk.Label(frame_input, text = 'NÚMERO DE GERAÇÕES', font = ('Roboto Mono Regular', 10), fg = 'red', bg = 'white', relief = 'solid', bd = 1)
    label_gen.grid(row = 0, column = 0, padx = 10, pady = 10, sticky = 'w')
    # cria um entry para entrada de dados do número de gerações, com valor padrão 100
    gen = tk.Entry(frame_input, width = 10, font = ('Roboto Mono Regular', 10))
    gen.grid(row = 0, column = 1, padx = 10, pady = 10, sticky = 'w')
    gen.insert(0, '100')
    # quando o usuário sair da entry, verifica se o valor é um inteiro maior que 0, se não for mostra uma mensagem de erro e limpa o entry
    gen.bind('<FocusOut>', lambda event: check_int(gen))
    # cria um label para entrada de dados do tamanho da população
    label_pop = tk.Label(frame_input, text = 'TAMANHO DA POPULAÇÃO', font = ('Roboto Mono Regular', 10), fg = 'red', bg = 'white', relief = 'solid', bd = 1)
    label_pop.grid(row = 1, column = 0, padx = 10, pady = 10, sticky = 'w')
    # cria um entry para a entrada do tamanho da população, com valor padrão 100
    pop = tk.Entry(frame_input, width = 10, font = ('Roboto Mono Regular', 10))
    pop.grid(row = 1, column = 1, padx = 10, pady = 10, sticky = 'w')
    pop.insert(0, '100')
    # quando o usuário sair da entry, verifica se o valor é um inteiro maior que 0, se não for mostra uma mensagem de erro e limpa o entry
    pop.bind('<FocusOut>', lambda event: check_int(pop))
    # cria um label para entrada de dados do tamanho do torneio
    label_tour = tk.Label(frame_input, text = 'TAMANHO DO TORNEIO', font = ('Roboto Mono Regular', 10), fg = 'red', bg = 'white', relief = 'solid', bd = 1)
    label_tour.grid(row = 2, column = 0, padx = 10, pady = 10, sticky = 'w')
    # cria um entry para a entrada do tamanho do torneio, com valor padrão 5
    tour = tk.Entry(frame_input, width = 10, font = ('Roboto Mono Regular', 10))
    tour.grid(row = 2, column = 1, padx = 10, pady = 10, sticky = 'w')
    tour.insert(0, '5')
    # quando o usuário sair da entry, verifica se o valor é um inteiro maior que 0 e menor que a população, se não for mostra uma mensagem de erro e limpa o entry
    tour.bind('<FocusOut>', lambda event: check_int_tour(tour, pop))
    # cria um label para entrada de dados da probabilidade de crossover
    label_pc = tk.Label(frame_input, text = 'PROBABILIDADE DE CROSSOVER', font = ('Roboto Mono Regular', 10), fg = 'red', bg = 'white', relief = 'solid', bd = 1)
    label_pc.grid(row = 3, column = 0, padx = 10, pady = 10, sticky = 'w')
    # cria um entry para a entrada da probabilidade de crossover, com valor padrão 0.8
    pc = tk.Entry(frame_input, width = 10, font = ('Roboto Mono Regular', 10))
    pc.grid(row = 3, column = 1, padx = 10, pady = 10, sticky = 'w')
    pc.insert(0, '0.8')
    # quando o usuário sair da entry, verifica se o valor é um float entre 0 e 1, se não for mostra uma mensagem de erro e limpa o entry
    pc.bind('<FocusOut>', lambda event: check_float_pc(pc))
    # cria um label para entrada de dados da probabilidade de mutação
    label_pm = tk.Label(frame_input, text = 'PROBABILIDADE DE MUTAÇÃO', font = ('Roboto Mono Regular', 10), fg = 'red', bg = 'white', relief = 'solid', bd = 1)
    label_pm.grid(row = 4, column = 0, padx = 10, pady = 10, sticky = 'w')
    # cria um entry para a entrada da probabilidade de mutação, com valor padrão 0.05
    pm = tk.Entry(frame_input, width = 10, font = ('Roboto Mono Regular', 10))
    pm.grid(row = 4, column = 1, padx = 10, pady = 10, sticky = 'w')
    pm.insert(0, '0.05')
    # quando o usuário sair da entry, verifica se o valor é um float entre 0 e 1, se não for mostra uma mensagem de erro e limpa o entry
    pm.bind('<FocusOut>', lambda event: check_float_pm(pm))
    # cria uma combobox para a escolha do tipo de seleção
    label_sel = tk.Label(frame_input, text = 'TIPO DE SELEÇÃO', font = ('Roboto Mono Regular', 10), fg = 'red', bg = 'white', relief = 'solid', bd = 1)
    label_sel.grid(row = 5, column = 0, padx = 10, pady = 10, sticky = 'w')
    combo_sel = ttk.Combobox(frame_input, width = 10, font = ('Roboto Mono Regular', 10), values = ['Roleta', 'Torneio'])
    combo_sel.current(0)
    combo_sel.grid(row = 5, column = 1, padx = 10, pady = 10, sticky = 'w')
    # cria uma caixa de seleção para optar por elitismo, o valor padrão é marcado, se estiver marcado elt recebe True, se não recebe False
    label_elt = tk.Label(frame_input, text = 'ELITISMO', font = ('Roboto Mono Regular', 10), fg = 'red', bg = 'white', relief = 'solid', bd = 1)
    label_elt.grid(row = 6, column = 0, padx = 10, pady = 10, sticky = 'w')
    elt = tk.BooleanVar()
    elt.set(True)
    check_elt = tk.Checkbutton(frame_input, variable = elt, onvalue = True, offvalue = False)
    check_elt.grid(row = 6, column = 1, padx = 10, pady = 10, sticky = 'w')
    # cria um entry para a entrada do número de pontos, com valor padrão 10
    label_points = tk.Label(frame_input, text = 'NÚMERO DE PONTOS', font = ('Roboto Mono Regular', 10), fg = 'red', bg = 'white', relief = 'solid', bd = 1)
    label_points.grid(row = 7, column = 0, padx = 10, pady = 10, sticky = 'w')
    num_points = tk.Entry(frame_input, width = 10, font = ('Roboto Mono Regular', 10))
    num_points.grid(row = 7, column = 1, padx = 10, pady = 10, sticky = 'w')
    num_points.insert(0, '10')
    # quando o usuário sair da entry, verifica se o valor é um inteiro maior que 0, se não for mostra uma mensagem de erro e limpa o entry
    num_points.bind('<FocusOut>', lambda event: check_points(num_points))
    # cria uma caixa de seleção para optar por uso de arquivo ou não, o valor padrão é marcado, se estiver marcado file recebe True, se não recebe False
    label_file = tk.Label(frame_input, text = 'USAR ARQUIVO', font = ('Roboto Mono Regular', 10), fg = 'red', bg = 'white', relief = 'solid', bd = 1)
    label_file.grid(row = 8, column = 0, padx = 10, pady = 10, sticky = 'w')
    file = tk.BooleanVar()
    file.set(True)
    check_file = tk.Checkbutton(frame_input, variable = file, onvalue = True, offvalue = False)
    check_file.grid(row = 8, column = 1, padx = 10, pady = 10, sticky = 'w')
    # cria um botão 'Executar' que permanece desabilitado até que o usuário preencha todos os campos corretamente
    # ao ser clicado, retorna os valores das entradas e da caixa de seleção para a função callback 
    button = tk.Button(frame_input, text = 'Cria Mapa', command = lambda: callback(gen, pop, tour, pc, pm, combo_sel, elt, file, num_points))
    button.grid(row = 9, column = 0, padx = 10, pady = 10, sticky = 'w')
    
def check_int(entry_int):
    try: # tenta converter o valor de entry para inteiro
        entryint_value = int(entry_int.get())
        # se for um inteiro, checa se é maior que 0
        if entryint_value <= 0:
            messagebox.showerror('Erro', 'O valor deve ser um inteiro maior que 0.')
            entry_int.delete(0, 'end')
            entry_int.insert(0, '100')
        else: return entry_int
    except:
        messagebox.showerror('Erro', 'O valor deve ser um inteiro maior que 0.')
        entry_int.delete(0, 'end')
        entry_int.insert(0, '100')  

def check_points(entry_int):
    try: # tenta converter o valor de entry para inteiro
        entryint_value = int(entry_int.get())
        # se for um inteiro, checa se é maior que 0
        if entryint_value <= 0:
            messagebox.showerror('Erro', 'O valor deve ser um inteiro maior que 0.')
            entry_int.delete(0, 'end')
            entry_int.insert(0, '10')
        else: return entry_int
    except:
        messagebox.showerror('Erro', 'O valor deve ser um inteiro maior que 0.')
        entry_int.delete(0, 'end')
        entry_int.insert(0, '10')    

def check_int_tour(entry_tour, pop):
    pop_entry = int(pop.get())
    try: 
        entrytour_value = int(entry_tour.get())
        if entrytour_value <= 0 or entrytour_value > pop_entry:
            messagebox.showerror('Erro', 'O valor deve ser um inteiro maior que 0 e menor que a população.')
            entry_tour.delete(0, 'end')
            entry_tour.insert(0, '5')
        else: return entry_tour
    except ValueError:
        messagebox.showerror('Erro', 'O valor deve ser um inteiro maior que 0 e menor que a população.')
        entry_tour.delete(0, 'end')
        entry_tour.insert(0, '5')

def check_float_pc(entry_pc):
    
        try: # tenta converter o valor de entry para float
            entrypc_value = float(entry_pc.get())
            # se for um float, checa se está entre 0 e 1
            if entrypc_value <= 0 or entrypc_value > 1:
                messagebox.showerror('Erro', 'O valor deve ser um float entre 0 e 1.')
                entry_pc.delete(0, 'end')
                entry_pc.insert(0, '0.8')
            else: return entry_pc
        except:
            messagebox.showerror('Erro', 'O valor deve ser um float entre 0 e 1.')
            entry_pc.delete(0, 'end')
            entry_pc.insert(0, '0.8')

def check_float_pm(entry_pm):
    try: # tenta converter o valor de entry para float
        entrypm_value = float(entry_pm.get())
        # se for um float, checa se está entre 0 e 1
        if entrypm_value <= 0 or entrypm_value > 1:
            messagebox.showerror('Erro', 'O valor deve ser um float entre 0 e 1.')
            entry_pm.delete(0, 'end')
            entry_pm.insert(0, '0.01')
        else: return entry_pm
    except:
        messagebox.showerror('Erro', 'O valor deve ser um float entre 0 e 1.')
        entry_pm.delete(0, 'end')
        entry_pm.insert(0, '0.01')

def read_input():
    # abre o arquivo de entrada xlsx
    data = pd.read_excel('coordenadas.xlsx', sheet_name='coordenadas',index_col='num', decimal=',')
    # preenche os valores nulos com 0
    data = data.fillna(0)
    transformer = Transformer.from_crs(input_crs, output_crs, always_xy=True)
    lat = data['latitude'].tolist()#[data.iloc[i]['latitude'] for i in data.index]
    long = data['longitude'].tolist()#[data.iloc[i]['longitude'] for i in data.index]
    #name = data.iloc[i]['nome']
    points = []
    # converte as coordenadas de lat/long para x/y
    for i in range(len(lat)):
        x, y = transformer.transform(lat[i], long[i])
        x = float(f'{(x/1000):.3f}')
        y = float(f'{(y/1000):.3f}')
        points.append([x, y])
    return points

def show_map(points, best_ind, frame_map, num_points):
    if points is None: points = create_points(num_points)
    num_points = len(points)
    x = []
    y = []
    for i in range(num_points):
        x.append(points[i][0])
        y.append(points[i][1])
    # plota o mapa de pontos no frame_map
    fig = Figure(figsize = (6, 5), dpi = 100)
    ax = fig.add_subplot(111)
    ax.scatter(x, y, color = 'red')
    ax.set_title('Mapa de Coordenadas no Plano Cartesiano (Projeção Mercator)')
    if abs(x[0]) > 100:
        ax.set_xlabel('X (km)')
        ax.set_ylabel('Y (km)')
        for i in range(num_points): ax.annotate(str(i), (x[i]+2, y[i]), color = 'black', fontsize = 8)
    else:
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        for i in range(num_points): ax.annotate(str(i), (x[i]+0.02, y[i]), color = 'black', fontsize = 8)
    # cria o canvas para plotar o mapa   
    canvas = FigureCanvasTkAgg(fig, master = frame_map)
    canvas.draw()
    canvas.get_tk_widget().grid(row = 0, column = 0, padx = 10, pady = 10)
    if best_ind is not None: # plota o melhor indivíduo no frame_map
        # incrementa em 1 o índice de cada ponto do melhor indivíduo para que o primeiro ponto não seja 0
        # cria um label com os pontos do melhor indivíduo e posiciona no frame_map
        #label_best_ind = tk.Label(frame_map, text = 'Melhor indivíduo: ' + str(best_ind[-1]))
        #label_best_ind.grid(row = 1, column = 0, padx = 10, pady = 10)
        # cria setas que interligam os pontos na ordem do melhor indivíduo
        for i in range(num_points):
            start = best_ind[-1][i]
            end = best_ind[-1][(i+1) % num_points]
            ax.annotate('', xy = (x[end], y[end]), xytext = (x[start], y[start]), arrowprops = dict(arrowstyle = '->', connectionstyle = 'arc3', color = 'blue'))
        start = best_ind[-1][-1]
        end = best_ind[-1][0]
        ax.annotate('', xy = (x[end], y[end]), xytext = (x[start], y[start]), arrowprops = dict(arrowstyle = '->', connectionstyle = 'arc3', color = 'blue'))
        canvas.draw()
    else: return points # retorna os pontos do mapa

main(num_generations = None, population_size = None, pm = None, pc = None, tournament_size = None, sel = None, elt = None, points = None, frame_map = None)
