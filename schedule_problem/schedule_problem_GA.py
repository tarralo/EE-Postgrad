'''Scheduling of manutention tasks using a genetic algorithm
A power plant has 7 machines that need to be maintained in intervals of 3 months
Objective: find the best sequence of maintenance task in a year to reduce the power loss
Liquid power loss of system - Pl = Pt - Pp - Pd 
Pl must be maximized for each interval
Pt = total power installed
Pp = power loss due to maintenance
Pd = maximum power demand of the interval'''

import random
import pandas as pd
import numpy as np
import PIL.Image
import PIL.ImageTk
from tkinter import *
import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from tkinter import ttk
from tkinter import messagebox

PRIMARY_COLOR = "#3366CC"
SECONDARY_COLOR = "#DCDCDC" 

def main(num_gen, pop_size, tournament_size, pc, pm, sel, elt, frame_sch, frame_best, frame_fit):

    # se na primeira execução, chama a função que cria a interface gráfica
    if num_gen == None: interface()
    else: # só inicia o algoritmo genético quando o usuário clicar no botão "Iniciar"

        machines, demand = import_schedule() # recebe os dados da planilha
        num_machines = len(machines.index) # número de máquinas
        num_intervals = len(demand.index) # número de intervalos

        num_intervals_machine = [machines.iloc[i]['intervals'] for i in range(num_machines)] # cria uma lista para o número de intervalos de cada máquina
        total_power = machines['power'].sum() # potência total instalada
        max_demand = [demand.iloc[i]['max_demand'] for i in range(num_intervals)] # cria uma lista para demanda máxima de cada intervalo
        power_capacity = [machines.iloc[i]['power'] for i in range(num_machines)] # cria uma lista para a potência de cada máquina

        best_ind = [] # lista que armazena o melhor indivíduo
        best_power = [] # lista que armazena a potência líquida do melhor indivíduo (melhor solução)
        avg_power = [] # lista que armazena a potência líquida da população (média)
        best_fit = [] # lista que armazena o melhor fitness de cada geração
        avg_fitness = [] # lista que armazena o fitness médio de cada geração

        pop = init_pop(num_machines, num_intervals, pop_size, num_intervals_machine, total_power, max_demand, power_capacity)

        for i in range(num_gen):

            # chama a função que calcula a potência líquida
            power_liquid = calc_power_liquid(pop, power_capacity, num_intervals, max_demand, total_power)

            # chama a função que calcula o fitness
            fitness = calc_fitness(power_liquid, max_demand)

            # reserva as informações do melhor indivíduo atual
            best_ind.append(pop[np.argmax(fitness)])
            best_power.append(power_liquid[np.argmax(fitness)])
            best_fit.append(np.amax(fitness))            

            # reserva as informações sobre média da população
            avg_power.append(np.mean(power_liquid, axis=0))
            avg_fitness.append(np.mean(fitness))

            if elt == True: # checa se 'elitismo' está ativado
                elite = np.zeros((num_intervals, num_machines), dtype = int) # cria uma matriz de zeros para o elite
                if i == 0: # se for a primeira geração, elite recebe o melhor indivíduo atual
                    elite = best_ind[i].copy()
                else: # se não for a primeira geração, checa se o melhor indivíduo atual é melhor que o melhor indivíduo anterior
                    if best_fit[i] > best_fit[i-1]: elite = best_ind[i].copy()
                    else: elite = best_ind[i-1].copy()

            # chama a função que seleciona os pais de acordo com o método de seleção escolhido
            parents = select_parents(pop, fitness, tournament_size, sel)

            # chama a função que faz o crossover
            offsprings = crossover(parents, num_intervals, num_machines, pc)

            # chama a função que faz a mutação, dentro dela valida os descendentes conforme restrições
            offsprings = mutation(parents, offsprings, num_intervals, num_intervals_machine, num_machines, pm, total_power, max_demand, power_capacity)

            if elt == True: offsprings.append(elite) # adiciona o elite aos descendentes se 'elitismo' estiver ativado
            
            # substitui os indivíduos menos aptos da população pelos descendentes
            pop = replace_pop(pop, offsprings, fitness)      

        # chama a função que mostra os dados do último melhor indivíduo dentro do frame 'frame_sch'
        show_schedule(best_ind[-1], frame_sch, power_liquid)
        # chama a função que plota os gráficos com as melhores potências líquidas para cada geração
        plot_all(best_power, avg_power, frame_best, frame_fit, num_gen, best_fit, avg_fitness)

def import_schedule():

    machine = pd.read_excel('schedule.xlsx', sheet_name='machine', index_col='number') # recebe os dados da lista de máquinas
    demand = pd.read_excel('schedule.xlsx', sheet_name='intervals', index_col='number') # recebe os dados da lista de intervalos e demandas

    # NaN = 0
    machine = machine.fillna(0)
    demand = demand.fillna(0)

    return machine, demand

def init_pop(num_machines, num_intervals, pop_size, num_intervals_machine, total_power, max_demand, power_capacity):

    pop = [] # população

    for _ in range(pop_size): # corre a população

        check = 0 # checa se o indivíduo é válido para segunda restrição
        ind = np.zeros((num_intervals, num_machines), dtype = int) # cada indivíduo é uma matriz de zeros num_intervals x num_machines
        
        while check == 0:
            # chama a função que gera um indivíduo e checa as duas restrições
            check, ind = check_init(ind, num_intervals_machine, num_machines, num_intervals, total_power, max_demand, power_capacity) 

        pop.append(ind) # adiciona o indivíduo na população
    
    return pop

def check_init(ind, num_intervals_machine, num_machines, num_intervals, total_power, max_demand, power_capacity):
    
    aux = np.zeros(num_machines, dtype = int) # auxiliar para o número de intervalos da máquina

    for j in range(num_intervals): # corre as linhas da matriz
            for k in range(num_machines): # corre as colunas da matriz
                if j == 0: # se for a primeira linha 
                        ind[j][k] = random.randint(0,1) # o indivíduo recebe randomicamente 1 ou 0 para a máquina k no intervalo j
                        if ind[j][k] == 1: aux[k] += 1 # se o indivíduo for 1, incrementa o auxiliar da máquina k
                else: # se não for a primeira linha, precisa checar se aux[k] já atingiu o número de intervalos da máquina k
                    if num_intervals_machine[k] > 1: # se o número de intervalos da máquina k for maior que 1
                        if aux[k] < num_intervals_machine[k]: # se a máquina não atingiu seu intervalo, deve receber 1 
                            ind[j][k] = 1
                            aux[k] += 1 # incrementa o auxiliar da máquina k
                        else: ind[j][k] = 0 # se já atingiu, só pode receber 0
                    else: # se o número de intervalos da máquina k for igual a 1
                        if aux[k] == 0: # se a máquina ainda não recebeu 1, randomicamente recebe 1 ou 0
                            ind[j][k] = random.randint(0,1)
                            if ind[j][k] == 1: aux[k] += 1 # se o indivíduo for 1, incrementa o auxiliar da máquina k
                        else: ind[j][k] = 0 # se já recebeu 1, só pode receber 0

    # refaz a checagem do número de intervalos do indivíduo para que não haja máquinas com 0 intervalos
    for i in range(num_machines): # checa se a máquina i não recebeu nenhum intervalo     
        if aux[i] == 0: ind[random.randint(0, num_intervals-1)][i] = 1 # coloca 1 na máquina i em um intervalo aleatório

    power_loss = np.zeros(num_intervals, dtype = int) # matriz de perdas por intervalo para cada indivíduo
    power_liquid = np.zeros(num_intervals, dtype = int) # matriz de potência líquida por intervalo para cada indivíduo

    # calcula o power_loss do indivíduo
    for i in range(num_intervals): # corre os intervalos do indivíduo
        for j in range(num_machines): # corre as máquinas do indivíduo
            if ind[i][j] == 1: power_loss[i] += power_capacity[j] # se a máquina j for 1 no intervalo i soma a potência da máquina j na perda do intervalo i

    # calcula o power_liquid do indivíduo, percorrendo os intervalos
    for i in range(num_intervals): power_liquid[i] = total_power - power_loss[i] - max_demand[i] # potência líquida do intervalo i é a potência total - a perda do intervalo i - a demanda do intervalo i

    # checa se a potência líquida é negativa em qualquer intervalo
    if np.any(power_liquid <= 0): return 0, ind

    return 1, ind

def calc_power_liquid(pop, power_capacity, num_intervals, max_demand, total_power):

    power_loss = np.zeros((len(pop), num_intervals), dtype = int) # matriz de perdas por intervalo para cada indivíduo
    power_liquid = np.zeros((len(power_loss), num_intervals), dtype = int) # matriz de potência líquida por intervalo para cada indivíduo

    for i in range(len(pop)): # corre a população
        for j in range(num_intervals): # corre os intervalos
            for k in range(len(pop[i][j])): # corre as máquinas
                if pop[i][j][k] == 1: # se a máquina k for 1 no intervalo j
                    power_loss[i][j] += power_capacity[k] # soma a potência da máquina k na perda do intervalo j

    for i in range(len(power_loss)): # corre a população
        for j in range(num_intervals): # corre os intervalos
            power_liquid[i][j] = total_power - power_loss[i][j] - max_demand[j] # calcula a potência líquida do intervalo j para o indivíduo i

    return power_liquid   

def calc_fitness(power_liquid, max_demand):

    num_objectives = len(power_liquid[0]) # número de soluções objetivo
    scores = []
    weights = []

    # calcula o peso para o cáculo do fitness, sendo que o peso é definido de acordo com o max_demand do intervalo
    for i in range(num_objectives): weights.append((max_demand[i])/sum(max_demand)) # calcula o peso para cada intervalo 

    for ind in power_liquid:
        score = 0
        for i in range(len(weights)): # checa se a potência líquida é negativa ou igual a 0
            if any(pl <= 0 for pl in ind): 
                score = 0 # se for, recebe 0
                break
            score += weights[i]*ind[i] # se não, recebe o peso do intervalo multiplicado pela potência líquida
        scores.append(score)
    
    return scores

def select_parents(pop, fitness, tournament_size, sel):

    parents = [] # lista de pais

    if sel == 'Roleta': # se a seleção for por roleta

        total_fit = sum(fitness) # soma o fitness da população
        prob = [fitness[i]/total_fit for i in range(len(pop))] # calcula a probabilidade de cada indivíduo ser selecionado
        prob_acum = [sum(prob[:i+1]) for i in range(len(pop))] # calcula a probabilidade acumulada

        for _ in range(2): # seleciona 2 pais
            r = random.random() # gera um número aleatório entre 0 e 1
            for (i, individual) in enumerate(pop): # percorre a população
                if r <= prob_acum[i]: # se o número aleatório for menor ou igual a probabilidade acumulada do indivíduo
                    parents.append(individual) # adiciona o indivíduo na lista de pais
                    break
    
    if sel == 'Torneio': # se a seleção for por torneio
        
        pop_copy = pop.copy() # copia a população
        tournament_fitness = []

        for i in range(2): # seleciona 2 pais sem repetição

            # seleciona tournament_size indivíduos aleatoriamente
            tournament = random.sample(pop_copy, tournament_size)

            # pega o fitness de cada indivíduo do torneio
            for j in range(tournament_size): 
                # pega o indíce do indivíduo no torneio na população
                index = np.where(pop == tournament[j])[0][0]
                tournament_fitness.append(fitness[index]) # adiciona o fitness do indivíduo na lista de fitness do torneio

            # seleciona o melhor entre os indivíduos do torneio
            for k in range(tournament_size):
                if tournament_fitness[k] == np.amax(tournament_fitness): champ = tournament[k]

            if i == 1 and (champ == parents[0]).all(): i -= 1 # se o segundo pai for igual ao primeiro pai, repete o processo
            else: parents.append(champ) # se não for, adiciona o pai na lista de pais
                
    return parents

def crossover(parents, num_intervals, num_machines, pc):

    # cria uma lista vazia para armazenas os descendentes
    offsprings = []

    # se um número aleatório for menor que a probabilidade de crossover, faz o crossover
    if random.random() < pc:

        for i in range(len(parents)): # faz o crossover de acordo com o número de pais
            offspring = np.zeros((num_intervals, num_machines), dtype = int) # cria um descendente vazio
            for j in range(num_intervals): # percorre as linhas da matriz dos pais
                # seleciona um ponto de corte aleatório entre 0 e o número de máquinas
                cut = random.randint(0, num_machines)
                # reserva os valores do pai 1 até o ponto de corte
                offspring[j][:cut] = parents[i][j][:cut]
                # reserva os valores do pai 2 a partir do ponto de corte
                offspring[j][cut:] = parents[i][j][cut:]
            offsprings.append(offspring)

    else: offsprings = parents

    return offsprings

def mutation(parents, offsprings, num_intervals, num_intervals_machine, num_machines, pm, total_power, max_demand, power_capacity):

    offsprings_new = offsprings.copy() # copia os descendentes

    # se um número aleatório for menor que a probabilidade de mutação, faz a mutação
    if random.random() < pm:
        for i in range(len(offsprings_new)): # percorre os descendentes
            for j in range(num_intervals): # percorre as linhas da matriz do descendente
                for k in range(num_machines): # percorre as colunas da matriz do descendente
                    if offsprings_new[i][j][k] == 1: # se a coluna k da linha j do descendente for 1
                        if random.random() < 0.5: offsprings_new[i][j][k] = 0 # se um número aleatório for menor que 0.5, o indivíduo vira 0
                    else: # se o indivíduo for 0
                        if random.random() < 0.5: offsprings_new[i][j][k] = 1 # se um número aleatório for menor que 0.5, o indivíduo vira 1

    for i in range(len(offsprings_new)):
        check, offsprings_new[i] = check_offspring(offsprings_new[i], num_intervals_machine, num_machines, num_intervals, total_power, max_demand, power_capacity) # valida os descendentes
        if check == 1: continue # se o descendente for válido, continua 
        else: offsprings_new[i] = parents[i] # se não for, o descendente vira o pai
            
    return offsprings_new

def check_offspring(offspring, num_intervals_machine, num_machines, num_intervals, total_power, max_demand, power_capacity):

    aux = np.zeros(num_machines, dtype = int) # auxiliar para o número de intervalos da máquina

    for i in range(num_intervals): # corre os intervalos do indivíduo
        for j in range(num_machines): # corre as máquinas do indivíduo
            if i == 0: # se for o primeiro intervalo
                if offspring[i][j] == 1: aux[j] += 1 # se a máquina estiver ligada no intervalo, soma 1 no auxiliar
            else: 
                if offspring[i][j] == 1: aux[j] += 1
                if aux[j] < num_intervals_machine[j]: 
                    if offspring[i-1][j] == 1: 
                        offspring[i][j] = 1
                        aux[j] += 1
                        
    # refaz a checagem para garantir que não haja máquinas com 0 intervalo
    for i in range(num_machines): 
        if aux[i] == 0: offspring[random.randint(0, num_intervals-1)][i] = 1 # se a máquina tiver 0 intervalo, coloca 1 em um intervalo aleatório

    power_loss = np.zeros(num_intervals, dtype = int) # matriz de perdas por intervalo para cada indivíduo
    power_liquid = np.zeros(num_intervals, dtype = int) # matriz de potência líquida por intervalo para cada indivíduo

    # calcula o power_loss do indivíduo
    for i in range(num_intervals): # corre os intervalos do indivíduo
        for j in range(num_machines): # corre as máquinas do indivíduo
            if offspring[i][j] == 1: power_loss[i] += power_capacity[j] # se a máquina j for 1 no intervalo i soma a potência da máquina j na perda do intervalo i

    # calcula o power_liquid do indivíduo, percorrendo os intervalos
    for i in range(num_intervals): power_liquid[i] = total_power - power_loss[i] - max_demand[i] # potência líquida do intervalo i é a potência total - a perda do intervalo i - a demanda do intervalo i

    if np.any(power_liquid <= 0): return 0, offspring

    return 1, offspring

def replace_pop(pop, offsprings, fitness):

    sorted_indices = np.array(fitness).argsort().tolist()[::-1] # pega os índices dos indivíduos em ordem decrescente de fitness
    sorted_pop = [pop[i] for i in sorted_indices] # ordena a população de acordo com o fitness em ordem decrescente
    selected_pop = sorted_pop[len(offsprings):] # seleciona os piores indivíduos da população

    for i, offspring in enumerate(offsprings): # percorre os descendentes
        index_to_replace = np.where(pop == selected_pop[i])[0][0] # pega o índice do indivíduo selecionado na população
        pop[index_to_replace] = offspring # substitui o indivíduo selecionado pelo descendente

    return pop

def plot_all(best_power, avg_power, frame_best, frame_fit, num_gen, best_fit, avg_fitness):

    num_intervals = len(best_power[0]) # número de intervalos
    x = [i for i in range(num_gen)] # gerações, eixo x

    y_bp = [[] for _ in range(num_intervals)] # melhor potência
    y_ap = [[] for _ in range(num_intervals)] # potência média
    
    for i in range(num_gen):
        for j in range(num_intervals):
            y_bp[j].append(best_power[i][j]) # adiciona a melhor potência do intervalo j na geração i
            y_ap[j].append(avg_power[i][j]) # adiciona a potência média do intervalo j na geração i

    fig_power = Figure(figsize = (6, 6), dpi = 100) # cria a figura
    y_lim1_sup = np.amax(best_power) + 10
    y_lim1_inf = 0
    y_pass1 = float(f'{(y_lim1_sup - y_lim1_inf)/10:.1f}')
    xticks1 = np.arange(0, num_gen+1, (num_gen/10))
    yticks1 = np.arange(y_lim1_inf, y_lim1_sup+1, y_pass1)

    ax = fig_power.add_subplot(111) # cria o eixo
    ax.set_xlabel('GERAÇÕES') # define o rótulo do eixo x
    ax.set_ylabel('POTÊNCIA (MW)') # define o rótulo do eixo y
    ax.set_title('Gráfico da Potência Líquida para as soluções - AG') # define o título do gráfico
    ax.set_xlim([0, num_gen])
    ax.set_ylim([y_lim1_inf, y_lim1_sup])
    ax.set_xticks(xticks1)
    ax.set_yticks(yticks1)
    for k in range(num_intervals): # plota os gráficos
        ax.plot(x, y_bp[k], label = 'PL Máxima Intervalo ' + str(k + 1))
        ax.plot(x, y_ap[k], label = 'PL Média Intervalo ' + str(k + 1))
    
    # cria a legenda do gráfico com os rótulos
    pos_power = ax.get_position() # pega a posição do gráfico
    ax.set_position([pos_power.x0, pos_power.y0 + pos_power.height * 0.3, pos_power.width, pos_power.height*0.7]) # ajusta a posição do gráfico
    ax.legend(loc = 'upper center', bbox_to_anchor = (0.5, -0.2), fancybox = True, shadow = True, ncol = 2) 
    
    # adiciona uma grade no gráfico com espaçamento de 0.5
    ax.grid(linestyle = '--', linewidth = 1.0, which = 'major', axis = 'both', alpha = 0.5)

    # cria um botão para salvar o gráfico
    button_power = tk.Button(master = frame_best, text = 'Salvar Gráfico', command = lambda: fig_power.savefig('power.png'))
    button_power.pack(side = tk.TOP)

    # cria um canvas para o gráfico das potências
    canvas_power = FigureCanvasTkAgg(fig_power, master = frame_best)
    canvas_power.draw()
    canvas_power.get_tk_widget().pack(side = tk.TOP)

    fig_fit = Figure(figsize = (6, 6), dpi = 100) # cria a figura
    y_lim2_sup = np.amax(best_fit) + 1
    y_lim2_inf = np.amin(avg_fitness) - 1
    y_pass2 = float(f'{(y_lim2_sup - y_lim2_inf)/10:.2f}')
    xticks2 = np.arange(0, num_gen+1, (num_gen/10))
    yticks2 = np.arange(y_lim2_inf, y_lim2_sup+1, y_pass2)

    ax2 = fig_fit.add_subplot(111) # cria o eixo
    ax2.set_xlabel('GERAÇÕES') # define o rótulo do eixo x
    ax2.set_ylabel('FITNESS') # define o rótulo do eixo y
    ax2.set_title('Gráfico do Fitness para as soluções - AG \n Fitness definido pelo Método da Agregação') # define o título do gráfico
    ax2.set_xlim([0, num_gen])
    ax2.set_ylim([y_lim2_inf, y_lim2_sup])
    ax2.set_xticks(xticks2)
    ax2.set_yticks(yticks2)
    ax2.plot(x, best_fit, label = 'Melhor Fitness')
    ax2.plot(x, avg_fitness, label = 'Fitness Médio')

    # cria a legenda do gráfico com os rótulos
    pos_fit = ax2.get_position() # pega a posição do gráfico
    ax2.set_position([pos_fit.x0 + pos_fit.width * 0.1, pos_fit.y0 + pos_fit.height * 0.1, pos_fit.width*0.9, pos_fit.height*0.9]) # ajusta a posição do gráfico
    ax2.legend(loc = 'upper center', bbox_to_anchor = (0.5, -0.15), fancybox = True, shadow = True, ncol = 2) 

    # adiciona uma grade no gráfico com espaçamento de 0.5
    ax2.grid(linestyle = '--', linewidth = 0.5, which = 'major', axis = 'both', alpha = 1.0)

    # cria um botão para salvar o gráfico
    button_fit = tk.Button(master = frame_fit, text = 'Salvar Gráfico', command = lambda: fig_fit.savefig('fitness.png'))
    button_fit.pack(side = tk.TOP)

    # cria um canvas para o gráfico do fitness
    canvas_fit = FigureCanvasTkAgg(fig_fit, master = frame_fit)
    canvas_fit.draw()
    canvas_fit.get_tk_widget().pack(side = tk.BOTTOM)

def interface():

    def callback(gen, pop, tour, pc, pm, sel, elt): # define a função que será chamada quando o botão 'Executar' for pressionado

        # obtém os valores dos parâmetros dos widgets de entrada de dados
        num_gen = int(gen.get())
        pop_size = int(pop.get())
        tournament_size = int(tour.get())
        pc = float(pc.get())
        pm = float(pm.get())
        sel = sel.get()
        elt = elt.get()

        # cria uma janela para mostrar os resultados
        window_results = tk.Toplevel()
        window_results.title('Resultados obtidos pelo AG - Luiz Tarralo')
        window_results.geometry('1250x600')
        window_results.resizable(True, True)
        window_results.rowconfigure(0, weight = 1) # define que a linha 0 deve se expandir se necessário
        window_results.configure(bg = PRIMARY_COLOR)

        # cria um frame para mostrar o melhor indivíduo e seus dados no canto inferior da janela
        frame_sch = tk.Frame(window, width = 500, height = 300, bg = SECONDARY_COLOR, highlightbackground = 'black', highlightthickness = 1)
        frame_sch.grid(row=1, column=0, padx = 10, pady = 10, columnspan = 2, sticky = 'n')

        # cria um frame para um dos gráficos 2D no canto esquerdo da janela que preenche além de sua coluna se necessário
        frame_best = tk.Frame(window_results, width = 500, height = 500, bg = SECONDARY_COLOR, highlightbackground = 'black', highlightthickness = 1)
        frame_best.grid(row=0, column=0, padx = 10, pady = 10, rowspan = 2, sticky = 'w')
        frame_best.config(width = 500, height = 750)

        # cria um frame para um dos gráficos 2D no canto superior direito da janela
        frame_fit = tk.Frame(window_results, width = 500, height = 500, bg = SECONDARY_COLOR, highlightbackground = 'black', highlightthickness = 1)
        frame_fit.grid(row=0, column=1, padx = 10, pady = 10, rowspan = 2, sticky = 'e')
        frame_fit.config(width = 500, height = 750)

        # passa os dados e os frames para a função principal
        main(num_gen, pop_size, tournament_size, pc, pm, sel, elt, frame_sch, frame_best, frame_fit)

    window = tk.Tk() # cria a janela
    window.title('Algoritmo Genético - Programação de Manutenção de Sistemas Elétricos de Potência - Luiz Tarralo') # define o título da janela
    window.geometry('800x650') # define o tamanho da janela
    window.columnconfigure(1, weight = 1) # configura a coluna 1 para que o frame se expanda
    window.rowconfigure(1, weight = 1) # configura a linha 1 para que o frame se expanda
    window.configure(bg = PRIMARY_COLOR)

    # cria um frame para os widgets de entrada de dados no centro superior da janela
    frame_w = tk.Frame(window, width = 250, height = 250, bg = 'blue', highlightbackground= 'black', highlightthickness=2)
    frame_w.grid(row=0, column=0, padx = 10, pady = 10, sticky = 'nw')

    # aqui começa a parte do gif
    def gif():
        gif_image = "charlie.gif"
        open_image = PIL.Image.open(gif_image)
        frames = open_image.n_frames
        image_object = [PhotoImage(file=gif_image, format = f"gif -index {i}") for i in range(frames)]
        count = 0
        show_animation = None
        
        def animation(count):
            global show_animation
            new_image = image_object[count]

            gif_label.configure(image=new_image)
            count += 1
            if count == frames:
                count = 0
            show_animation = window.after(50, lambda:animation(count))
        
        gif_label = Label(window, image="")
        gif_label.grid(row=0, column=1, padx = 10, pady = 10, sticky = 'ne')

        animation(count)


    # aqui termina a parte do gif

    # chama função que cria os widgets de entrada de dados
    create_widgets(frame_w, callback)
    gif()

    # inicia o loop da janela
    window.mainloop()

def create_widgets(frame_w, callback):

    # cria um label para entrada de dados do número de gerações
    label_gen = tk.Label(frame_w, text = 'NÚMERO DE GERAÇÕES', font = ('Roboto Mono Regular', 10), fg = 'blue', bg = 'white', relief = 'solid', bd = 1)
    label_gen.grid(row = 0, column = 0, padx = 10, pady = 10, sticky = 'w')

    # cria um entry para entrada de dados do número de gerações, com valor padrão 100
    gen = tk.Entry(frame_w, width = 10, font = ('Roboto Mono Regular', 10))
    gen.grid(row = 0, column = 1, padx = 10, pady = 10, sticky = 'w')
    gen.insert(0, '100')

    # quando o usuário sair da entry, verifica se o valor é um inteiro maior que 0, se não for mostra uma mensagem de erro e limpa o entry
    gen.bind('<FocusOut>', lambda event: check_int(gen))

    # cria um label para entrada de dados do tamanho da população
    label_pop = tk.Label(frame_w, text = 'TAMANHO DA POPULAÇÃO', font = ('Roboto Mono Regular', 10), fg = 'blue', bg = 'white', relief = 'solid', bd = 1)
    label_pop.grid(row = 1, column = 0, padx = 10, pady = 10, sticky = 'w')

    # cria um entry para a entrada do tamanho da população, com valor padrão 100
    pop = tk.Entry(frame_w, width = 10, font = ('Roboto Mono Regular', 10))
    pop.grid(row = 1, column = 1, padx = 10, pady = 10, sticky = 'w')
    pop.insert(0, '100')

    # quando o usuário sair da entry, verifica se o valor é um inteiro maior que 0, se não for mostra uma mensagem de erro e limpa o entry
    pop.bind('<FocusOut>', lambda event: check_int(pop))

    # cria um label para entrada de dados do tamanho do torneio
    label_tour = tk.Label(frame_w, text = 'TAMANHO DO TORNEIO', font = ('Roboto Mono Regular', 10), fg = 'blue', bg = 'white', relief = 'solid', bd = 1)
    label_tour.grid(row = 2, column = 0, padx = 10, pady = 10, sticky = 'w')
    
    # cria um entry para a entrada do tamanho do torneio, com valor padrão 5
    tour = tk.Entry(frame_w, width = 10, font = ('Roboto Mono Regular', 10))
    tour.grid(row = 2, column = 1, padx = 10, pady = 10, sticky = 'w')
    tour.insert(0, '5')

    # quando o usuário sair da entry, verifica se o valor é um inteiro maior que 0 e menor que a população, se não for mostra uma mensagem de erro e limpa o entry
    tour.bind('<FocusOut>', lambda event: check_int_tour(tour, pop))

    # cria um label para entrada de dados da probabilidade de crossover
    label_pc = tk.Label(frame_w, text = 'PROBABILIDADE DE CROSSOVER', font = ('Roboto Mono Regular', 10), fg = 'blue', bg = 'white', relief = 'solid', bd = 1)
    label_pc.grid(row = 3, column = 0, padx = 10, pady = 10, sticky = 'w')

    # cria um entry para a entrada da probabilidade de crossover, com valor padrão 0.8
    pc = tk.Entry(frame_w, width = 10, font = ('Roboto Mono Regular', 10))
    pc.grid(row = 3, column = 1, padx = 10, pady = 10, sticky = 'w')
    pc.insert(0, '0.8')

    # quando o usuário sair da entry, verifica se o valor é um float entre 0 e 1, se não for mostra uma mensagem de erro e limpa o entry
    pc.bind('<FocusOut>', lambda event: check_float_pc(pc))

    # cria um label para entrada de dados da probabilidade de mutação
    label_pm = tk.Label(frame_w, text = 'PROBABILIDADE DE MUTAÇÃO', font = ('Roboto Mono Regular', 10), fg = 'blue', bg = 'white', relief = 'solid', bd = 1)
    label_pm.grid(row = 4, column = 0, padx = 10, pady = 10, sticky = 'w')

    # cria um entry para a entrada da probabilidade de mutação, com valor padrão 0.05
    pm = tk.Entry(frame_w, width = 10, font = ('Roboto Mono Regular', 10))
    pm.grid(row = 4, column = 1, padx = 10, pady = 10, sticky = 'w')
    pm.insert(0, '0.05')

    # quando o usuário sair da entry, verifica se o valor é um float entre 0 e 1, se não for mostra uma mensagem de erro e limpa o entry
    pm.bind('<FocusOut>', lambda event: check_float_pm(pm))

    # cria uma combobox para a escolha do tipo de seleção
    label_sel = tk.Label(frame_w, text = 'TIPO DE SELEÇÃO', font = ('Roboto Mono Regular', 10), fg = 'blue', bg = 'white', relief = 'solid', bd = 1)
    label_sel.grid(row = 5, column = 0, padx = 10, pady = 10, sticky = 'w')
    combo_sel = ttk.Combobox(frame_w, width = 10, font = ('Roboto Mono Regular', 10), values = ['Roleta', 'Torneio'])
    combo_sel.current(0)
    combo_sel.grid(row = 5, column = 1, padx = 10, pady = 10, sticky = 'w')

    # cria uma caixa de seleção para optar por elitismo, o valor padrão é marcado, se estiver marcado elt recebe True, se não recebe False
    label_elt = tk.Label(frame_w, text = 'ELITISMO', font = ('Roboto Mono Regular', 10), fg = 'blue', bg = 'white', relief = 'solid', bd = 1)
    label_elt.grid(row = 6, column = 0, padx = 10, pady = 10, sticky = 'w')
    elt = tk.BooleanVar()
    elt.set(True)
    check_elt = tk.Checkbutton(frame_w, variable = elt, onvalue = True, offvalue = False)
    check_elt.grid(row = 6, column = 1, padx = 10, pady = 10, sticky = 'w')

    # cria um botão 'Executar' que permanece desabilitado até que o usuário preencha todos os campos corretamente
    # ao ser clicado, retorna os valores das entradas e da caixa de seleção para a função callback 
    button = tk.Button(frame_w, text = 'Executar', font = ('Roboto Mono Regular', 10), fg = 'blue', bg = 'white', relief = 'solid', bd = 1, command = lambda: callback(gen, pop, tour, pc, pm, combo_sel, elt))
    button.grid(row = 7, column = 0, padx = 10, pady = 10, sticky = 'w')
    
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

def show_schedule(ind, frame_sch, power_liquid):

    # pega o número de linhas e colunas em ind
    rows = len(ind)
    cols = len(ind[0])

    # cria um label para a tabela
    label_table = tk.Label(frame_sch, text = 'AGENDA DE PRODUÇÃO DAS MÁQUINAS - PARADA PARA MANUTENÇÃO (On, Off)', font = ('Roboto Mono Regular', 10), fg = 'blue', bg = 'white', relief = 'solid', bd = 1)
    label_table.grid(row = 0, column = 0, padx = 10, pady = 10, sticky = 'n')

    # cria um label para a potência líquida em cada intervalo
    label_pl = tk.Label(frame_sch, text = f'Potência Líquida (MW) para os {rows} intervalos \n', font = ('Roboto Mono Regular', 10), fg = 'blue', bg = 'white', relief = 'solid', bd = 1)
    label_pl.grid(row = 1, column = 0, padx = 10, pady = 10, sticky = 'n')
    # mostra a potência líquida em cada intervalo do 'ind' junto a label 'label_pl'
    for i in range(0, rows, 10): label_pl['text'] += f'{power_liquid[i]} '
    
    # cria uma tabela com o número de linhas e colunas de ind que preencha todo o frame_sch
    table = ttk.Treeview(frame_sch, columns = [i for i in range(cols+1)], show = 'headings')
    table.grid(row = 4, column = 0, padx = 10, pady = 5, sticky = 'n')

    # cria as colunas da tabela
    for i in range(cols+1):
        if i == 0:
            table.heading(i, text = 'Intervalos ')
            table.column(i, minwidth = 0, width = 75, anchor = 'w')
        else:
            table.heading(i, text = f'Máquina {i}')
            table.column(i, minwidth = 0, width = 75, anchor = 'center')

    ind = [['Intervalo ' + str(i+1)]+[['On', 'Off'][ind[i][j]] for j in range(cols)] for i in range(rows)] # cria uma lista com os valores de ind
    for i in range(rows): table.insert('', 'end', values = ind[i]) # preenche a tabela com os valores de ind

    # ajusta o tamanho da tabela
    table['height'] = rows
        
main(num_gen = None, pop_size = None, tournament_size = None, pc = None, pm = None, sel = None, elt = None, frame_sch = None, frame_best = None, frame_fit = None)
        


