'''TRABALHO 8 ALGORITMOS GENÉTICOS - PROBLEMAS COM RESTRIÇÕES
LUIZ TARRALO 
PROBLEMA 1 - MINIMIZAR A FUNÇÃO CUSTO PHI(X1,X2) = (X1 - 1)^2 + (X2 - 1)^2 + rp((MAX(0,X1 + X2 - 0.5))^2 + (X1 - X2 - 2)^2) ONDE rp É O COEFICIENTE DE PENALIZAÇÃO
PARA X1 E X2 NO INTERVALO [-3,5]
PROBLEMA 2 - MINIMIZAR A FUNÇÃO F(X) = 5 * (SUM(i=1,4) Xi) - 5 * (SUM(i=1,4) Xi^2) - SUM(i=5,13) Xi 
SUJEITO A G1(X) = 2*X1 + 2*X2 + X10 + X11 - 10 <= 0
G2(X) = 2*X1 + 2*X3 + X10 + X12 - 10 <= 0
G3(X) = 2*X2 + 2*X3 + X11 + X12 - 10 <= 0
G4(X) = -8*X1 + X10 <= 0
G5(X) = -8*X2 + X11 <= 0
G6(X) = -8*X3 + X12 <= 0
G7(X) = -2*X4 - X5 + X10 <= 0
G8(X) = -2*X6 - X7 + X11 <= 0
G9(X) = -2*X8 - X9 + X12 <= 0
ONDE PARA i = [1,9] E i = 13 Xi ESTÁ NO INTERVALO [0,1]
E PARA i = [10,12] Xi ESTÁ NO INTERVALO [0,100]
A FUNÇÃO CUSTO É IGUAL A F(X) + rp * SUM(i=1,9) Gi(X)^2 -> EXECUTAR 30 VEZES O AG E OBTER A MÉDIA E O DESVIO PADRÃO DOS MELHORES VALORES DE CADA EXECUÇÃO'''

# Importando bibliotecas
import numpy as np
import matplotlib.pyplot as plt
import random
import tkinter as tk
from tkinter import messagebox
from tkinter import ttk
from copy import deepcopy

PRIMARY_COLOR = 'ivory2'
SECONDARY_COLOR = 'ivory4'
TERTIARY_COLOR = 'black'

def main(num_gen, pop_size, pc, pm, sel, elit, tour, cross, problem, window):
    if num_gen == None: interface()
    if window == None: return
    iter = 0
    result_aux = np.zeros((30, 1), dtype = float) # vetor para armazenar o resultado médio de cada execução
    current_result = tk.Label(window, text = '', font = ('Arial', 12, 'bold'), bg = PRIMARY_COLOR, fg = TERTIARY_COLOR)
    current_result.grid(row = 2, column = 0, padx = 20, pady = 20, sticky = 'n')
    while iter < 30:
        my_progress = ttk.Progressbar(window, orient = tk.VERTICAL, length = 422, mode = 'determinate')
        my_progress.grid(row = 0, column = 1, pady = 20, sticky = 'n')
        if problem == 1: 
            x_interval = [-3, 5]; # intervalo de x1 e x2
            rp = 100 # coeficiente de penalização
            pop = np.random.uniform(x_interval[0], x_interval[1], (pop_size,2)) # gera população inicial 
            best_ind = np.zeros((num_gen, 2), dtype = float) # melhor indivíduo de cada geração
            best_fit = np.zeros((num_gen, 1), dtype = float) # melhor fitness de cada geração
            mean_fit = np.zeros((num_gen, 1), dtype = float) # fitness médio de cada geração
        if problem == 2:
            x_first_interval = [0,1]; x_second_interval = [0,100]
            rp = 1e-1 # coeficiente de penalização
            pop = generate_pop(pop_size, x_first_interval, x_second_interval) # gera população inicial
            best_ind = np.zeros((num_gen, 13), dtype = float) # melhor indivíduo de cada geração
            best_fit = np.zeros((num_gen, 1), dtype = float) # melhor fitness de cada geração
            mean_fit = np.zeros((num_gen, 1), dtype = float) # fitness médio de cada geração
        for i in range(num_gen):
            # Calcula o fitness da população
            result = np.zeros((pop_size, 1), dtype = float) # vetor para armazenar o resultado da função objetivo
            if problem == 1: 
                result = objective_1(pop, rp) # calcula o retorno da função objetivo
                fitness = [1 / (x + 1e-3) for x in result] # o fitness é o inverso da função objetivo, pois queremos minimizar
            if problem == 2: 
                for j in range(len(pop)): result[j] = objective_2(pop[j], rp) # calcula o retorno da função objetivo
                fitness = [1 / (x + 1e-3) for x in result] # o fitness é o inverso da função objetivo, pois queremos minimizar
            if elit == 1:
                if i == 0: # se for a primeira geração, o melhor indivíduo é o elite
                    elite = pop[fitness.index(np.amax(fitness))] 
                    best_ind[i] = elite # melhor indivíduo da geração
                    best_fit[i] = np.amax(fitness) # melhor fitness da geração
                else: # se não for a primeira geração, verifica se o elite é melhor que o melhor indivíduo da geração anterior
                    if np.amax(fitness) > best_fit[i-1]: 
                        elite = pop[fitness.index(np.amax(fitness))] # se for, o elite é o melhor indivíduo da geração atual
                        best_ind[i] = elite # melhor indivíduo da geração
                        best_fit[i] = np.amax(fitness) # melhor fitness da geração
                    else: 
                        elite = best_ind[i-1, :] # se não for, o elite é o melhor indivíduo da geração anterior
                        best_ind[i] = elite # melhor indivíduo da geração
                        best_fit[i] = best_fit[i-1] # melhor fitness da geração
            else: 
                best_ind[i] = pop[fitness.index(np.amax(fitness))] # melhor indivíduo da geração
                best_fit[i] = np.amax(fitness) # melhor fitness da geração
            mean_fit[i] = np.mean(fitness) # fitness médio da geração
            my_progress['value'] += (1/num_gen)*100
            window.update_idletasks()
            # Seleciona os indivíduos para o crossover
            parents = selection(pop, fitness, sel, tour) # seleciona os indivíduos para o crossover
            # Realiza o crossover caso a probabilidade seja atendida
            if random.random() < pc: children = crossover(parents, cross, problem, rp) # realiza o crossover
            else: children = parents # se a probabilidade não for atendida, os pais são os filhos
            # Realiza a mutação caso a probabilidade seja atendida
            if random.random() < pm: 
                if problem == 1: children = mutation(children, x_interval, False) # realiza a mutação
                if problem == 2: children = mutation(children, x_first_interval, x_second_interval) # realiza a mutação
            if elit == 1: children.append(elite) # se elitismo for atendido, o elite é adicionado à população
            pop = replacement(pop, children, fitness) # reinicia os indivíduos que não atendem às restrições
        result_aux[iter] = np.amax(best_fit) # armazena o melhor fitness de cada execução
        current_result.config(text = f'{iter+1}: {float(result_aux[iter]):.4f}')
        window.update_idletasks()
        iter += 1
    current_result.destroy()
    # Calcula a média dos melhores fitness de cada execução
    mean = np.mean(result_aux)
    std_dev = np.std(result_aux)
    label = tk.LabelFrame(window, text = f'Média dos melhores fitness: {mean:.4f} ± {std_dev:.4f}', font = ('Arial', 12, 'bold'))
    label.grid(row = 3, column = 0, padx = 10, pady = 10)
    # Cria um botão para gerar o gráfico
    button = tk.Button(label, text = 'Gerar Gráfico', command = lambda: plot_graph(result_aux))
    button.grid(row = 4, column = 0, padx = 10, pady = 10) 
# Gera a população inicial para o problema 2
def generate_pop(pop_size, x_first_interval, x_second_interval):
    pop = np.zeros((pop_size, 13))
    for i in range(pop_size):
        pop[i,0:9] = np.random.uniform(x_first_interval[0], x_first_interval[1], 9)
        pop[i,9:12] = np.random.uniform(x_second_interval[0], x_second_interval[1], 3)
        pop[i,12] = np.random.uniform(x_first_interval[0], x_first_interval[1], 1)
    return pop
# Função objetivo do problema 1
def phi(x1, x2, rp):
    return (x1 - 1)**2 + (x2 - 1)**2 + rp * (max(0, x1 + x2 - 0.5)**2 + (x1 - x2 - 2)**2)
# Calcula a função objetivo para o problema 1
def objective_1(pop, rp):
    '''MINIMIZAR FUNÇÃO PHI(X1,X2) = (X1 - 1)^2 + (X2 - 1)^2 + rp((MAX(0,X1 + X2 - 0.5))^2 + (X1 - X2 - 2)^2)'''
    return np.array([phi(x1, x2, rp) for x1, x2 in pop])
# Função objetivo do problema 2
def f(x):
    return 5 * np.sum(x[0:4]) - 5 * np.sum(x[0:4]**2) - np.sum(x[4:13])
# Calcula a função objetivo para o problema 2
def objective_2(ind, rp):
    '''MINIMIZAR FUNÇÃO F(X) = 5 * (SUM(i=1,4) Xi) - 5 * (SUM(i=1,4) Xi^2) - SUM(i=5,13) Xi PARA 
    AS RESTRIÇÕES DEFINIDAS NO PROBLEMA'''
    g1 = 2 * ind[0] + 2 * ind[1] + ind[9] + ind[10] - 10 # restrição 1 g1 <= 0 
    g1 = max(0, g1) # se g1 > 0, g1 = 0
    g2 = 2 * ind[0] + 2 * ind[2] + ind[9] + ind[11] - 10 # restrição 2 g2 <= 0
    g2 = max(0, g2) # se g2 > 0, g2 = 0
    g3 = 2 * ind[1] + 2 * ind[2] + ind[10] + ind[11] - 10 # restrição 3 g3 <= 0
    g3 = max(0, g3) # se g3 > 0, g3 = 0
    g4 = -8 * ind[0] + ind[9] # restrição 4 g4 <= 0
    g4 = max(0, g4) # se g4 > 0, g4 = 0
    g5 = -8 * ind[1] + ind[10] # restrição 5 g5 <= 0
    g5 = max(0, g5) # se g5 > 0, g5 = 0
    g6 = -8 * ind[2] + ind[11] # restrição 6  g6 <= 0
    g6 = max(0, g6) # se g6 > 0, g6 = 0
    g7 = -2 * ind[3] - ind[4] + ind[9] # restrição 7 g7 <= 0
    g7 = max(0, g7) # se g7 > 0, g7 = 0
    g8 = -2 * ind[5] - ind[6] + ind[10] # restrição 8 g8 <= 0
    g8 = max(0, g8) # se g8 > 0, g8 = 0
    g9 = -2 * ind[7] - ind[8] + ind[11] # restrição 9 g9 <= 0
    g9 = max(0, g9) # se g9 > 0, g9 = 0
    g = np.array([g1, g2, g3, g4, g5, g6, g7, g8, g9]) # vetor de restrições
    g = np.sum(g**2, axis=0) # soma das restrições
    return f(ind) + rp * g 
# Função que seleciona os indivíduos para o crossover
def selection(pop, fitness, sel, tour):
    parents = []
    if sel == 1: # seleção por roleta
        total_fit = np.sum(fitness) # soma dos fitness
        prob_sel = [fitness[i] / total_fit for i in range(len(fitness))] # probabilidade de seleção
        acum_pro = [sum(prob_sel[:i+1]) for i in range(len(prob_sel))] # probabilidade acumulada
        while len(parents) < 2:
            r = random.random() # número aleatório
            for i,j in enumerate(acum_pro):
                if r <= j: parents.append(pop[i]); break # seleciona o indivíduo
    if sel == 2: # seleção por torneio
        while len(parents) < 2:
            tour_fit = [] # lista para armazenar os valores de fitness dos indivíduos do torneio
            tournament = [] # lista para armazenar os indivíduos do torneio
            for _ in range(tour):
                rand_index = random.randint(0, len(pop) - 1) # índice aleatório
                tournament.append(pop[rand_index]) # adiciona o indivíduo ao torneio
                tour_fit.append(fitness[rand_index]) # adiciona o fitness do indivíduo ao torneio
            parents.append(tournament[np.argmax(tour_fit)]) # seleciona o indivíduo com maior fitness
    return parents
# Função que realiza o crossover
def crossover(parents, cross, problem, rp):
    children = []
    if cross == 1: # crossover Radcliff
        beta = random.random() # número aleatório não nulo 
        children.append(beta*parents[0] + (1-beta)*parents[1]) # primeiro filho
        children.append(beta*parents[1] + (1-beta)*parents[0]) # segundo filho
    if cross == 2: # crossover Wright
        fit_list = []
        if problem == 1: # problema 1
            wright1 = 0.5 * (parents[0] + parents[1]) # primeiro filho
            fit_list.append(1/(1 + phi(wright1[0], wright1[1], rp))) # fitness do primeiro filho
            wright2 = 1.5 * parents[0] - 0.5 * parents[1] # segundo filho
            fit_list.append(1/(1 + phi(wright2[0], wright2[1], rp))) # fitness do segundo filho
            wright3 = 1.5 * parents[1] - 0.5 * parents[0] # terceiro filho
            fit_list.append(1/(1 + phi(wright3[0], wright3[1], rp))) # fitness do terceiro filho
            # Seleciona os dois filhos com maior fitness
            while len(children) < 2:
                children.append([wright1, wright2, wright3][fit_list.index(np.amax(fit_list))])
                fit_list.remove(np.amax(fit_list))
        if problem == 2: # problema 2
            wright1 = 0.5 * (parents[0] + parents[1]) # primeiro filho
            fit_list.append(1/(1 + objective_2(wright1, rp))) # fitness do primeiro filho
            wright2 = 1.5 * parents[0] - 0.5 * parents[1] # segundo filho
            fit_list.append(1/(1 + objective_2(wright2, rp))) # fitness do segundo filho
            wright3 = 1.5 * parents[1] - 0.5 * parents[0] # terceiro filho
            fit_list.append(1/(1 + objective_2(wright3, rp))) # fitness do terceiro filho
            # Seleciona os dois filhos com maior fitness
            while len(children) < 2:
                children.append([wright1, wright2, wright3][fit_list.index(np.amax(fit_list))])
                fit_list.remove(np.amax(fit_list))
    return children
# Função que realiza a mutação
def mutation(children, x_1, x_2):
    if not x_2: # problema 1
        for i in range(len(children)): children[i] = np.random.uniform(x_1[0],x_1[1],len(children[i])) # mutação uniforme 
    else: # problema 2
        for i in range(len(children)): 
            if i < 8 or i == 12: children[i] = np.random.uniform(x_1[0],x_1[1],len(children[i])) # mutação uniforme no primeiro intervalo
            else: children[i] = np.random.uniform(x_2[0],x_2[1],len(children[i])) # mutação uniforme no segundo intervalo
    return children
# Função que substitui os piores indivíduos da população pelos filhos	
def replacement(pop, children, fitness):
    worst_index = np.array(fitness).argsort()[:len(children)] # índices dos piores indivíduos
    new_pop = deepcopy(pop) # copia a população
    for i,j in zip(worst_index,children): new_pop[i] = j # substitui os piores indivíduos pelos filhos
    return new_pop
# Função que plota o gráfico
def plot_graph(result_aux):
    x = np.arange(0, 30, 1)
    yticks = np.arange(np.amin(result_aux) - np.amax(np.std(result_aux)), np.amax(result_aux) + np.amax(np.std(result_aux)), np.amax(result_aux)/10)
    plt.plot(x, result_aux, label = 'Melhor fitness por execução')
    # coloca pontos do desvio padrão de cada execução
    plt.plot(x, result_aux + np.std(result_aux), '--', color = 'red', label = 'Desvio padrão')
    plt.plot(x, result_aux - np.std(result_aux), '--', color = 'red')
    plt.yticks(yticks)
    plt.grid(linestyle = '--')
    plt.xlabel('Execução')
    plt.ylabel('Fitness')
    plt.title('Melhor fitness por execução')
    plt.legend(loc = 'best')
    manager = plt.get_current_fig_manager() # gerenciador da figura
    manager.set_window_title('Gráfico para as Execuções') # título da figura
    plt.show()
# Função que cria a interface gráfica
def interface():
    def callback(num_gen, pop_size, pc, pm, selection, elitism, tour, crossover, problema, window):
        if selection == 'Roleta': sel = 1
        elif selection == 'Torneio': sel = 2
        if elitism == 'Sim': elit = 1
        elif elitism == 'Não': elit = 2
        if crossover == 'Radcliff': cross = 1
        elif crossover == 'Wright': cross = 2
        if problema == 'Problema 1': problem = 1
        elif problema == 'Problema 2': problem = 2
        main(int(num_gen), int(pop_size), float(pc), float(pm), int(sel), int(elit), int(tour), int(cross), int(problem), window)
    window = tk.Tk() # cria a janela
    window.title('Algoritmo Genético para Minimização Restritiva') # título da janela
    window.geometry('450x600') # tamanho da janela
    window.configure(bg = PRIMARY_COLOR) # cor de fundo da janela
    frame = tk.LabelFrame(window, text = "PARÂMETROS", bg = PRIMARY_COLOR) # cria o frame
    frame.grid(row = 0, column = 0, padx = 10, pady = 10, sticky = 'nw') # posiciona o frame
    # cria os labels
    label_num_gen = tk.Label(frame, text = 'NÚMERO DE GERAÇÕES', fg = TERTIARY_COLOR, bg = SECONDARY_COLOR)
    label_pop_size = tk.Label(frame, text = 'TAMANHO POPULACIONAL', fg = TERTIARY_COLOR, bg = SECONDARY_COLOR)
    label_pc = tk.Label(frame, text = 'PROBABILIDADE DE CRUZAMENTO', fg = TERTIARY_COLOR, bg = SECONDARY_COLOR)
    label_pm = tk.Label(frame, text = 'PROBABILIDADE DE MUTAÇÃO', fg = TERTIARY_COLOR, bg = SECONDARY_COLOR)
    label_sel = tk.Label(frame, text = 'TIPO DE SELEÇÃO', fg = TERTIARY_COLOR, bg = SECONDARY_COLOR)
    label_elit = tk.Label(frame, text = 'ELITISMO', fg = TERTIARY_COLOR, bg = SECONDARY_COLOR)
    label_tour = tk.Label(frame, text = 'TAMANHO DO TORNEIO', fg = TERTIARY_COLOR, bg = SECONDARY_COLOR)
    label_cross = tk.Label(frame, text = 'TIPO DE CROSSOVER', fg = TERTIARY_COLOR, bg = SECONDARY_COLOR)
    label_problem = tk.Label(frame, text = 'PROBLEMA', fg = TERTIARY_COLOR, bg = SECONDARY_COLOR)
    # posiciona os labels
    label_num_gen.grid(row = 0, column = 0, padx = 10, pady = 10, sticky = 'w')
    label_pop_size.grid(row = 1, column = 0, padx = 10, pady = 10, sticky = 'w')
    label_pc.grid(row = 2, column = 0, padx = 10, pady = 10, sticky = 'w')
    label_pm.grid(row = 3, column = 0, padx = 10, pady = 10, sticky = 'w')
    label_sel.grid(row = 4, column = 0, padx = 10, pady = 10, sticky = 'w')
    label_elit.grid(row = 5, column = 0, padx = 10, pady = 10, sticky = 'w')
    label_tour.grid(row = 6, column = 0, padx = 10, pady = 10, sticky = 'w')
    label_cross.grid(row = 7, column = 0, padx = 10, pady = 10, sticky = 'w')
    label_problem.grid(row = 8, column = 0, padx = 10, pady = 10, sticky = 'w')
    # cria as entradas, posiciona, adiciona valores padrão e faz a validação
    entry_num_gen = tk.Entry(frame)
    entry_num_gen.grid(row = 0, column = 1, padx = 10, pady = 10, sticky = 'w')
    entry_num_gen.insert(0, '1000')
    entry_num_gen.bind('<FocusOut>', lambda event: check_int(entry_num_gen, 1000))
    entry_pop_size = tk.Entry(frame)
    entry_pop_size.grid(row = 1, column = 1, padx = 10, pady = 10, sticky = 'w')
    entry_pop_size.insert(0, '100')
    entry_pop_size.bind('<FocusOut>', lambda event: check_int(entry_pop_size, 100))
    entry_pc = tk.Entry(frame)
    entry_pc.grid(row = 2, column = 1, padx = 10, pady = 10, sticky = 'w')
    entry_pc.insert(0, '0.8')
    entry_pc.bind('<FocusOut>', lambda event: check_pc(entry_pc))
    entry_pm = tk.Entry(frame)
    entry_pm.grid(row = 3, column = 1, padx = 10, pady = 10, sticky = 'w')
    entry_pm.insert(0, '0.1')
    entry_pm.bind('<FocusOut>', lambda event: check_pm(entry_pm))
    entry_tour = tk.Entry(frame)
    entry_tour.grid(row = 6, column = 1, padx = 10, pady = 10, sticky = 'w')
    entry_tour.insert(0, '2')
    entry_tour.bind('<FocusOut>', lambda event: check_int(entry_tour, 2))
    # cria os comboboxes, posiciona, adiciona valores padrão e faz a validação
    combobox_sel = ttk.Combobox(frame, values = ['Roleta', 'Torneio'], state = 'readonly')
    combobox_sel.grid(row = 4, column = 1, padx = 10, pady = 10, sticky = 'w')
    combobox_sel.current(0)
    combobox_elit = ttk.Combobox(frame, values = ['Sim', 'Não'], state = 'readonly')
    combobox_elit.grid(row = 5, column = 1, padx = 10, pady = 10, sticky = 'w')
    combobox_elit.current(0)
    combobox_cross = ttk.Combobox(frame, values = ['Radcliff', 'Wright'], state = 'readonly')
    combobox_cross.grid(row = 7, column = 1, padx = 10, pady = 10, sticky = 'w')
    combobox_cross.current(0)
    combobox_problem = ttk.Combobox(frame, values = ['Problema 1', 'Problema 2'], state = 'readonly')
    combobox_problem.grid(row = 8, column = 1, padx = 10, pady = 10, sticky = 'w')
    combobox_problem.current(0)
    # cria o botão de iniciar
    button_start = tk.Button(frame, text = 'Iniciar AG', command = lambda: callback(entry_num_gen.get(), entry_pop_size.get(), entry_pc.get(), entry_pm.get(), combobox_sel.get(), combobox_elit.get(), entry_tour.get(), combobox_cross.get(), combobox_problem.get(), window))
    button_start.grid(row = 9, column = 0, columnspan = 2, padx = 10, pady = 10, sticky = 'w')
    window.mainloop()
# Funções de validação
def check_int(entry, default):
    try:
        entry = int(entry.get())
        if entry <= 0:
            messagebox.showerror('Erro', 'O valor deve ser maior que zero.')
            entry.delete(0, 'end')
            entry.insert(0, default)
        else: return entry
    except:
        messagebox.showerror('Erro', 'O valor deve ser um número inteiro.')
        entry.delete(0, 'end')
        entry.insert(0, default)
def check_pc(entry):
    try:
        entry = float(entry.get())
        if entry <= 0 or entry > 1:
            messagebox.showerror('Erro', 'O valor deve estar entre 0 e 1.')
            entry.delete(0, 'end')
            entry.insert(0, '0.8')
        else: return entry
    except:
        messagebox.showerror('Erro', 'O valor deve ser um número real.')
        entry.delete(0, 'end')
        entry.insert(0, '0.8')
def check_pm(entry):
    try:
        entry = float(entry.get())
        if entry <= 0 or entry > 1:
            messagebox.showerror('Erro', 'O valor deve estar entre 0 e 1.')
            entry.delete(0, 'end')
            entry.insert(0, '0.1')
        else: return entry
    except:
        messagebox.showerror('Erro', 'O valor deve ser um número real.')
        entry.delete(0, 'end')
        entry.insert(0, '0.1')

main(num_gen = None, pop_size = None, pc = None, pm = None, sel = None, elit = None, tour = None, cross = None, problem = None, window = None)
