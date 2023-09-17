'''ALGORITMO GENÉTICO PARA MINIMIZAÇÃO DA FUNÇÃO DE RASTRIGIN
AUTOR: LUIZ TARRALO 
FUNÇÃO DE RASTRIGIN: f(x) = A*n + sum(x^2 - A*cos(2pi*x)) para x em [-5.12, 5.12], A = 10, n = 2
x = 0 é o mínimo global, f(x) = 0'''

import numpy as np
import random
import math
import tkinter as tk
import matplotlib.pyplot as plt
from tkinter import messagebox
from tkinter import ttk
from matplotlib import cm
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

def main(num_gen, pop_size, pc, pm, sel, elit, tour, cross, window):
    # parâmetros teste
    if num_gen == None: interface() # se não houver parâmetros, abre a interface
    if window == None: return # se a janela for fechada, o programa é encerrado
    # elementos da função de rastrigin
    x_interval = [-5.12, 5.12] # intervalo de x
    n = 2 # número de variáveis
    # cria a população inicial 
    pop = create_population(pop_size, x_interval[0], x_interval[1], n)
    # lista para armazenar os valores de fitness
    best_fit = np.zeros(num_gen, dtype=float) 
    avg_fit = np.zeros(num_gen, dtype=float)
    # lista para armazenar os mínimos de cada geração
    best_ind = np.zeros((num_gen, n),dtype=float)
    my_progress = ttk.Progressbar(window, orient='horizontal', length=200, mode='determinate')
    my_progress.grid(row=1, column=0, padx=10, pady=10, sticky='n')
    i = 0
    # inicia o loop de gerações
    while i < num_gen:
        # calcula o fitness de cada indivíduo
        fitness = calc_fitness(pop)
        # armazena o melhor indivíduo da geração se houver elitismo
        if elit == 'Não': 
            if i == 0: elite = pop[fitness.index(np.amax(fitness))]
            else: 
                if np.amax(fitness) > best_fit[-1]: elite = pop[fitness.index(np.amax(fitness))]
                else: elite = best_ind[-1]
        # armazena os valores de fitness 
        best_ind[i] = pop[fitness.index(np.amax(fitness))]
        best_fit[i] = np.amax(fitness)
        avg_fit[i] = np.mean(fitness)
        if best_fit[i] == 1: break # se o fitness for 1, o algoritmo é encerrado
        # atualiza o label com o melhor indivíduo da geração
        my_progress['value'] = (i+1)/num_gen*100
        window.update_idletasks()
        # seleciona os indivíduos para o crossover
        parents = selection(pop, fitness, sel, tour)
        # realiza o crossover
        children = crossover(parents, cross, pc)
        # realiza a mutação
        children = mutation(children, pm, x_interval[0], x_interval[1])
        if elit == 'Não': children.append(elite) # adiciona o melhor indivíduo da geração anterior
        # substitui os piores indivíduos da população pelos filhos
        pop = restart(pop, fitness, children)
        i += 1 # incrementa o contador de gerações
    # salva os melhores indivíduos em um arquivo .txt
    np.savetxt('best_points.txt', best_ind, fmt='%.4f')
    # plota os gráficos de fitness na interface
    graph = plot_fit(best_fit, avg_fit, i, window)
    # botão que ao ser pressionado mostra o plot da função de rastrigin e os melhores indivíduos de cada geração
    plot_button = tk.Button(graph, text="Plotar Função", command=lambda: plot_rastrigin(best_ind, x_interval, n))
    plot_button.grid(row=2, column=0, padx=10, pady=10, sticky='n')
    return 0
# função que cria a população inicial
def create_population(pop_size, x_min, x_max, n):
    return np.random.uniform(x_min, x_max, (pop_size, n))
# função que retorna o valor da função de rastrigin
def rastrigin(x):
    return 10*len(x) + sum([(xi**2 - 10*np.cos(2*math.pi*xi)) for xi in x])
# função que calcula o fitness de cada indivíduo
def calc_fitness(pop):
    return [1/(rastrigin(ind)+1) for ind in pop]
# função que seleciona os indivíduos para o crossover
def selection(pop, fitness, sel, tour):
    parents = [] # lista para armazenar os pais
    if sel == 'Roleta':
        total_fit = sum(fitness) # soma dos valores de fitness
        probability = [fitness[i]/total_fit for i in range(len(fitness))] # probabilidade de cada indivíduo ser selecionado
        acumulated = [sum(probability[:i+1]) for i in range(len(probability))] # probabilidade acumulada
        while len(parents) < 2:
            r = random.random() # número aleatório entre 0 e 1
            for i, j in enumerate(acumulated):
                if r <= j: parents.append(pop[i]); break # se o número aleatório for menor ou igual à probabilidade acumulada, o indivíduo é selecionado
    elif sel == 'Torneio':
        while len(parents) < 2:
            tour_fit = [] # lista para armazenar os valores de fitness dos indivíduos selecionados para o torneio
            tournament = [] # lista para armazenar os indivíduos selecionados para o torneio
            for _ in range(tour): 
                rand_index = random.randint(0, len(pop)-1) # índice aleatório
                tournament.append(pop[rand_index]) # adiciona o indivíduo selecionado para o torneio
                tour_fit.append(fitness[rand_index]) # adiciona o valor de fitness do indivíduo selecionado para o torneio
            parents.append(tournament[np.argmax(tour_fit)]) # seleciona o indivíduo com maior fitness
    return parents
# função que realiza o crossover
def crossover(parents, cross, pc):
    children = [] # lista para armazenar os filhos
    if random.random() <= pc: # se o número aleatório for menor ou igual à probabilidade de crossover, o crossover é realizado
        if cross == 'Radcliff':
            beta = random.random() # número aleatório entre 0 e 1
            children.append(beta*parents[0] + (1-beta)*parents[1]) # calcula o primeiro filho
            children.append(beta*parents[1] + (1-beta)*parents[0]) # calcula o segundo filho
        elif cross == 'Wright':
            fit_list = []
            wright1 = 0.5*(parents[0] + parents[1])
            fit_list.append(1/(rastrigin(wright1)+1)) # fitness do primeiro filho 
            wright2 = 1.5*parents[0] - 0.5*parents[1]
            fit_list.append(1/(rastrigin(wright2)+1)) # fitness do segundo filho
            wright3 = 1.5*parents[1] - 0.5*parents[0]
            fit_list.append(1/(rastrigin(wright3)+1)) # fitness do terceiro filho
            # seleciona os dois filhos com maior fitness 
            while len(children) < 2:
                children.append([wright1, wright2, wright3][fit_list.index(np.amax(fit_list))])
                fit_list.remove(np.amax(fit_list))
    else: children = parents
    return children
# função que realiza a mutação
def mutation(children, pm, x_min, x_max):
    for i in range(len(children)):
        if random.random() <= pm: # se o número aleatório for menor ou igual à probabilidade de mutação, a mutação é realizada
            children[i] = np.random.uniform(x_min, x_max, len(children[i])) # gera um novo indivíduo
    return children        
# função que substitui os piores indivíduos da população pelos filhos
def restart(pop, fitness, children):
    worst_index = np.array(fitness).argsort()[:len(children)] # índices dos piores indivíduos
    for i,j in zip(worst_index, children): pop[i] = j # substitui os piores indivíduos pelos filhos
    return pop
# função que plota os gráficos de fitness
def plot_fit(best_fit, avg_fit, num_gen, window):
    x = np.arange(0, num_gen, 1)
    xticks = np.arange(0, num_gen+num_gen/100, num_gen/10)
    yticks = np.arange(0,1.05,0.05)
    # cria um frame para os gráficos
    graph = tk.LabelFrame(window, text="Gráficos de Fitness")
    graph.grid(row=0, column=1, padx=10, pady=10)
    fig_graph = Figure(figsize=(5,5), dpi=100)
    ax = fig_graph.add_subplot(111)
    ax.set_xlabel('Gerações')
    ax.set_ylabel('Fitness')
    ax.set_xticks(xticks)
    ax.set_yticks(yticks)
    ax.plot(x, best_fit, label='Melhor Fitness')
    ax.plot(x, avg_fit, label='Fitness Médio')
    ax.legend(loc='best')
    ax.grid(linestyle = '--', linewidth = 0.5, which = 'both', axis = 'both', alpha = 0.5)
    canvas = FigureCanvasTkAgg(fig_graph, master=graph)
    canvas.draw()
    canvas.get_tk_widget().grid(row=0, column=0, padx=10, pady=10, sticky='nw')
    # cria um botão para salvar o gráfico
    save_button = tk.Button(graph, text="Salvar", command=lambda: fig_graph.savefig('fitness.png'))
    save_button.grid(row=1, column=0, padx=10, pady=10, sticky='n')
    return graph
# função que plota os melhores indivíduos de cada geração na função de rastrigin
def plot_rastrigin(best_ind, x_interval, n):
    x = np.arange(x_interval[0], x_interval[1], 0.1) # cria um vetor de valores de x e y 
    y = np.arange(x_interval[0], x_interval[1], 0.1) # cria um vetor de valores de x e y
    x, y = np.meshgrid(x, y) # cria uma malha
    z = 10*n + (x**2 - 10*np.cos(2*math.pi*x)) + (y**2 - 10*np.cos(2*math.pi*y)) # função de rastrigin
    fig = plt.figure() # cria uma figura
    manager = plt.get_current_fig_manager() # gerenciador da figura
    manager.set_window_title('Função de Rastrigin - Melhores Indivíduos') # título da janela
    ax = fig.add_subplot(111, projection='3d') # cria um subplot 3D
    surf = ax.plot_surface(x, y, z, cmap=cm.inferno, linewidth = 0.1, edgecolor = 'black', antialiased=True) # plota a função de rastrigin
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.set_title('Pontos de mínimo da função de Rastrigin') 
    for i in range(len(best_ind)):
        ax.scatter(best_ind[i][0], best_ind[i][1], rastrigin(best_ind[i]), marker='x', s=100) # plota o melhor indivíduo de cada geração
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5) # adiciona a barra de cores
    fig.tight_layout() # ajusta o layout da figura
    plt.show() # mostra a figura 
# função que cria a interface
def interface():
    def callback(num_gen, pop_size, pc, pm, sel, elit, tour, cross, window):
        # passa os parâmetros para a função principal
        main(int(num_gen), int(pop_size), float(pc), float(pm), sel, elit, int(tour), cross, window)
    window = tk.Tk()
    window.title("Algoritmo Genético para Minimização da Função de Rastrigin - Luiz Tarralo")
    window.geometry("900x650")
    # cria um frame para os parâmetros
    frame = tk.LabelFrame(window, text="PARÂMETROS DO AG")
    frame.grid(row=0, column=0, padx=10, pady=10, sticky='nw')
    # cria um label para cada parâmetro
    label_num_gen = tk.Label(frame, text="Número de Gerações")
    label_num_gen.grid(row=0, column=0, padx=10, pady=10)
    label_pop_size = tk.Label(frame, text="Tamanho da População")   
    label_pop_size.grid(row=1, column=0, padx=10, pady=10)
    label_pc = tk.Label(frame, text="Probabilidade de Crossover")
    label_pc.grid(row=2, column=0, padx=10, pady=10)
    label_pm = tk.Label(frame, text="Probabilidade de Mutação")
    label_pm.grid(row=3, column=0, padx=10, pady=10)
    label_sel = tk.Label(frame, text="Método de Seleção")
    label_sel.grid(row=4, column=0, padx=10, pady=10)
    label_elit = tk.Label(frame, text="Elitismo")
    label_elit.grid(row=5, column=0, padx=10, pady=10)
    label_tour = tk.Label(frame, text="Tamanho do Torneio")
    label_tour.grid(row=6, column=0, padx=10, pady=10)
    label_cross = tk.Label(frame, text="Método de Crossover")
    label_cross.grid(row=7, column=0, padx=10, pady=10)
    # cria um entry para cada parâmetro
    entry_num_gen = tk.Entry(frame)
    entry_num_gen.grid(row=0, column=1, padx=10, pady=10)
    entry_num_gen.insert(0, '500')
    entry_num_gen.bind('<FocusOut>', lambda event: check_gen(entry_num_gen))
    entry_pop_size = tk.Entry(frame)
    entry_pop_size.grid(row=1, column=1, padx=10, pady=10)
    entry_pop_size.insert(0, '100')
    entry_pop_size.bind('<FocusOut>', lambda event: check_pop(entry_pop_size))
    entry_pc = tk.Entry(frame)
    entry_pc.grid(row=2, column=1, padx=10, pady=10)
    entry_pc.insert(0, '0.8')
    entry_pc.bind('<FocusOut>', lambda event: check_pc(entry_pc))
    entry_pm = tk.Entry(frame)
    entry_pm.grid(row=3, column=1, padx=10, pady=10)
    entry_pm.insert(0, '0.05')
    entry_pm.bind('<FocusOut>', lambda event: check_pm(entry_pm))
    entry_sel = ttk.Combobox(frame, values=['Roleta', 'Torneio'])
    entry_sel.grid(row=4, column=1, padx=10, pady=10)
    entry_sel.current(0)
    entry_elit = ttk.Combobox(frame, values=['Sim', 'Não'])
    entry_elit.grid(row=5, column=1, padx=10, pady=10)
    entry_elit.current(0)
    entry_tour = tk.Entry(frame)
    entry_tour.grid(row=6, column=1, padx=10, pady=10)
    entry_tour.insert(0, '5')
    entry_tour.bind('<FocusOut>', lambda event: check_tour(entry_tour))
    entry_cross = ttk.Combobox(frame, values=['Radcliff', 'Wright'])
    entry_cross.grid(row=7, column=1, padx=10, pady=10)
    entry_cross.current(0)
    # cria um botão para executar o algoritmo
    button = tk.Button(frame, text="Executar", command=lambda: callback(entry_num_gen.get(), entry_pop_size.get(), entry_pc.get(), entry_pm.get(), entry_sel.get(), entry_elit.get(), entry_tour.get(), entry_cross.get(), window))
    button.grid(row=8, column=0, padx=10, pady=10)
    window.mainloop()
# funções que fazem as verificações de cada parâmetro
def check_gen(entry_num_gen):
    try:
        entry = int(entry_num_gen.get())
        if entry <= 0:
            messagebox.showerror('Erro', 'O número de gerações deve ser maior que 0.')
            entry.delete(0,'end')
            entry.insert(0, '500')
        else: return entry
    except:
        messagebox.showerror('Erro', 'O número de gerações deve ser um número inteiro.')
        entry.delete(0,'end')
        entry.insert(0, '500')
def check_pop(entry_pop_size):
    try:
        entry = int(entry_pop_size.get())
        if entry <= 0:
            messagebox.showerror('Erro', 'O tamanho da população deve ser maior que 0.')
            entry.delete(0,'end')
            entry.insert(0, '100')
        else: return entry
    except:
        messagebox.showerror('Erro', 'O tamanho da população deve ser um número inteiro.')
        entry.delete(0,'end')
        entry.insert(0, '100')
def check_pc(entry_pc):
    try:
        entry = float(entry_pc.get())
        if entry <= 0 or entry > 1:
            messagebox.showerror('Erro', 'A probabilidade de crossover deve estar entre 0 e 1.')
            entry.delete(0,'end')
            entry.insert(0, '0.8')
        else: return entry
    except:
        messagebox.showerror('Erro', 'A probabilidade de crossover deve ser um número real.')
        entry.delete(0,'end')
        entry.insert(0, '0.8')
def check_pm(entry_pm):
    try:
        entry = float(entry_pm.get())
        if entry <= 0 or entry > 1:
            messagebox.showerror('Erro', 'A probabilidade de mutação deve estar entre 0 e 1.')
            entry.delete(0,'end')
            entry.insert(0, '0.05')
        else: return entry
    except:
        messagebox.showerror('Erro', 'A probabilidade de mutação deve ser um número real.')
        entry.delete(0,'end')
        entry.insert(0, '0.05')
def check_tour(entry_tour):
    try:
        entry = int(entry_tour.get())
        if entry <= 0:
            messagebox.showerror('Erro', 'O tamanho do torneio deve ser maior que 0.')
            entry.delete(0,'end')
            entry.insert(0, '5')
        else: return entry
    except:
        messagebox.showerror('Erro', 'O tamanho do torneio deve ser um número inteiro.')
        entry.delete(0,'end')
        entry.insert(0, '5')
if __name__ == "__main__":
    main(num_gen = None, pop_size = None, pc = None, pm = None, sel = None, elit = None, tour = None, cross = None, window = None)