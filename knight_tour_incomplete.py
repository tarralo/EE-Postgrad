'''Knight Tour Problem with Genetic Algorithm
Author: Luiz Tarralo 
Representação do cromossomo: lista dos movimentos do cavalo no tabuleiro
Função fitness: número de casas visitadas válidas pelo cavalo
Tamanho padrão do tabuleiro: 8x8 (64 casas)
Movimentos válidos do cavalo (x,y) compondo cada tupla
move 0: x+1, y+2
move 1: x+2, y+1
move 2: x+2, y-1
move 3: x+1, y-2
move 4: x-1, y-2
move 5: x-2, y-1
move 6: x-2, y+1
move 7: x-1, y+2'''

import numpy as np
import random
import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from copy import deepcopy

PRIMARY_COLOR = 'blue'
SECONDARY_COLOR = 'white'

def main(max_gen, pop_size, pm, pc, tournament_size, sel, elitism, board_size, frame_board):
    if max_gen == None: interface()
    # se não houver dados, sai do programa
    if pop_size == None: return
    gene_x = False
    best_ind = []; best_fit = []; avg_fit = []
    population = [pop_init(board_size) for _ in range(pop_size)]
    label_x = tk.Label(frame_board, text = "", font = ('Roboto Mono Regular', 12), bg = SECONDARY_COLOR) 
    label_x.grid(row = 0, column = 0, padx = 10, pady = 10, sticky = 's')
    for i in range(max_gen):
        # calcula o fitness da população
        fitness = [calc_fitness(ind) for ind in population]
        # reserva informações
        best_ind.append(population[np.argmax(fitness)])
        best_fit.append(np.amax(fitness))
        avg_fit.append(np.mean(fitness))
        curr_gen = i + 1
        if np.amax(fitness) == board_size**2: break
        label_x.config(text = f"Geração {i+1} Movimentos máximos {best_fit[i]}")
        frame_board.update_idletasks()
        # se houver elitismo 
        if elitism: 
            if i == 0: elite = population[np.argmax(fitness)]
            else:
                if best_fit[i] > best_fit[i-1]: elite = population[np.argmax(fitness)]
        # seleciona os pais para o crossover
        parents = select_parents(population,fitness,sel,tournament_size)
        # realiza o crossover
        offspring = crossover(parents,pc)
        # realiza a mutação
        if i == 0: offspring = mutation(offspring,pm,board_size, False)
        # chama a função de otimização repair_tour caso não estiver estagnado
        for j in range(len(offspring)): 
            if i > 0:
                if best_fit[i] != best_fit[i-1]: 
                    offspring[j] = repair_tour(offspring[j], board_size)
                    gene_x = False
                else: gene_x = True
        if gene_x: 
            worst_ind = np.array(fitness).argsort()[:tournament_size] # pega os piores indivíduos
            for k in range(len(worst_ind)):
                for _ in range(board_size): population[worst_ind[k]] = repair_tour(population[worst_ind[k]],board_size) 
            offspring_x = deepcopy(offspring)
            offspring = mutation(offspring,pm,board_size, gene_x)
            for k in range(len(offspring)):
                if calc_fitness(offspring[k]) < calc_fitness(offspring_x[k]): offspring[k] = offspring_x[k]
        # se houver elitismo, adiciona o elite ao offspring, se houver estagnação, tenta um novo movimento
        if elitism: 
            if gene_x: elite = last_try(elite, board_size)
            offspring.append(elite) 
        # chama a função que atualiza a população
        population = restart(population,offspring, fitness)
    # mostra o último melhor indivíduo no tabuleiro
    show_board(best_ind[-1], board_size, frame_board, curr_gen)
    # cria um botão abaixo do tabuleiro que ao ser clicado chama a função show_graph
    button_graph = tk.Button(frame_board, text = 'Mostrar gráficos', command = lambda: show_graph(best_fit, avg_fit, curr_gen))
    button_graph.grid(row = 1, column = 0, padx = 10, pady = 10)
# define a função que gera a população inicial
def pop_init(board_size):
    ind = [(0,0) for _ in range(board_size**2)]
    for i in range(1,board_size**2):
        move = get_possible_moves(ind[i-1][0],ind[i-1][1],board_size) # recebe movimentos dentro do tabuleiro
        ind[i] = move
    return ind
# define a função que gera os movimentos legais
def get_possible_moves(x,y,board_size):
    moves = []
    possible_moves = {0:(1,2),1:(2,1),2:(2,-1),3:(1,-2),4:(-1,-2),5:(-2,-1),6:(-2,1),7:(-1,2)}
    for i in range(len(possible_moves)):
        new_x = x + possible_moves[i][0]
        new_y = y + possible_moves[i][1]
        if new_x >= 0 and new_x < board_size and new_y >= 0 and new_y < board_size: moves.append((new_x,new_y))
    return random.choice(moves)
# define a função que calcula o fitness
def calc_fitness(individual):
    ind_fit = 1
    for i in range(1,len(individual)):
        check = 0
        if abs(individual[i][0] - individual[i-1][0]) == 1 and abs(individual[i][1] - individual[i-1][1]) == 2: check += 1
        elif abs(individual[i][0] - individual[i-1][0]) == 2 and abs(individual[i][1] - individual[i-1][1]) == 1: check += 1
        if individual[i] not in individual[:i] and check > 0: ind_fit += 1
        else: break
    return ind_fit
# define a função que seleciona os pais
def select_parents(population,fitness,sel,tournament_size):
    parents = [] # lista com os pais
    if sel == 'Torneio':
        while len(parents) < 2: # seleciona 2 pais
                tournament_fit = [] # lista com o fitness dos indivíduos do torneio
                tournament = random.sample(population, tournament_size) # seleciona aleatoriamente o número de indivíduos do torneio
                for i in range(tournament_size): tournament_fit.append(fitness[population.index(tournament[i])]) # calcula o fitness de cada indivíduo do torneio
                parents.append(tournament[np.argmax(tournament_fit)]) # seleciona o melhor indivíduo do torneio
    elif sel == 'Roleta':
        total_fitness = sum(fitness) # soma dos fitness
        probability = [fitness[i]/total_fitness for i in range(len(fitness))] # probabilidade de cada indivíduo ser escolhido
        acumulated = [sum(probability[:i+1]) for i in range(len(probability))] # probabilidade acumulada
        while len(parents) < 2:
            r = random.random() # número aleatório entre 0 e 1
            for i, p in enumerate(acumulated):
                if r <= p:
                    parents.append(population[i])
                    break
    return parents
# define a função que realiza o crossover
def crossover(parents, pc):
    if random.random() < pc:
        offspring = []
        # seleciona aleatoriamente o ponto de corte entre 1 e o fitness do pai 1
        cut1 = random.randint(1, calc_fitness(parents[0]))
        genes1 = parents[0][:cut1] + parents[1][cut1:]
        # primeiro filho recebe a primeira parte do pai 1 e o restante do pai 2
        offspring.append(genes1)
        # seleciona aleatoriamente o ponto de corte entre 1 e o fitness do pai 2
        cut2 = random.randint(1, calc_fitness(parents[1]))
        genes2 = parents[1][:cut2] + parents[0][cut2:]
        # segundo filho recebe a primeira parte do pai 2 e o restante do pai 1
        offspring.append(genes2)
        return offspring
    else: return parents
# define a função que realiza a mutação
def mutation(offspring, pm, board_size, gene_x):
    if random.random() < pm or gene_x:
        cut = random.randint(1, calc_fitness(offspring[0])) # seleciona aleatoriamente o ponto de corte entre 1 e o fitness do indivíduo
        offspring[0][cut] = get_possible_moves(offspring[0][cut-1][0], offspring[0][cut-1][1], board_size) # substitui o movimento no ponto de corte pelo novo movimento
        cut = random.randint(1, calc_fitness(offspring[1])) # seleciona aleatoriamente o ponto de corte entre 1 e o fitness do indivíduo
        offspring[1][cut] = get_possible_moves(offspring[1][cut-1][0], offspring[1][cut-1][1], board_size) # substitui o movimento no ponto de corte pelo novo movimento
        return offspring
    else: return offspring
# define a função que realiza a busca local
def repair_tour(individual, board_size):
    local_fit = calc_fitness(individual)
    ind_copy = deepcopy(individual)
    possible_moves = {0:(1,2),1:(2,1),2:(2,-1),3:(1,-2),4:(-1,-2),5:(-2,-1),6:(-2,1),7:(-1,2)}
    for i in range(1,len(ind_copy)):
        cand = []
        aux = []
        for j in range(len(possible_moves)):
            check = 0
            x = 0; y = 0
            x_curr = ind_copy[i-1][0]
            y_curr = ind_copy[i-1][1]
            x = x_curr + possible_moves[j][0]
            y = y_curr + possible_moves[j][1]
            if abs(x - ind_copy[i-1][0]) == 2 and abs(y - ind_copy[i-1][1]) == 2: check += 1
            if abs(x - ind_copy[i-1][0]) == 1 and abs(y - ind_copy[i-1][1]) == 1: check += 1
            if x >= 0 and x < board_size and y >= 0 and y < board_size and (x,y) not in ind_copy[:i] and check > 0: cand.append((x,y))
        if len(cand) == 0: break # se não houver movimentos possíveis, sai do loop 
        elif len(cand) == 1: ind_copy[i] = cand[0]
        else:
            for k in cand: 
                x_1 = x_curr + k[0]; y_1 = y_curr + k[1]
                can_jump = 0
                for l in range(len(possible_moves)): 
                    x_2 = x_1 + possible_moves[l][0]
                    y_2 = y_1 + possible_moves[l][1]
                    if x_2 >= 0 and x_2 < board_size and y_2 >= 0 and y_2 < board_size and (x_2,y_2) not in ind_copy[:i]: can_jump += 1
                aux.append(can_jump)
            # escolhe o candidato com o menor número de movimentos possíveis
            opt_move = cand[aux.index(np.amin(aux))]
            ind_copy[i] = opt_move
    delta_fit = calc_fitness(ind_copy) - local_fit
    if delta_fit > 0: return ind_copy
    else: return individual
# define a função de última tentativa
def last_try(individual, board_size):
    last_jump = calc_fitness(individual)
    ind_copy = deepcopy(individual)
    possible_moves = {0:(1,2),1:(2,1),2:(2,-1),3:(1,-2),4:(-1,-2),5:(-2,-1),6:(-2,1),7:(-1,2)}
    for moves in possible_moves:
        x = ind_copy[-1][0] + possible_moves[moves][0]
        y = ind_copy[-1][1] + possible_moves[moves][1]
        if x >= 0 and x < board_size and y >= 0 and y < board_size and (x,y) not in ind_copy:
            ind_copy[last_jump] = (x,y)
            new_tour = calc_fitness(ind_copy)
            if new_tour > last_jump: return ind_copy
    return individual
# define a função que atualiza a população
def restart(population, offspring, fitness):
    worst_index = np.array(fitness).argsort()[:len(offspring)] # índices dos piores indivíduos
    for i, j in zip(worst_index, offspring): population[i] = j # substitui os piores indivíduos pelos filhos
    return population
# define a função que mostra o tabuleiro com o melhor indivíduo
def show_board(knight_tour, board_size, frame_board, max_gen):
    fig = Figure(figsize=(5, 5), dpi=100)
    ax = fig.add_subplot(111)
    dx, dy = 0.016, 0.06
    P = np.arange(-5.0, 5.0, dx)
    Q = np.arange(-5.0, 5.0, dy)
    P, Q = np.meshgrid(P, Q)
    # define os limites do gráfico
    min_max = np.min(P), np.max(P), np.min(Q), np.max(Q)
    res = np.add.outer(range(board_size), range(board_size)) % 2
    if knight_tour != None:
        # printa o número do movimento em vermelho na coordenada x,y correspondente no tabuleiro
        for i in range(len(knight_tour)):
            if knight_tour[i] not in knight_tour[:i]: 
                x, y = knight_tour[i]
                ax.text(x, y, i+1, ha="center", va="center", color="red", size=10)
            else: break
    # printa o tabuleiro
    ax.imshow(res, cmap="binary_r")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(f"TABULEIRO DE XADREZ: {board_size} por {board_size}")
    canvas = FigureCanvasTkAgg(fig, master=frame_board)
    canvas.draw()
    canvas.get_tk_widget().grid(row = 0, column = 0, padx = 10, pady = 10, sticky = 'ne')
    if knight_tour != None: 
        label_final = tk.Label(frame_board, text = f"Gerações {max_gen} Movimentos máximos {calc_fitness(knight_tour)}", font = ('Roboto Mono Regular', 12), bg = SECONDARY_COLOR) 
        label_final.grid(row = 0, column = 0, padx = 10, pady = 10, sticky = 's')
# define a função que monta a interface gráfica
def interface():
    def callback(gen, pop, tour, pc, pm, sel, elt, board_size):
        max_gen = int(gen.get())
        pop_size = int(pop.get())
        pm = float(pm.get())
        pc = float(pc.get())
        tournament_size = int(tour.get())
        sel = sel.get()
        elt = elt.get()
        board_size = int(board_size.get())
        # cria um frame para mostrar o tabuleiro
        frame_board = tk.Frame(window, width = 500, height = 500, bg = SECONDARY_COLOR, highlightbackground = 'black', highlightthickness = 1)
        frame_board.grid(row = 0, column = 1, padx = 10, pady = 10, sticky = 'ne')
        show_board(None, board_size, frame_board, None)
        # cria um botão para iniciar o algoritmo genético
        button = tk.Button(frame_board, text = 'Iniciar AG', command = lambda: main(max_gen, pop_size, pm, pc, tournament_size, sel, elt, board_size, frame_board))
        button.grid(row = 1, column = 0, padx = 10, pady = 10, sticky = 'sw')
    window = tk.Tk()
    window.title('Problema do Cavalo (Knights Tour) com GA e Reparo - Luiz Tarralo')
    window.geometry('1000x700')
    window.columnconfigure(1,weight=1)
    window.rowconfigure(1,weight=1)
    window.configure(bg = PRIMARY_COLOR)
    # cria um frame para os widgets de entrada
    frame_input = tk.Frame(window, width = 250, height = 250, bg = SECONDARY_COLOR, highlightbackground = 'black', highlightthickness = 1)
    frame_input.grid(row = 0, column = 0, padx = 10, pady = 10, sticky = 'nw')
    create_inputs(frame_input, callback)
    window.mainloop()
# define a função que cria os widgets de entrada
def create_inputs(frame_input, callback):
    # cria um label para entrada de dados do máximo de gerações
    label_gen = tk.Label(frame_input, text = 'MÁXIMO DE GERAÇÕES', font = ('Roboto Mono Regular', 10), fg = 'white', bg = 'black', relief = 'solid', bd = 1)
    label_gen.grid(row = 0, column = 0, padx = 10, pady = 10, sticky = 'w')
    # cria um entry para entrada de dados do máximo de gerações, com valor padrão 1000
    gen = tk.Entry(frame_input, width = 10, font = ('Roboto Mono Regular', 10))
    gen.grid(row = 0, column = 1, padx = 10, pady = 10, sticky = 'w')
    gen.insert(0, '1000')
    # quando o usuário sair da entry, verifica se o valor é um inteiro maior que 0, se não for mostra uma mensagem de erro e limpa o entry
    gen.bind('<FocusOut>', lambda event: check_int(gen))
    # cria um label para entrada de dados do tamanho da população
    label_pop = tk.Label(frame_input, text = 'TAMANHO DA POPULAÇÃO', font = ('Roboto Mono Regular', 10), fg = 'white', bg = 'black', relief = 'solid', bd = 1)
    label_pop.grid(row = 1, column = 0, padx = 10, pady = 10, sticky = 'w')
    # cria um entry para a entrada do tamanho da população, com valor padrão 100
    pop = tk.Entry(frame_input, width = 10, font = ('Roboto Mono Regular', 10))
    pop.grid(row = 1, column = 1, padx = 10, pady = 10, sticky = 'w')
    pop.insert(0, '100')
    # quando o usuário sair da entry, verifica se o valor é um inteiro maior que 0, se não for mostra uma mensagem de erro e limpa o entry
    pop.bind('<FocusOut>', lambda event: check_int(pop))
    # cria um label para entrada de dados do tamanho do torneio
    label_tour = tk.Label(frame_input, text = 'TAMANHO DO TORNEIO', font = ('Roboto Mono Regular', 10), fg = 'white', bg = 'black', relief = 'solid', bd = 1)
    label_tour.grid(row = 2, column = 0, padx = 10, pady = 10, sticky = 'w')
    # cria um entry para a entrada do tamanho do torneio, com valor padrão 5
    tour = tk.Entry(frame_input, width = 10, font = ('Roboto Mono Regular', 10))
    tour.grid(row = 2, column = 1, padx = 10, pady = 10, sticky = 'w')
    tour.insert(0, '5')
    # quando o usuário sair da entry, verifica se o valor é um inteiro maior que 0 e menor que a população, se não for mostra uma mensagem de erro e limpa o entry
    tour.bind('<FocusOut>', lambda event: check_int_tour(tour, pop))
    # cria um label para entrada de dados da probabilidade de crossover
    label_pc = tk.Label(frame_input, text = 'PROBABILIDADE DE CROSSOVER', font = ('Roboto Mono Regular', 10), fg = 'white', bg = 'black', relief = 'solid', bd = 1)
    label_pc.grid(row = 3, column = 0, padx = 10, pady = 10, sticky = 'w')
    # cria um entry para a entrada da probabilidade de crossover, com valor padrão 0.8
    pc = tk.Entry(frame_input, width = 10, font = ('Roboto Mono Regular', 10))
    pc.grid(row = 3, column = 1, padx = 10, pady = 10, sticky = 'w')
    pc.insert(0, '0.8')
    # quando o usuário sair da entry, verifica se o valor é um float entre 0 e 1, se não for mostra uma mensagem de erro e limpa o entry
    pc.bind('<FocusOut>', lambda event: check_float_pc(pc))
    # cria um label para entrada de dados da probabilidade de mutação
    label_pm = tk.Label(frame_input, text = 'PROBABILIDADE DE MUTAÇÃO', font = ('Roboto Mono Regular', 10), fg = 'white', bg = 'black', relief = 'solid', bd = 1)
    label_pm.grid(row = 4, column = 0, padx = 10, pady = 10, sticky = 'w')
    # cria um entry para a entrada da probabilidade de mutação, com valor padrão 0.05
    pm = tk.Entry(frame_input, width = 10, font = ('Roboto Mono Regular', 10))
    pm.grid(row = 4, column = 1, padx = 10, pady = 10, sticky = 'w')
    pm.insert(0, '0.05')
    # quando o usuário sair da entry, verifica se o valor é um float entre 0 e 1, se não for mostra uma mensagem de erro e limpa o entry
    pm.bind('<FocusOut>', lambda event: check_float_pm(pm))
    # cria uma combobox para a escolha do tipo de seleção
    label_sel = tk.Label(frame_input, text = 'TIPO DE SELEÇÃO', font = ('Roboto Mono Regular', 10), fg = 'white', bg = 'black', relief = 'solid', bd = 1)
    label_sel.grid(row = 5, column = 0, padx = 10, pady = 10, sticky = 'w')
    combo_sel = ttk.Combobox(frame_input, width = 10, font = ('Roboto Mono Regular', 10), values = ['Roleta', 'Torneio'])
    combo_sel.current(0)
    combo_sel.grid(row = 5, column = 1, padx = 10, pady = 10, sticky = 'w')
    # cria uma caixa de seleção para optar por elitismo, o valor padrão é marcado, se estiver marcado elt recebe True, se não recebe False
    label_elt = tk.Label(frame_input, text = 'ELITISMO', font = ('Roboto Mono Regular', 10), fg = 'white', bg = 'black', relief = 'solid', bd = 1)
    label_elt.grid(row = 6, column = 0, padx = 10, pady = 10, sticky = 'w')
    elt = tk.BooleanVar()
    elt.set(True)
    check_elt = tk.Checkbutton(frame_input, variable = elt, onvalue = True, offvalue = False)
    check_elt.grid(row = 6, column = 1, padx = 10, pady = 10, sticky = 'w')
    # cria um label para entrada do tamanho do tabuleiro
    label_board = tk.Label(frame_input, text = 'TAMANHO DO TABULEIRO', font = ('Roboto Mono Regular', 10), fg = 'white', bg = 'black', relief = 'solid', bd = 1)
    label_board.grid(row = 7, column = 0, padx = 10, pady = 10, sticky = 'w')
    # cria um entry para entrada de dados do máximo de gerações, com valor padrão 1000
    board = tk.Entry(frame_input, width = 10, font = ('Roboto Mono Regular', 10))
    board.grid(row = 7, column = 1, padx = 10, pady = 10, sticky = 'w')
    board.insert(0, '8')
    # quando o usuário sair da entry, verifica se o valor é um inteiro maior que 0, se não for mostra uma mensagem de erro e limpa o entry
    board.bind('<FocusOut>', lambda event: check_board(board))
    # cria um botão 'Executar' que permanece desabilitado até que o usuário preencha todos os campos corretamente
    # ao ser clicado, retorna os valores das entradas e da caixa de seleção para a função callback 
    button = tk.Button(frame_input, text = 'Enviar', command = lambda: callback(gen, pop, tour, pc, pm, combo_sel, elt, board))
    button.grid(row = 8, column = 0, padx = 10, pady = 10, sticky = 'w')
# funções que fazem a checagem dos valores de entrada    
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
def check_board(entry_int):
    try: # tenta converter o valor de entry para inteiro
        entryint_value = int(entry_int.get())
        # se for um inteiro, checa se é maior que 0
        if entryint_value <= 0:
            messagebox.showerror('Erro', 'O valor deve ser um inteiro maior que 0.')
            entry_int.delete(0, 'end')
            entry_int.insert(0, '8')
        else: return entry_int
    except:
        messagebox.showerror('Erro', 'O valor deve ser um inteiro maior que 0.')
        entry_int.delete(0, 'end')
        entry_int.insert(0, '8')    
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
# define a função que mostra os gráficos 
def show_graph(best_fit, avg_fit, max_gen):
    window_results = tk.Toplevel()
    window_results.title('Resultados obtidos por AG com Reparo - Luiz Tarralo')
    window_results.geometry('800x800')
    window_results.columnconfigure(0, weight=1)
    window_results.configure(bg = PRIMARY_COLOR)
    # define o eixo x para os gráficos
    x = [i for i in range(max_gen)]
    xticks = np.arange(0, max_gen + 10, (max_gen / 10))
    # cria o gráfico do número de pulos possíveis por geração
    fig_jump = Figure(figsize = (5, 5), dpi = 100)
    y_lim1_inf = np.amin(avg_fit) // 1
    y_lim1_sup = np.amax(best_fit) // 1
    y_pass1 = (y_lim1_sup - y_lim1_inf) // 10
    #y_pass1 = float(f'{(y_lim1_sup - y_lim1_inf):.1f}')
    y_ticks1 = np.arange(y_lim1_inf, y_lim1_sup + y_pass1, y_pass1)
    ax1 = fig_jump.add_subplot(111)
    ax1.set_title('Máximo de movimentos por geração')
    ax1.plot(x, best_fit, color = 'blue', label = 'Máximo de movimentos')
    ax1.plot(x, avg_fit, color = 'red', label = 'Média de movimentos')
    ax1.set_xlabel('Gerações')
    ax1.set_ylabel('Máximo de movimentos')
    ax1.set_xticks(xticks)
    ax1.set_yticks(y_ticks1)
    pos_dist = ax1.get_position()
    ax1.set_position([pos_dist.x0, pos_dist.y0 + pos_dist.height*0.1, pos_dist.width, pos_dist.height*0.9])
    ax1.legend(loc='center left', bbox_to_anchor=(0.5, -0.15))
    ax1.grid(linestyle='--', linewidth=0.5, which = 'major', axis = 'both', alpha = 0.5)
    # cria um label com o valor máximo de movimentos
    label_max_jump = tk.Label(window_results, text = f'MÁXIMO DE MOVIMENTOS: {np.amax(best_fit)}', font = ('Arial', 14), bg = PRIMARY_COLOR, fg = SECONDARY_COLOR)
    label_max_jump.pack(side = tk.BOTTOM)
    # cria o botão para salvar o gráfico
    button_save_jump = tk.Button(window_results, text = 'Salvar gráfico', command = lambda: fig_jump.savefig('max_jumps.png'), bg = SECONDARY_COLOR, fg = PRIMARY_COLOR)
    button_save_jump.pack(side = tk.TOP)
    # cria o canvas para o plot do máximo de movimentos
    canvas_jump = FigureCanvasTkAgg(fig_jump, master = window_results)
    canvas_jump.draw()
    canvas_jump.get_tk_widget().pack(side = tk.BOTTOM, fill = tk.BOTH, expand = True)

main(max_gen = None, pop_size = None, pm = None, pc = None, tournament_size = None, sel = None, elitism = None, board_size = None, frame_board = None)
