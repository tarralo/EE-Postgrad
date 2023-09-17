'''Aplicação de Evolução Diferencial para minimização da função de Rosenbrock
f(x) = sum(100*(x(i+1)-x(i)^2)^2+(1-x(i))^2) for i = 1:n-1
com x(i) no intervalo [-1,2]
Parâmetros padronizados: 
max_gen = 1000, pop_size = 100, pc = 0.9, f = 0.5'''

import numpy as np, tkinter as tk, matplotlib.pyplot as plt
from tkinter import messagebox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from matplotlib import cm

PRIMARY_COLOR = 'white'
SECONDARY_COLOR = 'gray'

def main(max_gen, pop_size, pc, f, x1, x2, window):
    if max_gen == None: interface() # se o usuário não informar os parâmetros, abre a interface
    if window == None: return  # se o usuário fechar a janela, encerra o programa
    label_print = tk.Label(window, text = '', bg = PRIMARY_COLOR, fg = 'black')
    label_print.grid(row = 1, column = 0, columnspan = 2, sticky = 'nsew')
    x_limit = [x1, x2] # limites inferior e superior do espaço de busca
    n = 2 # número de dimensões
    list_fit = np.zeros(max_gen, dtype = float); list_sol = np.zeros((max_gen, n), dtype = float) # vetores para armazenar os resultados
    pop = np.random.uniform(x_limit[0], x_limit[1], (pop_size, n)) # população inicial
    fitness = np.array([rosenbrock(x) for x in pop]) # fitness da população inicial
    best_idx = np.argmin(fitness) # índice do melhor indivíduo
    best_fit = rosenbrock(pop[best_idx]) # valor da função objetivo para o melhor indivíduo
    best_sol = pop[best_idx] # melhor indivíduo
    list_fit[0] = best_fit; list_sol[0] = best_sol # armazena o melhor indivíduo
    # inicia a evolução diferencial
    for i in range(max_gen):
        for j in range(pop_size):
            candidates = np.random.choice(pop_size, 3, replace=False) # seleciona 3 indivíduos diferentes
            base, target, rand = pop[candidates] # atribui os indivíduos selecionados
            mutant = base + f*(target - rand) # gera o mutante
            trial = np.where(np.random.uniform(0,1,n) < pc, mutant, pop[j]) # gera o trial (combinação linear entre o mutante e o indivíduo da população)
            trial_fit = rosenbrock(trial) # calcula o fitness do trial
            if trial_fit < fitness[j]: 
                pop[j] = trial # seleção
                fitness[j] = trial_fit # atualiza o fitness
                if trial_fit < fitness[best_idx]: # verifica se o trial é o melhor indivíduo
                    best_idx = j # atualiza o índice do melhor indivíduo
                    best_sol = trial # atualiza o melhor indivíduo
        best_idx = np.argmin([rosenbrock(x) for x in pop]) # índice do melhor indivíduo
        best_fit = rosenbrock(pop[best_idx]) # valor da função objetivo para o melhor indivíduo
        best_sol = pop[best_idx] # melhor indivíduo
        list_fit[i+1] = best_fit; list_sol[i+1] = best_sol # armazena o melhor indivíduo
        label_print.config(text = f'Geração {i+1}: Mínimo em {best_sol} \n Fitness {best_fit}') # atualiza o label
        window.update_idletasks()
        # se o mínimo global for encontrado, para a evolução
        if best_fit < 1e-17: 
            num_gen = i+1 
            break
    label_print.destroy()
    graph = plot_fit(list_fit, list_sol, num_gen, window) # plota o gráfico do fitness
    # cria um botão para mostrar o gráfico
    button_plot = tk.Button(graph, text = 'Mostrar Plot de Rosenbrock', command = lambda: plot(x_limit, list_fit, list_sol, num_gen))
    button_plot.grid(row = 2, column = 0, columnspan = 2, padx = 10, pady = 10, sticky = 'nsew')
    return 
# função objetivo
def rosenbrock(x):
    result = 0
    for i in range(len(x)-1): result += 100*np.power((x[i+1] - np.power(x[i],2)),2) + np.power((x[i]-1),2)
    return result 
# função para plotar o gráfico do fitness
def plot_fit(list_fit, list_sol, num_gen, window):
    list_fit = list_fit[:num_gen+1]
    list_sol = list_sol[:num_gen+1]
    list_avg = np.zeros(num_gen+1, dtype = float)
    for i in range(num_gen+1): list_avg[i] = np.mean(list_fit[:i+1])
    x = np.arange(0, num_gen+1, 1)
    xtickes = np.arange(0, num_gen + (num_gen/100), 10)
    yticks = np.arange(0, max(list_fit) + (max(list_fit)/100), max(list_fit)/10)
    # cria um frame para os gráficos
    graph = tk.LabelFrame(window, text = 'Gráficos de Fitness (Gerações x Fitness)', bg = PRIMARY_COLOR)
    graph.grid(row = 0, column = 1, padx = 10, pady = 10, sticky = 'nsew', rowspan = 3)
    label_final = tk.Label(graph, text = f'MÍNIMO EM {num_gen} GERAÇÕES \n F(X,Y) = {list_fit[-1]} \n X,Y = {list_sol[-1]}', bg = SECONDARY_COLOR, fg = 'black')
    label_final.grid(row = 0, column = 0, sticky = 'nsew') 
    # cria o gráfico
    fig = Figure(figsize = (5,5), dpi = 100)
    ax = fig.add_subplot(111)
    ax.set_xticks(xtickes)
    ax.set_yticks(yticks)
    ax.plot(x, list_fit, label = 'Curva de Fitness')
    ax.plot(x, list_avg, label = 'Média de Fitness')
    ax.set_xlabel('Gerações')
    ax.set_ylabel('Fitness')
    pos_dist = ax.get_position()
    ax.set_position([pos_dist.x0 + pos_dist.width*0.1, pos_dist.y0, pos_dist.width*0.9, pos_dist.height])
    ax.legend(loc = 'best')
    ax.grid(linestyle = '--', linewidth = 0.5, which = 'both', axis = 'both', alpha = 0.5)
    canvas = FigureCanvasTkAgg(fig, master = graph)
    canvas.draw()
    canvas.get_tk_widget().grid(row = 1, column = 0, padx = 10, pady = 10, sticky = 'nsew')
    # cria um botão para salvar o gráfico
    button_save = tk.Button(graph, text = 'Salvar gráfico', command = lambda: fig.savefig('fitness.png'))
    button_save.grid(row = 3, column = 0, columnspan = 2, padx = 10, pady = 10, sticky = 'nsew')
    return graph
# função para plotar o gráfico
def plot(x_limit, list_fit, list_sol, num_gen):
    # tira os zeros do vetor
    list_fit = list_fit[:num_gen+1]
    list_sol = list_sol[:num_gen+1]
    x = np.arange(x_limit[0], x_limit[1], 0.01) # cria o vetor x
    y = np.arange(x_limit[0], x_limit[1], 0.01) # cria o vetor y
    x, y = np.meshgrid(x, y) # cria a malha
    z = 100*np.power((y - np.power(x,2)),2) + np.power((x-1),2) # calcula o valor da função objetivo
    figure = plt.figure()
    manager = plt.get_current_fig_manager()
    manager.set_window_title('Função de Rosenbrock - Melhores Indivíduos')
    ax = figure.add_subplot(111, projection = '3d')
    surf = ax.plot_surface(x, y, z, cmap = cm.magma, linewidth = 0, antialiased = False) # plota a superfície
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.set_title('Pontos de mínimo da função de Rosenbrock')
    for i in range(len(list_sol)): ax.scatter(list_sol[i][0], list_sol[i][1], list_fit[i], color = 'blue', marker = 'o', s = 50)
    figure.colorbar(surf, shrink = 0.5, aspect = 5)
    figure.tight_layout()
    plt.show()
# função para a interface
def interface():
    def callback(max_gen, pop_size, pc, f, x1, x2, window):
        main(int(max_gen), int(pop_size), float(pc), float(f), float(x1), float(x2), window)
    window = tk.Tk() # cria a janela
    window.title('Evolução Diferencial para Minimização da Função de Rosenbrock - Luiz Tarralo') # título da janela
    window.geometry('850x700') # dimensões da janela
    window.configure(bg = PRIMARY_COLOR) # cor de fundo da janela
    frame = tk.LabelFrame(window, text = 'Parâmetros', bg = PRIMARY_COLOR) # cria um frame
    frame.grid(row = 0, column = 0, padx = 10, pady = 10, sticky = 'nw') # posiciona o frame
    # cria os labels 
    label_max_gen = tk.Label(frame, text = 'Número máximo de gerações:', bg = PRIMARY_COLOR)
    label_pop_size = tk.Label(frame, text = 'Tamanho da população:', bg = PRIMARY_COLOR)
    label_pc = tk.Label(frame, text = 'Probabilidade de cruzamento:', bg = PRIMARY_COLOR)
    label_f = tk.Label(frame, text = 'Fator de escala:', bg = PRIMARY_COLOR)
    label_x1 = tk.Label(frame, text = 'Insira um limite inferior para x:', bg = PRIMARY_COLOR)
    label_x2 = tk.Label(frame, text = 'Insira um limite superior para x:', bg = PRIMARY_COLOR)
    # posiciona os labels
    label_max_gen.grid(row = 0, column = 0, padx = 10, pady = 10, sticky = 'w')
    label_pop_size.grid(row = 1, column = 0, padx = 10, pady = 10, sticky = 'w')
    label_pc.grid(row = 2, column = 0, padx = 10, pady = 10, sticky = 'w')
    label_f.grid(row = 3, column = 0, padx = 10, pady = 10, sticky = 'w')
    label_x1.grid(row = 4, column = 0, padx = 10, pady = 10, sticky = 'w')
    label_x2.grid(row = 5, column = 0, padx = 10, pady = 10, sticky = 'w')
    # cria as entradas de texto
    entry_max_gen = tk.Entry(frame, width = 10, bg = SECONDARY_COLOR)
    entry_pop_size = tk.Entry(frame, width = 10, bg = SECONDARY_COLOR)
    entry_pc = tk.Entry(frame, width = 10, bg = SECONDARY_COLOR)
    entry_f = tk.Entry(frame, width = 10, bg = SECONDARY_COLOR)
    entry_x1 = tk.Entry(frame, width = 10, bg = SECONDARY_COLOR)
    entry_x2 = tk.Entry(frame, width = 10, bg = SECONDARY_COLOR)
    # posiciona as entradas de texto
    entry_max_gen.grid(row = 0, column = 1, padx = 10, pady = 10, sticky = 'w')
    entry_pop_size.grid(row = 1, column = 1, padx = 10, pady = 10, sticky = 'w')
    entry_pc.grid(row = 2, column = 1, padx = 10, pady = 10, sticky = 'w')
    entry_f.grid(row = 3, column = 1, padx = 10, pady = 10, sticky = 'w')
    entry_x1.grid(row = 4, column = 1, padx = 10, pady = 10, sticky = 'w')
    entry_x2.grid(row = 5, column = 1, padx = 10, pady = 10, sticky = 'w')
    # checa se os valores são válidos
    entry_max_gen.insert(0,'1000')
    entry_max_gen.bind('<FocusOut>', lambda event: check_gen(entry_max_gen))
    entry_pop_size.insert(0,'100')
    entry_pop_size.bind('<FocusOut>', lambda event: check_pop(entry_pop_size))
    entry_pc.insert(0,'0.9')
    entry_pc.bind('<FocusOut>', lambda event: check_pc(entry_pc))
    entry_f.insert(0,'0.5')
    entry_f.bind('<FocusOut>', lambda event: check_f(entry_f))
    entry_x1.insert(0,'-2.0')
    entry_x1.bind('<FocusOut>', lambda event: check_x1(entry_x1))
    entry_x2.insert(0,'1.0')
    entry_x2.bind('<FocusOut>', lambda event: check_x2(entry_x2))
    # cria o botão para iniciar a evolução
    start = tk.Button(frame, text = 'Iniciar Evolução Diferencial', command = lambda: callback(entry_max_gen.get(), entry_pop_size.get(), entry_pc.get(), entry_f.get(), entry_x1.get(), entry_x2.get(), window), bg = SECONDARY_COLOR)
    start.grid(row = 6, column = 0, padx = 10, pady = 10, sticky = 'w')
    window.mainloop()
# funções para checar se os valores são válidos
def check_gen(entry):
    try: 
        entry = int(entry.get())
        if entry <= 0:
            messagebox.showerror('Erro', 'O número máximo de gerações deve ser maior que zero.')
            entry.delete(0, 'end')
            entry.insert(0, '1000')
        else: return entry 
    except:
        messagebox.showerror('Erro', 'O número máximo de gerações deve ser um número inteiro.')
        entry.delete(0, 'end')
        entry.insert(0, '1000')
def check_pop(entry):
    try:
        entry = int(entry.get())
        if entry <= 0:
            messagebox.showerror('Erro', 'O tamanho da população deve ser maior que zero.')
            entry.delete(0, 'end')
            entry.insert(0, '100')
        else: return entry
    except:
        messagebox.showerror('Erro', 'O tamanho da população deve ser um número inteiro.')
        entry.delete(0, 'end')
        entry.insert(0, '100')
def check_pc(entry):
    try:
        entry = float(entry.get())
        if entry <= 0 or entry > 1:
            messagebox.showerror('Erro', 'A probabilidade de cruzamento deve estar entre 0 e 1.')
            entry.delete(0, 'end')
            entry.insert(0, '0.9')
        else: return entry
    except:
        messagebox.showerror('Erro', 'A probabilidade de cruzamento deve ser um número real.')
        entry.delete(0, 'end')
        entry.insert(0, '0.9')
def check_f(entry):
    try:
        entry = float(entry.get())
        if entry <= 0 or entry > 2:
            messagebox.showerror('Erro', 'O fator de escala deve estar entre 0 e 2.')
            entry.delete(0, 'end')
            entry.insert(0, '0.5')
        else: return entry
    except:
        messagebox.showerror('Erro', 'O fator de escala deve ser um número real.')
        entry.delete(0, 'end')
        entry.insert(0, '0.5')
def check_x1(entry):
    try:
        entry = float(entry.get())
        if entry < -10 or entry > 0:
            messagebox.showerror('Erro', 'O limite inferior de x deve estar entre -10 e 0.')
            entry.delete(0, 'end')
            entry.insert(0, '-2.0')
        return entry
    except:
        messagebox.showerror('Erro', 'O limite inferior de x deve ser um número real.')
        entry.delete(0, 'end')
        entry.insert(0, '-2.0')
def check_x2(entry):
    try:
        entry = float(entry.get())
        if entry < 0 or entry > 10:
            messagebox.showerror('Erro', 'O limite superior de x deve estar entre 0 e 10.')
            entry.delete(0, 'end')
            entry.insert(0, '1.0')
        return entry
    except:
        messagebox.showerror('Erro', 'O limite superior de x deve ser um número real.')
        entry.delete(0, 'end')
        entry.insert(0, '1.0')
if __name__ == '__main__':
    main(max_gen = None, pop_size = None, pc = None, f = None, x1 = None, x2 = None, window = None)

