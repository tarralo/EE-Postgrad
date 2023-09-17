'''Trabalho 2 - Fluxo de Potência Continuado
   Autor: Luiz A. Tarralo
   Data: 08/05/2023'''

import pandas as pd
import numpy as np
import math
import cmath
import matplotlib.pyplot as plt
import tkinter as tk
import os
import time
from tkinter import messagebox
from tkinter import filedialog

def main(tol, max_iter, file_name, bus_plot, window):
    if tol == None: interface()
    if file_name == None: return 
    data_bus, data_branch = import_data(file_name) # recebe os dados do arquivo data.xlsx
    y_bus = create_y_bus(data_bus, data_branch) # cria a matriz de admitância
    volt, power, slack_index = init_flow(data_bus) # inicializa os valores de tensão, potência e o índice do slack bus
    volt, power, npv, npq = newton_raphson(data_bus, y_bus, tol, max_iter, window) # executa o método de Newton-Raphson
    max_iter_continued, sigma, frame_continued = window_flow(window)
    # cria um botão para executar o fluxo de potência continuado
    button = tk.Button(frame_continued, text="EXECUTAR FLUXO DE POTÊNCIA CONTINUADO", command=lambda: continued_flow(volt, power, y_bus, npv, npq, tol, data_bus, slack_index, max_iter_continued, sigma, bus_plot, window))
    button.grid(row=8, column=0, padx=10, pady=10)
# função que importa os dados do arquivo data.xlsx
def import_data(file_name):
    data_bus = pd.read_excel(file_name, sheet_name='bus', index_col = 'number') 
    data_branch = pd.read_excel(file_name, sheet_name='branch') 
    data_bus = data_bus.fillna(0); data_branch = data_branch.fillna(0) # substitui os valores NaN por 0
    return data_bus, data_branch
# função que cria a matriz de admitância
def create_y_bus(data_bus, data_branch):
    y_bus = np.zeros((len(data_bus.index), len(data_bus.index)), dtype=complex) # cria uma matriz de zeros
    for i in data_branch.index: 
        from_bus, to_bus = int(data_branch.iloc[i]['from']), int(data_branch.iloc[i]['to']) # recebe os valores das colunas from e to
        r, xl, b = data_branch.iloc[i]['r'], data_branch.iloc[i]['xl'], data_branch.iloc[i]['b'] # recebe os valores das colunas r, xl e b
        y_bus[from_bus][to_bus] += -1 / complex(r, xl) # adiciona o valor na posição [from_bus][to_bus]
        y_bus[to_bus][from_bus] += -1 / complex(r, xl) # adiciona o valor na posição [to_bus][from_bus]
        y_bus[from_bus][from_bus] += 1 / complex(r, xl) + complex(0, b / 2) # adiciona o valor na posição [from_bus][from_bus]
        y_bus[to_bus][to_bus] += 1 / complex(r, xl) + complex(0, b / 2) # adiciona o valor na posição [to_bus][to_bus]
    return y_bus
# função que inicializa os valores de tensão, potência e o índice da barra slack/referência
def init_flow(data_bus):
    volt_init = np.zeros(len(data_bus.index), dtype=complex) # cria um vetor de zeros para as tensões
    power_init = np.zeros(len(data_bus.index), dtype=complex) # cria um vetor de zeros para as potências
    slack_index = data_bus.loc[data_bus['type'] == 'SLACK'].index[0] # recebe o índice do slack bus
    slack_angle = data_bus.iloc[slack_index]['angle'] # recebe o valor do ângulo do slack bus
    for i in data_bus.index:
        # inicializa as tensões dos slack e PV
        if data_bus.iloc[i]['type'] == 'SLACK' or data_bus.iloc[i]['type'] == 'PV': volt_init[i] = cmath.rect(data_bus.iloc[i]['voltage'], slack_angle) 
        # inicializa as tensões dos PQ 
        else: volt_init[i] = cmath.rect(1, slack_angle)   
        # inicializa as potências
        power_init[i] = complex(data_bus.iloc[i]['pg'] - data_bus.iloc[i]['pl'], data_bus.iloc[i]['qg'] - data_bus.iloc[i]['ql']) 
    return volt_init, power_init, slack_index
# função que executa o método de Newton-Raphson
def newton_raphson(data_bus, y_bus, tol, max_iter, window):
    # cria um label para informar que o método de Newton-Raphson está sendo executado
    label = None
    label = tk.Label(window, text = "Executando o método de Newton-Raphson...", fg = 'black', bg = 'yellow', highlightbackground='black',highlightthickness=1)
    label.grid(row = 9, column = 0, padx = 5, pady = 5)
    volt, power, slack_index = init_flow(data_bus) # inicializa os valores de tensão, potência e o índice da barra de referência
    # npv = número de barras PV, npq = número de barras PQ
    npv = 0; npq = 0
    for i in data_bus.index: # conta o número de barras PV e PQ
        if data_bus.iloc[i]['type'] == 'PQ': npq += 1
        elif data_bus.iloc[i]['type'] == 'PV': npv += 1
    # delta P e delta Q
    dPdQ = np.zeros(2*npq+npv)
    power_esp = np.copy(power) # potência esperada
    erro = 1 # erro inicial
    iter = 0 # número de iterações
    while erro > tol: # loop principal
        nP = 0; nQ = 0
        for i in data_bus.index: 
            if i != slack_index: # se não for a barra de referência
                current = complex(0,0) # corrente da barra i
                for j in data_bus.index: current += y_bus[i][j] * volt[j] # calcula a corrente da barra i
                power[i] = volt[i] * current.conjugate() # calcula a potência da barra i
                dPdQ[nP] = power_esp[i].real - power[i].real # delta P
                nP += 1 # incrementa o contador de delta P
                if data_bus.iloc[i]['type'] == 'PQ': # se for uma barra PQ
                    dPdQ[nQ+npq+npv] = power_esp[i].imag - power[i].imag # delta Q
                    nQ += 1 # incrementa o contador de delta Q
        erro = np.amax(abs(dPdQ)) # atualiza o erro
        if erro < tol: break # se o erro for menor que a tolerância, para o loop
        j_matrix = jacobian_matrix(volt, power, y_bus, data_bus, npq, npv, slack_index) # calcula a matriz jacobiana
        dTdV = np.linalg.solve(j_matrix, dPdQ) # calcula o delta de tensão e ângulo
        # atualiza os valores de tensão e ângulo
        nT = 0; nV = 0
        for i in data_bus.index:
            if data_bus.iloc[i]['type'] == 'PQ':
                volt[i] = cmath.rect(abs(volt[i]) + dTdV[nT+npq+npv], cmath.phase(volt[i]) + dTdV[nV])
                nT += 1; nV += 1
            elif data_bus.iloc[i]['type'] == 'PV':
                volt[i] = cmath.rect(abs(volt[i]), cmath.phase(volt[i]) + dTdV[nV])
                nV += 1
        iter += 1 # incrementa o número de iterações
        if iter >= max_iter: label.config(text = f"O método de Newton-Raphson não convergiu! Limite de {max_iter} iterações atingido.", fg = 'red'); break # se o número de iterações for maior que o máximo, para o loop 
    # se o método convergiu, imprime o número de iterações
    if iter < max_iter: label.config(text = f"O método de Newton-Raphson convergiu em {iter} iterações.", fg = 'green')
    # calcula a potência da barra slack
    current = complex(0,0)
    for j in data_bus.index: current += y_bus[slack_index][j] * volt[j]
    power[slack_index] = volt[slack_index] * current.conjugate()
    # retorna os valores de tensão, potência, número de iterações, número de barras PV e número de barras PQ
    return volt, power, npv, npq
# função que calcula a matriz jacobiana
def jacobian_matrix(volt, power, y_bus, data_bus, npq, npv, slack_index):
    j1 = np.zeros((npq+npv, npq+npv)) # matriz jacobiana 1
    j2 = np.zeros((npq+npv, npq)) # matriz jacobiana 2
    j3 = np.zeros((npq, npq+npv)) # matriz jacobiana 3
    j4 = np.zeros((npq, npq)) # matriz jacobiana 4
    bus_index = np.delete(data_bus.index.to_numpy(), slack_index) # vetor com os índices das barras PV e PQ
    # calcula a matriz jacobiana 1
    j1_index = {'x': 0, 'y': 0}
    for i in bus_index: # npq + npv
        j1_index['y'] = 0
        for j in bus_index:  # npq + npv
            if i == j: j1[j1_index['x'],j1_index['y']] = -power[i].imag - pow(abs(volt[i]),2) * y_bus[i][i].imag
            else: j1[j1_index['x'],j1_index['y']] = abs(volt[i]) * abs(volt[j]) * (y_bus[i][j].real * math.sin(cmath.phase(y_bus[i][j])) - y_bus[i][j].imag * math.cos(cmath.phase(y_bus[i][j])))
            j1_index['y'] += 1
        j1_index['x'] += 1
    # calcula a matriz jacobiana 2
    j2_index = {'x': 0, 'y': 0}
    for i in bus_index: # npq + npv
        j2_index['y'] = 0
        for j in bus_index: # npq 
            if data_bus.iloc[j]['type'] == 'PQ': 
                if i == j: j2[j2_index['x'], j2_index['y']] = (power[i].real + abs(volt[i])**2 * y_bus[i][i].real) / abs(volt[i])
                else: j2[j2_index['x'], j2_index['y']] = abs(volt[i]) * abs(volt[j]) * (y_bus[i][j].real * math.cos(cmath.phase(y_bus[i][j])) + y_bus[i][j].imag * math.sin(cmath.phase(y_bus[i][j])))
                j2_index['y'] += 1
        j2_index['x'] += 1
    # calcula a matriz jacobiana 3
    j3_index = {'x': 0, 'y': 0}
    for i in bus_index: # npq
        if data_bus.iloc[i]['type'] == 'PQ': # npq
            j3_index['y'] = 0
            for j in bus_index: # npq + npv
                if i == j: j3[j3_index['x'], j3_index['y']] = power[i].real - (abs(volt[i]))**2 * y_bus[i][i].real
                else: j3[j3_index['x'], j3_index['y']] = - abs(volt[i]) * abs(volt[j]) * (y_bus[i][j].real * math.cos(cmath.phase(y_bus[i][j])) + y_bus[i][j].imag * math.sin(cmath.phase(y_bus[i][j])))
                j3_index['y'] += 1
            j3_index['x'] += 1
    # calcula a matriz jacobiana 4
    j4_index = {'x': 0, 'y': 0}
    for i in bus_index: 
        if data_bus.iloc[i]['type'] == 'PQ': # npq
            j4_index['y'] = 0
            for j in bus_index:
                if data_bus.iloc[j]['type'] == 'PQ': # npq
                    if i == j: j4[j4_index['x'], j4_index['y']] = (power[i].imag - (abs(volt[i]))**2 * y_bus[i][i].imag) / (abs(volt[i]))
                    else: j4[j4_index['x'], j4_index['y']] = abs(volt[i]) * (y_bus[i][j].real * math.sin(cmath.phase(y_bus[i][j])) - y_bus[i][j].imag * math.cos(cmath.phase(y_bus[i][j])))
                    j4_index['y'] += 1
            j4_index['x'] += 1
    # concatena as submatrizes e retorna a matriz jacobiana
    return np.concatenate((np.concatenate((j1, j2), axis=1),np.concatenate((j3, j4), axis=1)), axis=0)
# função que executa o fluxo de potência continuado
def continued_flow(volt, power, y_bus, npv, npq, tol, data_bus, slack_index, max_iter_continued, sigma, bus_plot, window):
    start_time = time.time() # tempo inicial
    sigmax = 2*sigma # sigma máximo  
    sigmin = sigmax/1e3 # sigma mínimo 
    lambda_corr = [0.0] # cria uma lista com o valor inicial de lambda
    lambda_pred = [0.0] # cria uma lista para os valores corrigidos de lambda
    volt_corr = abs(np.matrix(volt).T) # cria uma matriz com os valores de tensão e ângulo 
    volt_pred = abs(np.matrix(volt).T) # cria uma matriz para os valores corrigidos de tensão e ângulo
    lamb = 0 # inicializa o valor de lambda
    norm = np.zeros((2*npq+npv+1, 1)) # ek = normalização de T 
    s = 1 # passo
    k = 2 * npq + npv # fixando o índice k como sendo 1, garante que o jacobiano seja singular
    theta_vl = np.zeros((2*npq+npv+1, 1)) # vetor tangente inicial (theta, v, lambda)
    s_vec = np.zeros((2*npq+npv,1)) # nova coluna da matriz jacobiana aumentada
    # faz a contagem de theta, v, P e Q, calcula o vetor tangente inicial e a nova coluna da matriz jacobiana aumentada
    nT = 0; nV = 0; nP = 0; nQ = 0
    for i in data_bus.index:
        if data_bus.iloc[i]['type'] == 'PV':
            theta_vl[nT] = cmath.phase(volt[i])
            # Pi = Pli0 * kl - Pgi0 * kg
            s_vec[nP] = data_bus.iloc[i]['pl']*data_bus.iloc[i]['kl'] - data_bus.iloc[i]['pg']*data_bus.iloc[i]['kg']
            nT += 1; nP += 1
        if data_bus.iloc[i]['type'] == 'PQ':
            theta_vl[nT] = cmath.phase(volt[i])
            theta_vl[nV+npq+npv] = abs(volt[i])
            # Pi = Pli0 * kl - Pgi0 * kg
            s_vec[nP] = data_bus.iloc[i]['pl']*data_bus.iloc[i]['kl'] - data_bus.iloc[i]['pg']*data_bus.iloc[i]['kg']
            s_vec[nQ+npq+npv] = data_bus.iloc[i]['ql']*data_bus.iloc[i]['kl'] - data_bus.iloc[i]['qg']*data_bus.iloc[i]['kg']
            nT += 1; nV += 1; nP += 1; nQ += 1
    # inicializa o loop principal
    counter = 0 # contador de iterações
    fail = False # inicializa a variável de falha
    loop_label = tk.Label(window, text="", bg = 'yellow', fg = 'black', highlightbackground = 'black', highlightthickness = 1)
    loop_label.grid(row = 10, column = 0, padx = 5, pady = 5)
    while np.amin(abs(volt)) >= 1e-2 and not fail:
        counter += 1
        loop_label.config(text=f"Loop {counter} Lambda: {lamb:.5f} Passo: {sigma:.5f}")
        window.update_idletasks()
        # armazena os valores de tensão e do lambda em listas
        lambda_pred = np.append(lambda_pred, lamb)
        volt_pred = np.append(volt_pred, abs(np.matrix(volt).T), axis=1)
        # preditor
        old_lambda = lamb # lambda anterior
        j = jacobian_matrix(volt, power, y_bus, data_bus, npq, npv, slack_index) # calcula a matriz jacobiana
        e_vec = np.zeros((1,2*npq+npv+1)) # nova linha da matriz jacobiana aumentada
        e_vec[0][k] = 1 # adiciona o valor 1 na posição k
        # aumenta a matriz jacobiana 
        j_aum = np.concatenate((j, s_vec), axis=1)
        j_aum = np.concatenate((j_aum, e_vec), axis=0)
        norm[2*npq+npv] = s # normalização de T fixada em s
        tan_vec = np.linalg.solve(j_aum, norm) # solução do vetor tangente
        k = np.argmax(abs(tan_vec)) # atualiza o índice k
        lamb += sigma * tan_vec[2*npq+npv][0] # atualiza o valor de lambda e verifica o sinal
        if lamb <= old_lambda:
            s = -1 # inverte o sinal do passo
            norm[2*npq+npv] = s # atualiza a normalização de T
            tan_vec = np.linalg.solve(j_aum, norm) # solução do vetor tangente
            k = np.argmax(abs(tan_vec)) # atualiza o índice k
            sigma = min(max(sigmin, 2 * sigma), sigmax)
            lamb = old_lambda + sigma * tan_vec[2*npq+npv][0] # atualiza o valor de lambda
        # atualiza os valores de tensão e ângulo
        nT = 0; nV = 0
        for i in data_bus.index:
            if data_bus.iloc[i]['type'] == 'PQ':
                volt[i] = (cmath.rect(abs(volt[i]) + sigma * tan_vec[nV+npq+npv],cmath.phase(volt[i]) + sigma * tan_vec[nT]))
                nT += 1; nV += 1
            if data_bus.iloc[i]['type'] == 'PV':
                volt[i] = (cmath.rect(abs(volt[i]),cmath.phase(volt[i]) + sigma * tan_vec[nT]))
                nT += 1
        # atualiza o vetor de potência
        for i in data_bus.index:
            if i != slack_index:
                curr = complex(0,0)
                for j in data_bus.index: curr += y_bus[i][j] * volt[j]
                power[i] = volt[i] * curr.conjugate()
        lambda_pred = np.append(lambda_pred, lamb)
        volt_pred = np.append(volt_pred, abs(np.matrix(volt).T), axis=1)
        # corretor 
        j_c = jacobian_matrix(volt, power, y_bus, data_bus, npq, npv, slack_index) # calcula a matriz jacobiana
        e_vec = np.zeros((1,2*npq+npv+1)) # nova linha da matriz jacobiana aumentada
        e_vec[0][k] = 1 # adiciona o valor 1 na posição k
        j_aum = np.concatenate((j_c, s_vec), axis=1) # aumenta a matriz jacobiana
        j_aum = np.concatenate((j_aum, e_vec), axis=0) # obtém a matriz jacobiana aumentada
        norm[2*npq+npv] = s # normalização de T fixada em s
        tan_vec = np.linalg.solve(j_aum, norm) # solução do vetor tangente
        k = np.argmax(abs(tan_vec)) # atualiza o índice k
        lamb = old_lambda + sigma * tan_vec[2*npq+npv][0] # atualiza o valor de lambda
        if lamb < 0: break # se o valor de lambda for negativo, para o loop
        erro_i = 1 # erro inicial
        num_it = 0 # número de iterações
        dPdQdL = np.zeros(2*npq+npv+1) # delta P, delta Q e delta lambda
        # Newton-Raphson modificado 
        for _ in range(max_iter_continued):
            nP = 0; nQ = 0
            for i in data_bus.index:
                if i != slack_index:
                    curr = complex(0,0)
                    for j in data_bus.index: curr += y_bus[i][j] * volt[j] 
                    power[i] = volt[i] * curr.conjugate()
                    if data_bus.iloc[i]['type'] == 'PV':
                        dPdQdL[nP] = data_bus.iloc[i]['pg'] * (1+lamb*data_bus.iloc[i]['kg']) - data_bus.iloc[i]['pl'] * (1+lamb*data_bus.iloc[i]['kl']) - power[i].real
                        nP += 1
                    if data_bus.iloc[i]['type'] == 'PQ':
                        dPdQdL[nP] = data_bus.iloc[i]['pg'] * (1+lamb*data_bus.iloc[i]['kg']) - data_bus.iloc[i]['pl'] * (1+lamb*data_bus.iloc[i]['kl']) - power[i].real
                        dPdQdL[nQ+npv+npq] = data_bus.iloc[i]['qg'] - data_bus.iloc[i]['ql'] * (1 + lamb * data_bus.iloc[i]['kl']) - power[i].imag
                        nP += 1; nQ += 1
            erro_i = np.amax(abs(dPdQdL)) # atualiza o erro
            if erro_i < tol: break # se o erro for menor que a tolerância, para o loop
            dTdVdL = np.linalg.solve(j_aum, dPdQdL) # solução do vetor tangente
            # atualiza ângulo, tensão e lambda
            nT = 0; nV = 0
            for i in data_bus.index:
                if data_bus.iloc[i]['type'] == 'PQ':
                    volt[i] = cmath.rect(abs(volt[i]) + dTdVdL[nV+npq+npv], cmath.phase(volt[i]) + dTdVdL[nT])
                    nT += 1; nV += 1
                if data_bus.iloc[i]['type'] == 'PV':
                    volt[i] = cmath.rect(abs(volt[i]), cmath.phase(volt[i]) + dTdVdL[nT])
                    nT += 1
            lamb += dTdVdL[2*npq+npv] # atualiza o valor de lambda
            num_it += 1 # incrementa o número de iterações
            if num_it > max_iter_continued or np.isnan(lamb): 
                fail = True 
                break
            sigma /= np.linalg.norm(tan_vec)   
            sigma = max(min(sigma, sigmax), sigmin)
        lambda_corr = np.append(lambda_corr, lamb) # adiciona o valor de lambda corrigido na lista
        volt_corr = np.append(volt_corr, abs(np.matrix(volt).T), axis=1) # adiciona os valores de tensão corrigido na lista
    label_final = None
    if not fail: 
        final_time = time.time() # tempo final
        print(final_time - start_time)
        # cria um label que informa que o fluxo de potência continuado convergiu
        label_final = tk.Label(window, text = f"O fluxo de potência continuado convergiu em {counter} iterações.", fg = 'green', bg = 'yellow', highlightbackground='black',highlightthickness=1)
        label_final.grid(row = 11, column = 0, padx = 5, pady = 5)
        # cria um botão que chama a função plot_continued
        button = None
        button = tk.Button(window, text="PLOTAR GRÁFICO", command=lambda: plot_continued(data_bus, lambda_corr, volt_corr, lambda_pred, volt_pred, bus_plot))
        button.grid(row = 12, column = 0, padx = 5, pady = 5)
    else: 
        label_final = tk.Label(window, text = f"O fluxo de potência continuado não convergiu. Lambda → ∞!", fg = 'red', bg = 'yellow', highlightbackground='black',highlightthickness=1)
        label_final.grid(row = 11, column = 0, padx = 5, pady = 5)
        return 
# função que cria a interface gráfica
def interface():
    def callback(tol, max_iter, file_name, bus_plot, window):
        tol = float(tol.get())
        max_iter = int(max_iter.get())
        file_name = file_name.get()
        input_bus = bus_plot.get()
        bus_plot = [int(num) for num in input_bus.split(',')]
        # chama a função main
        main(tol, max_iter, file_name, bus_plot, window)
    window = tk.Tk() # cria a janela
    window.title('ANÁLISE COM FLUXO DE POTÊNCIA CONTINUADO') # título da janela
    window.geometry('450x550')
    window.configure(bg = 'white')
    # cria um frame para os parâmetros
    frame = tk.LabelFrame(window, text='PARÂMETROS', padx=5, pady=5)
    frame.grid(row=0, column=0, padx=10, pady=10)
    # cria um label para a tolerância
    tol_label = tk.Label(frame, text='TOLERÂNCIA')
    tol_label.grid(row=0, column=0, padx=5, pady=5)
    # cria um label para o número máximo de iterações
    max_iter_label = tk.Label(frame, text='MÁXIMO DE ITERAÇÕES')
    max_iter_label.grid(row=1, column=0, padx=5, pady=5)
    # cria um label para o nome do arquivo
    file_name_label = tk.Label(frame, text='NOME DO ARQUIVO (.xlsx)')
    file_name_label.grid(row=2, column=0, padx=5, pady=5)
    # cria um label para o número das barras
    bus_plot_label = tk.Label(frame, text='NÚMERO DAS BARRAS (INICIA EM 0)')
    bus_plot_label.grid(row=3, column=0, padx=5, pady=5)
    # cria um entry para a tolerância
    tol = tk.Entry(frame)
    tol.grid(row=0, column=1, padx=5, pady=5)
    tol.insert(0, '1e-5')
    tol.bind('<FocusOut>', lambda event: check_tol(tol))
    # cria um entry para o número máximo de iterações
    max_iter = tk.Entry(frame)
    max_iter.grid(row=1, column=1, padx=5, pady=5)
    max_iter.insert(0, '50')
    max_iter.bind('<FocusOut>', lambda event: check_max_iter(max_iter))
    # cria um entry para o nome do arquivo
    file_name = tk.Entry(frame)
    file_name.grid(row=2, column=1, padx=5, pady=5)
    file_name_button = tk.Button(frame, text='SELECIONAR', command=lambda: file_name.insert(0, filedialog.askopenfilename(initialdir = '/', title = 'Select a file', filetypes = (('xlsx files', '*.xlsx'), ('all files', '*.*')))))
    file_name_button.grid(row=2, column=2, padx=5, pady=5)
    # cria um entry para o número das barras
    bus_plot = tk.Entry(frame)
    bus_plot.grid(row=3, column=1, padx=5, pady=5)
    bus_plot.insert(0, '0,1,2')
    bus_plot.bind('<FocusOut>', lambda event: check_bus_plot(bus_plot))
    # cria um botão que chama a função callback
    button = tk.Button(frame, text="EXECUTAR NEWTON-RAPHSON", command=lambda: callback(tol, max_iter, file_name, bus_plot, window))
    button.grid(row=4, column=0, columnspan=2, pady=10)
    window.mainloop()
# função que verifica se a tolerância é um número real
def check_tol(tol):
    try: 
        entry = float(tol.get())
        if entry <= 0: 
            messagebox.showerror('Erro', 'A tolerância deve ser um número real positivo.')
            entry.delete(0, 'end')
            entry.insert(0, '1e-5')
        else: return entry
    except: 
        messagebox.showerror('Erro', 'A tolerância deve ser um número real.')
        entry.delete(0, 'end')
        entry.insert(0, '1e-5')
# função que verifica se o número máximo de iterações é um número inteiro
def check_max_iter(max_iter):
    try:
        entry = int(max_iter.get())
        if entry <= 0: 
            messagebox.showerror('Erro', 'O número máximo de iterações deve ser um número inteiro positivo.')
            entry.delete(0, 'end')
            entry.insert(0, '50')
        else: return entry
    except:
        messagebox.showerror('Erro', 'O número máximo de iterações deve ser um número inteiro.')
        entry.delete(0, 'end')
        entry.insert(0, '50')
# função que verifica se o número das barras é uma string
def check_bus_plot(bus_plot):
    entry = bus_plot.get()
    if entry == '': 
        messagebox.showerror('Erro', 'O número das barras não pode ser vazio.')
        entry.delete(0, 'end')
        entry.insert(0, '0,1,2')
    else: return entry
# função que cria um frame para os parâmetros do fluxo de potência continuado
def window_flow(window):
    frame_continued = tk.LabelFrame(window, text='PARÂMETROS DO FLUXO DE POTÊNCIA CONTINUADO', padx=5, pady=5)
    frame_continued.grid(row=5, column=0, padx=10, pady=10)
    # cria um label para o número máximo de iterações
    max_iter_label = tk.Label(frame_continued, text='MÁXIMO DE ITERAÇÕES')
    max_iter_label.grid(row=6, column=0, padx=5, pady=5)
    # cria um entry para o número máximo de iterações
    max_iter = tk.Entry(frame_continued)
    max_iter.grid(row=6, column=1, padx=5, pady=5)
    max_iter.insert(0, '50')
    max_iter.bind('<FocusOut>', lambda event: check_max_iter(max_iter))
    # cria um label para o sigma inicial 
    sigma_label = tk.Label(frame_continued, text='CONSTANTE DE PASSO INICIAL')
    sigma_label.grid(row=7, column=0, padx=5, pady=5)
    # cria um entry para o sigma inicial
    sigma = tk.Entry(frame_continued)
    sigma.grid(row=7, column=1, padx=5, pady=5)
    sigma.insert(0, '0.05')
    sigma.bind('<FocusOut>', lambda event: check_sigma(sigma))
    max_iter = int(max_iter.get())
    sigma = float(sigma.get())
    return max_iter, sigma, frame_continued
# função que verifica se o sigma inicial é um número real
def check_sigma(sigma):
    try:
        entry = float(sigma.get())
        if entry <= 0: 
            messagebox.showerror('Erro', 'A constante de passo deve ser um número real positivo.')
            entry.delete(0, 'end')
            entry.insert(0, '0.05')
        else: return entry
    except:
        messagebox.showerror('Erro', 'A constante de passo deve ser um número real.')
        entry.delete(0, 'end')
        entry.insert(0, '0.05')
# função que plota o fluxo de potência continuado
def plot_continued(data_bus, lambda_corr, volt_corr, lambda_pred, volt_pred, bus_plot):
    save_path = filedialog.asksaveasfilename(initialdir = os.getcwd(), title = 'Salvar arquivo', filetypes = (('Arquivo de texto', '*.txt'), ('Todos os arquivos', '*.*')))
    plt.clf()
    manager = plt.get_current_fig_manager()
    manager.set_window_title('Gráfico do Fluxo de potência continuado')
    plt.xlabel('Lambda (pu)')
    plt.ylabel('Tensão (pu)')
    plt.title('Fluxo de potência continuado')
    plt.grid(True,which='both',axis='both',linestyle='dotted')
    for i in bus_plot: 
        if data_bus.loc[i, 'type'] == 'PQ':
            plt.plot(lambda_pred, np.asarray(volt_pred[i,:].T), linestyle = 'dashed', color = 'red', label = f'Barra {i}') # plota o preditor
            plt.plot(lambda_corr, np.asarray(volt_corr[i,:].T), color = 'green', alpha = 0.5) # plota o corretor
            max_index = np.argmax(lambda_corr)
            max_volt = volt_corr[bus_plot[i],max_index]
            max_lambda = lambda_corr[max_index]
        if save_path != '':
                    np.savetxt(save_path + 'lambda.txt', lambda_corr, fmt = '%.5f')
                    np.savetxt(save_path + 'voltage.txt', np.asarray(volt_corr[i,:].T), fmt = '%.5f')
    plt.legend(loc = 'best')
    plt.annotate(f'Ponto de colapso: λ {round(max_lambda, 5)} pu, V {round(max_volt, 5)} pu', xy=(max_lambda, max_volt), xytext=(0, max_volt), bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.9))
    plt.show()
main(tol = None, max_iter = None, file_name = None, bus_plot = None, window = None)