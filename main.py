from statistics import variance
import networkx as nx
import matplotlib.pyplot as plt
import random
import numpy as np
import time, threading
import math
from collections import defaultdict
from bisect import bisect_left
from datetime import datetime
from sklearn.linear_model import LinearRegression
from sqlalchemy import ForeignKey

import process_dynamics
import average_distribution_annd
import average_distribution_value

# Данная программа предназначена для симуляции и анализа сетей
# Результат работы: распределение средней степени соседей и индекса дружбы в 
# реальных и синтетических сетях
# В программе моделируются сети Барабаши-Альберт и тройственного замыкания (Хольме и Ким) 
### Программу можно запустить двумя способами
# 1) Вручную конфигурируя параметры, использовав основной файл main.py
# 2) Запустив main-ui.py, в котором предоставляется простой интерфейс, упрощающий выбор параметров эксперимента  

# Для запуска с интерфейсом загрузите файл main-ui.py в интерпретатор Python

# Для ручного запуска
# Отредактируйте параметры ниже:
# a) Выберите значение переменной 'experiment_type_num', ответственной за тип проводимого эксперимента
# b1) Для реальных сетей отредактируйте параметр 'filename'
# b2) Для модели BA измените параметры 'n, m'
# b3) Для модели TC измените параметры 'n, m, p'
#
# Для сохранения вывода и/или получения усредненных результатов синтетических сетей измените параметр 'save_data' на True 
#
# Массив 'focus_indices' позволяет записывать траекторию для величин s, a, b (сумма степеней соседей, средняя степень соседей, индекс дружбы)
# для узлов с заданными индексами, напр. [10, 50, 100, 1000]
# 'focus_period' задает период записи значений величин для данных узлов
# process_dynamics.py обрабатывает средние траектории данных узлов
# 
# для получения распределений величин: средняя степень узлов, индекс дружбы, ANND и дисперсия средних степеней 
# измените параметр 'values_to_analyze'
#
# файлы с префиксом 'hist_' содержат гистограммы на линейных и логарифмических шкалах, а также результат линейной регрессии
# файлы с префиксом 'out_' содержат необработанные результаты для узлов из массива 'focus_indices'
# пожалуйста, с осторожностью редактируйте выходные файлы и папки, чтобы избежать ошибок 

# Простестировано на версии Python 3.7.6

#                        0               1              2         3              
experiment_types = ["from_file", "barabasi-albert", "triadic", "test"]
# Измените данное значение для выбора типа эксперимента из массива выше
experiment_type_num = 0
# Параметры для синтетических сетей
number_of_experiments = 10
n = 750
m = 5
p = 0.75 # для модели тройственного замыкания
focus_indices = [50, 100]
focus_period = 50

save_data = True

ALPHA = "alpha"
BETA = "beta"
DEG_ALPHA = "deg-alpha"
NONE = "none"
# Измените данные значения для получения распределения средней степени (ALPHA) 
# или индекса дружбы (BETA) или средней степени соседей ANND (DEG_ALPHA)
value_to_analyze = ALPHA
values_to_analyze = [ALPHA]
value_log_binning = False

# Для экспериментов над реальными сетями
#filename = "phonecalls.edgelist.txt"
#filename = "amazon.txt"
#filename = "musae_git_edges.txt"
#filename = "artist_edges.txt"
#filename = "soc-twitter-follows.txt"
#filename = "soc-flickr.txt"
#filename = "soc-twitter-follows-mun.txt"
#filename = "citation.edgelist.txt"
filename = "ia-facebook-wall-wosn-dir.edges"
#filename = "soc-epinions-trust-dir.edges" # unsorted
#filename = "ca-cit-HepPh-sorted.edges"
#filename = "ia-yahoo-messages.mtx"
#filename = "ia-stackexch-user-marks-post-und-sorted.edges"
#filename = "fb-wosn-friends.edges"
#filename = "sx-superuser-sorted.txt"
#filename = "sx-askubuntu-sorted.txt"
#filename = "web-google-dir.txt"
#filename = "ia-enron-email-dynamic-sorted.edges"

real_directed = False
real_dynamic = True
dynamic_iterations = [ 1000, 5000, 10000, 15000, 20000, 25000, 30000, 35000
                     , 40000, 45000, 50000, 55000, 60000, 65000, 70000, 75000
                     , 80000, 100000, 120000, 140000, 160000, 180000
                    #  , 200000, 220000, 240000, 260000, 280000
                    #  , 300000, 320000, 340000, 360000, 380000
                     ]
# dynamic_focus_nodes = [
#     [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
#     [41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60],
#     [91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110]
# ]
dynamic_focus_nodes = []
dynamic_focus_nodes_ranges = [(21, 50), (101, 130), (171, 200)]

# Не меняйте вручную
progress_bar = None

def get_neighbor_summary_degree(graph, node, directed = False):
    neighbors_of_node = graph.neighbors(node)
    if not directed:
        return sum(graph.degree(neighbor) for neighbor in neighbors_of_node)
    else:
        return sum(graph.in_degree(neighbor) for neighbor in neighbors_of_node)


def get_neighbor_average_degree(graph, node, si=None, directed = False):
    if not si:
        si = get_neighbor_summary_degree(graph, node, directed=directed)
    if not directed:
        return si / graph.degree(node)
    else:
        deg = graph.in_degree(node)  
        return 0 if deg == 0 else si / deg


def get_friendship_index(graph, node, ai=None, directed = False):
    if not ai:
        ai = get_neighbor_average_degree(graph, node, directed=directed)
    if not directed:
        return ai / graph.degree(node)
    else:
        deg = graph.in_degree(node)
        return 0 if deg == 0 else ai / deg


# Данная функция получает суммарную степень, среднюю степень соседей, индекс дружбы для заданных вершин
# G - граф
# focus_indices - отслеживаемые узлы
# s_a_b_focus - тройка ([s], [a], [b]) для каждого узла из 'focus_indices'
# k - текущая интерация. Необходима для пропуска узлов, которые еще не появились
def update_s_a_b(G, focus_indices, s_a_b_focus, k):
    for i in range(len(s_a_b_focus)):
                s_a_b = s_a_b_focus[i]
                focus_ind = focus_indices[i]
                if focus_ind < k:
                    si = get_neighbor_summary_degree(G, focus_ind)
                    ai = get_neighbor_average_degree(G, focus_ind, si)
                    bi = get_friendship_index(G, focus_ind, ai)
                    s_a_b[0].append(si)
                    s_a_b[1].append(round(ai, 4))
                    s_a_b[2].append(round(bi, 4))


def plot_s_a_b(s_a_b_focus):
    for i in range(len(focus_indices)):
        s_a_b = s_a_b_focus[i]
        s_focus_xrange = [x * focus_period for x in range(len(s_a_b[0]))]
        plt.plot(s_focus_xrange, s_a_b[0])
        plt.title(f"Динамика суммарной степени соседей узла {focus_indices[i]}")
        plt.xlabel("t")
        plt.ylabel("s")
        plt.show()
        s_focus_xrange = [x * focus_period for x in range(len(s_a_b[1]))]
        plt.plot(s_focus_xrange, s_a_b[1])
        plt.title(f"Динамика средней степени соседей узла {focus_indices[i]}")
        plt.xlabel("t")
        plt.ylabel("a")
        plt.show()
        s_focus_xrange = [x * focus_period for x in range(len(s_a_b[2]))]
        plt.plot(s_focus_xrange, s_a_b[2])
        plt.title(f"Динамика индекса дружбы для узла {focus_indices[i]}")
        plt.xlabel("t")
        plt.ylabel("b")
        plt.show()


# возвращает два отображения
# первое - степень -> (сумма средних степеней, количество средних степеней)
# второе - степень -> [средняя_степень1, средняя_степень2, ...] (напр. для вычисления дисперсии)
def acquire_deg_alpha(graph):
    graph_nodes = graph.nodes()
    deg_alpha = dict()
    deg_alphas = defaultdict(list)

    for node in graph_nodes:
        degree = graph.degree(node)
        alpha = get_neighbor_average_degree(graph, node)
        deg_alpha_cur = deg_alpha.get(degree, (0, 0))
        deg_alpha[degree] = (deg_alpha_cur[0] + alpha, deg_alpha_cur[1] + 1)
        # needed to calculate sigma
        deg_alphas[degree].append(alpha)
    return deg_alpha, deg_alphas


# deg_alpha = отображение степени на суммарные степени соседей и количество таких средних степеней
def visualize_deg_alpha_distribution(deg_alpha, deg_alphas):
    degrees = deg_alpha.keys()
    alphas = []
    log_alphas = []
    log_degs = [math.log(deg, 10) for deg in degrees]
    for key in degrees:
        alpha = deg_alpha[key][0] / deg_alpha[key][1]
        deg_alpha[key] = (alpha, deg_alpha[key][1])
        alphas.append(alpha)
        log_alphas.append(math.log(alpha, 10))
    #plt.scatter(degrees, alphas, s = 3)
    plt.scatter(log_degs, log_alphas, s = 3)
    plt.title('График корреляции степеней и ср. ст. соседей (ANND)')
    plt.xlabel("log10(k)")
    plt.ylabel("log ANND")
    plt.show()

    sigmas = []
    log_sigmas = []
    for degree in deg_alphas.keys():
        sigma2 = 0
        for alpha in deg_alphas[degree]:
            sigma2 += math.pow((alpha - deg_alpha[degree][0]), 2)
        sigma2 /= len(deg_alphas[key])
        sigma = math.sqrt(sigma2)
        sigmas.append(sigma)
        log_sigmas.append(0 if sigma <= 0 else math.log(sigma, 10))

    #plt.scatter(degrees, sigmas, s = 3)
    plt.scatter(log_degs, log_sigmas, s = 3)
    plt.title('Корреляция степени и дисперсии ср. ст. соседей (ANND)')
    plt.xlabel("log10(k)")
    plt.ylabel("дисперсия")
    plt.show()


# записывает распределение ANND для каждой степени, а также дисперсию средних степеней
def write_deg_alpha_distribution(deg_alpha, deg_alphas, filename, overwrite):
    degrees = deg_alpha.keys()
    alphas = []
    for key in degrees:
        alpha = deg_alpha[key][0] / deg_alpha[key][1]
        deg_alpha[key] = (alpha, deg_alpha[key][1])
        alphas.append(alpha)

    deg_sigma = dict()
    for degree in deg_alphas.keys():
        sigma2 = 0
        for alpha in deg_alphas[degree]:
            sigma2 += math.pow((alpha - deg_alpha[degree][0]), 2)
        sigma2 /= len(deg_alphas[key])
        sigma = math.sqrt(sigma2)
        deg_sigma[degree] = sigma

    filename_a = f"{filename.split('.txt')[0]}_dist_as.txt"
    file_a = open(filename_a, "w+" if overwrite else "a+") 
    filename_sig = f"{filename.split('.txt')[0]}_dist_sig.txt"
    file_sig = open(filename_sig, "w+" if overwrite else "a+") 

    file_a.write(" ".join([f"({deg_alpha[degree][0]}, {degree})" for degree in deg_alpha.keys()]))
    file_a.write("\n")
    file_sig.write(" ".join([f"({deg_sigma[degree]}, {degree})" for degree in deg_alpha.keys()]))
    file_sig.write("\n")

    file_a.close()
    file_sig.close()
    return [filename_a, filename_sig]


# получить значение заданной величины для каждого узла в сети
# возвращает пару типа ([value_1, value_2, ...], max_value)
def acquire_values(graph, value_to_analyze):
    graph_nodes = graph.nodes()
    vs = []
    maxv = 0
    for node in graph_nodes:
        new_v = 0
        if value_to_analyze == ALPHA:
            new_v = get_neighbor_average_degree(graph, node)
        elif value_to_analyze == BETA:
            new_v = get_friendship_index(graph, node, directed= nx.is_directed(graph))
        else:
            raise Exception(f"Incorrect value to analyze {value_to_analyze}. Check experiment parameters block. Is it ALPHA or BETA?")
        if new_v > maxv:
            maxv = new_v
        vs.append(new_v)
    return (vs, maxv)


# суммирует значения величины для каждого отрезка размера 1 (напр. [1,2) or [5,6))
def accumulate_value(vs, bins, filename, value_to_analyze, overwrite):
    n, bins = np.histogram(vs, bins)
    value_id = ""
    if value_to_analyze == BETA:
        value_id = "b"
    elif value_to_analyze == ALPHA:
        value_id = "a"
    else:
        raise Exception(f"Incorrect value to analyze {value_to_analyze}. Check experiment parameters block. Is it ALPHA or BETA?")
    filename_v = f"{filename.split('.txt')[0]}_dist_{value_id}.txt"
    file_v = open(filename_v, "w+" if overwrite else "a+") 
    file_v.write(" ".join([str(int(x)) for x in n]))
    file_v.write("\n")
    file_v.close()
    return [filename_v]


# линейный биннинг на линейных и логарифмических осях
def obtain_value_distribution_linear_binning(vs, maxv, filename, value_name):
    # n=values, bins=edges of bins
    n, bins, _ = plt.hist(vs, bins=range(int(maxv)), rwidth=0.85)
    plt.close()

    # оставить только ненулевые значения
    n_bins = zip(n, bins)
    n_bins = list(filter(lambda x: x[0] > 0, n_bins))
    n, bins = [ a for (a,b) in n_bins ], [ b for (a,b) in n_bins ]
    
    # получить распределение на логарифмических осях
    lnt, lnb = [], []
    for i in range(len(bins) - 1):
        if (n[i] != 0):
            lnt.append(math.log(bins[i]+1))
            lnb.append(math.log(n[i]) if n[i] != 0 else 0)

    # подготовка к линейной регрессии
    np_lnt = np.array(lnt).reshape(-1, 1)
    np_lnb = np.array(lnb)

    # линейная регрессия, чтобы найти экспоненту степенного закона
    model = LinearRegression()
    model.fit(np_lnt, np_lnb)
    linreg_predict = model.predict(np_lnt)

    if save_data:
        [directory, filename] = filename.split('/')
        with open(directory + "/hist_" + filename, "w") as f:
            f.write("t\tb\tlnt\tlnb\tlinreg\t k=" + str(model.coef_) + ", b=" + str(model.intercept_) + "\n")

            for i in range(len(lnb)):
                f.write(str(bins[i]) + "\t" + str(int(n[i])) + "\t" + str(lnt[i]) + "\t" + str(lnb[i]) + "\t" + str(linreg_predict[i]) + "\n")        
    else:
        plt.scatter(lnt, lnb)
        plt.title(f"Распределение {value_name}")
        plt.xlabel("log k")
        plt.ylabel(f"log {value_name}")
        plt.show()


# логарифмический биннинг на логарифмических шкалах
def obtain_value_distribution_log_binning(bins, hist, value_name):
    fig = plt.figure()
    ax = plt.gca()
    ax.scatter(bins[:-1], hist)
    ax.set_yscale('log')
    ax.set_xscale('log')
    
    plt.title(f"Распределение {value_name} (log-биннинг)")
    plt.xlabel("log k")
    plt.ylabel(f"log {value_name}")
    plt.show()

def analyze_mult_val_graph(graph, filename, overwrite=False):
    filenames = []
    for value in values_to_analyze:
        filenames += analyze_val_graph(graph, filename, value, overwrite)
    return filenames

# Получение гистограмм ANND и индекса дружбы 
def analyze_val_graph(graph, filename, value_to_analyze, overwrite=False):
    graph_nodes = graph.nodes()

    if value_to_analyze == DEG_ALPHA:
        deg_alpha, deg_alphas = acquire_deg_alpha(graph)
        
        if save_data:
            return write_deg_alpha_distribution(deg_alpha, deg_alphas, filename, overwrite)
        else:
            visualize_deg_alpha_distribution(deg_alpha, deg_alphas)
            return []
            
    elif value_to_analyze == ALPHA or value_to_analyze == BETA:
        # value = индекс дружбы (бета) or средняя степень соседей (альфа) 
        vs, maxv = acquire_values(graph, value_to_analyze)

        bins = None
        if value_log_binning:
            base = 1.5
            log_max = math.log(maxv, base) 
            bins = np.logspace(0, log_max, num=math.ceil(log_max), base=base)
        else:
            bins = np.linspace(0, math.ceil(maxv), num=int(math.ceil(maxv)+1))

        if save_data:
            #n, bins, _ = plt.hist(vs, bins=bins, rwidth=0.85)
            return accumulate_value(vs, bins, filename, value_to_analyze, overwrite)
        else:
            hist, bins = np.histogram(vs, bins)
            
            if not value_log_binning:
                obtain_value_distribution_linear_binning(vs, maxv, filename, value_to_analyze)
            else:
                obtain_value_distribution_log_binning(bins, hist, value_to_analyze)
            return []


def obtain_value_distribution(filenames):
    annd_files = filter(lambda x: "_as." in x or "_sig." in x, filenames)
    a_beta_files = filter(lambda x: "_a." in x or "_b." in x, filenames)
    if save_data:
        average_distribution_annd.obtain_average_distributions(annd_files)
        average_distribution_value.obtain_average_distribution(a_beta_files)            


def init_focus_indices_files(filename):
    files = []
    now = datetime.now()
    for ind in focus_indices:
        f_s = open(f"{filename}_{ind}_s.txt", "a")
        f_a = open(f"{filename}_{ind}_a.txt", "a")
        f_b = open(f"{filename}_{ind}_b.txt", "a")
        files.append((f_s, f_a, f_b))
    
    for i in range(len(focus_indices)):
        for f in files[i]:
            f.write("> n=" + str(n) + " m=" + str(m) + " " + now.strftime("%d/%m/%Y %H:%M:%S") + "\n")
    
    return files


def process_simulated_network(graph, result, files, filename):
    for i in range(len(focus_indices)):
        for j in range(len(result[i])):
            files[i][j].write(" ".join(str(x) for x in result[i][j]) + "\n")    
    
    return analyze_mult_val_graph(graph, filename + ".txt")


# 0 - Сеть берется из файла 
def experiment_file():
    graph_type = nx.Graph 
    if real_directed:
        graph_type = nx.DiGraph
        
    if not real_dynamic:
        graph = nx.read_edgelist(filename, create_using = graph_type)

        filenames = analyze_mult_val_graph(graph, "output/" + filename, overwrite=True)
        obtain_value_distribution(filenames)
    elif real_dynamic:
        # анализ динамики средней степени соседей и её дисперсии в динамических реальныхсетях

        # создаются массив групп узлов average_degrees и массив дисперсий для групп deviations 
        # в каждой группе узлов несколько, например, десять, узлов
        # для каждого узла записывается средняя степень соседей на итерациях из массива dynamic_iterations
        # для каждой группы узлов записываются дисперсии
        
        average_degrees = [] # Список списков из списков средних степеней для каждой группы
        deviations = [] # Список списков дисперсий
        coefs_variation = [] # Список списков коэффициентов вариации
        # Из диапазонов узлов заполняем список, состоящий из групп узлов 
        for focus_range in dynamic_focus_nodes_ranges:
            focus_nodes = list(range(focus_range[0], focus_range[1] + 1))
            dynamic_focus_nodes.append(focus_nodes)

        for node_group in dynamic_focus_nodes:
            deviations.append(list())
            coefs_variation.append(list())
            average_degrees_group = []
            for _ in node_group:
                average_degrees_group.append(list())
            average_degrees.append(average_degrees_group)

        # Список итераций на которых вычисляются и записываются значения величин 
        dynamic_iters = dynamic_iterations.copy()

        graph = nx.Graph(create_using = graph_type)
        last_time_stamp = 0
        with open(filename, 'r') as f:
            for line in f:
                if line.startswith('%'):
                    continue
                # Если ошибка в данной строке, поменяйте количество переменных, чтобы соответствовать схеме входных данных
                node_from, node_to, _, timestamp = [int(x) for x in line.split(' ')]

                if timestamp < last_time_stamp:
                    raise Exception(f"Network is not time-sorted. Previous time stamp: {last_time_stamp}. New: {timestamp}")
                else:
                    last_time_stamp = timestamp
                
                graph.add_edge(node_from, node_to)
                if len(dynamic_iters) == 0:
                    print(f"Network growth finished at {len(graph.nodes)}. Maybe add more dynamic focus iters?")
                    break 
                if len(graph.nodes) > dynamic_iters[0]:
                    dynamic_iters = dynamic_iters[1:]
                    for i in range(len(dynamic_focus_nodes)):
                        for j in range(len(dynamic_focus_nodes[i])):
                            average_degrees[i][j].append(get_neighbor_average_degree(graph, dynamic_focus_nodes[i][j], directed=real_directed))
                    for i in range(len(average_degrees)): # Для каждой группы узлов
                        #print("Average degrees / Last average degrees")
                        last_average_degrees = [x[-1] for x in average_degrees[i]] # собираем список последних средних степеней каждой вершины в группе
                        #print(average_degrees[i], last_average_degrees)

                        # получаем суммарную и среднюю - среднюю степень соседей
                        sum_average_degrees = sum(last_average_degrees)
                        average_average_degrees = sum_average_degrees / len(last_average_degrees)
                        
                        # вычисляем дисперсию
                        variance = 0
                        for average_degree in last_average_degrees:
                            variance += math.pow(average_average_degrees - average_degree, 2)
                        variance /= len(last_average_degrees)
                        
                        # вычисляем стандартное отклонение
                        deviation = math.sqrt(variance) 
                        deviations[i].append(deviation)
                        coefs_variation[i].append(deviation / average_average_degrees)
                    print(len(graph.nodes), graph.number_of_edges())

            # print(average_degrees)
            # print(deviations)
            print(len(graph.nodes), graph.number_of_edges())
            # для графиков выбираются только те итерации, номер которых меньше или равен размеру сети  
            dynamic_its_for_plot = dynamic_iterations.copy()
            dynamic_its_for_plot = dynamic_its_for_plot[0:len(dynamic_iterations) - len(dynamic_iters)] 
            group_0_avgs = zip(*average_degrees[0])
            group_1_avgs = zip(*average_degrees[1])
            group_2_avgs = zip(*average_degrees[2])
            plt.plot( dynamic_its_for_plot, [sum(x) / len(x) for x in group_0_avgs] 
                    , dynamic_its_for_plot, [sum(x) / len(x) for x in group_1_avgs]
                    , dynamic_its_for_plot, [sum(x) / len(x) for x in group_2_avgs]
                    )
            plt.title(f"Average alpha for nodes from {filename}")
            plt.legend([dynamic_focus_nodes_ranges[0], dynamic_focus_nodes_ranges[1], dynamic_focus_nodes_ranges[2]])
            plt.xlabel("t")
            plt.ylabel("E(alpha)")
            plt.show()
            plt.title(f"Standart deviation of alpha for nodes from {filename}")
            plt.xlabel("t")
            plt.ylabel("std(alpha)")
            plt.plot(dynamic_its_for_plot, deviations[0], dynamic_its_for_plot, deviations[1], dynamic_its_for_plot, deviations[2])
            plt.legend([dynamic_focus_nodes_ranges[0], dynamic_focus_nodes_ranges[1], dynamic_focus_nodes_ranges[2]])
            plt.show()
            plt.title(f"Variation coef. of alpha for nodes from {filename}")
            plt.xlabel("t")
            plt.ylabel("cv(alpha)")
            plt.plot(dynamic_its_for_plot, coefs_variation[0], dynamic_its_for_plot, coefs_variation[1], dynamic_its_for_plot, coefs_variation[2])
            plt.legend([dynamic_focus_nodes_ranges[0], dynamic_focus_nodes_ranges[1], dynamic_focus_nodes_ranges[2]])
            plt.show()


# 1 Barabasi-Albert
def create_ba(n, m, focus_indices, focus_period):
    G = nx.complete_graph(m)

    # сохраняет динамику для узлов
    s_a_b_focus = []
    for focus_ind in focus_indices:
        s_a_b_focus.append(([], [], []))

    for k in range(m, n + 1):
        deg = dict(G.degree)  
        
        vertex = list(deg.keys()) 
        degrees = list(deg.values())

        G.add_node(k) 

        # предпочтительное присоединение 
        v_count = len(vertex)
        for _ in range(m):
            [node_to_connect] = random.choices(range(v_count), weights=degrees)
            G.add_edge(k, node_to_connect)
            del(vertex[node_to_connect])
            del(degrees[node_to_connect])
            v_count -= 1      

        # сохранить динамику для отслеживаемых узлов 
        if k % focus_period == 0:
            update_s_a_b(G, focus_indices, s_a_b_focus, k)

        progress_bar_update_period = 50
        if k % progress_bar_update_period == 0 and progress_bar is not None:
            progress_bar['value'] += 100 * (1 / number_of_experiments / n * progress_bar_update_period)
            progress_bar.master.master.update()


    if not save_data and len(focus_indices) > 0:
        plot_s_a_b(s_a_b_focus)

    return (G, s_a_b_focus)


def experiment_ba():
    filename = f"output/out_ba_{n}_{m}"

    start_time = time.time()
    now = datetime.now()
    if save_data:
        files = init_focus_indices_files(filename)
        filenames_analyze_value = []
        for _ in range(number_of_experiments):
            graph, result = create_ba(n, m, focus_indices, focus_period)
            filenames_analyze_value = process_simulated_network(graph, result, files, filename)
            print("Elapsed time: ", round(time.time() - start_time, 2))
        print("Finished")
        process_dynamics.process_s_a_b_dynamics(files)
        print("Analyzing values from files:", filenames_analyze_value)
        obtain_value_distribution(filenames_analyze_value)
    else:
        graph, result = create_ba(n, m, focus_indices, focus_period)
        analyze_mult_val_graph(graph, "output/test.txt")
        

# 2 Тройственное замыкание
def create_triadic(n, m, p, focus_indices, focus_period):
    G = nx.complete_graph(m)

    s_a_b_focus = []
    for focus_ind in focus_indices:
        s_a_b_focus.append(([], [], []))

    # k - индекс добавляемой вершины
    for k in range(m, n + 1):
        deg = dict(G.degree)  
        G.add_node(k) 
          
        vertex = list(deg.keys()) 
        weights = list(deg.values())
            
        [j] = random.choices(range(0, k), weights) # выбрать первый узел
        j1 = vertex[j]
        del vertex[j]
        del weights[j]

        lenP1 = k - 1  # длина списка узлов

        vertex1 = G[j1]
        lenP2 = len(vertex1)
        
        numEdj = m - 1  # количество дополнительных ребер

        if numEdj > lenP1: # не больше чем размер графа
            numEdj = lenP1

        randNums = np.random.rand(numEdj)   # список случайных чисел
        neibCount = np.count_nonzero(randNums <= p) # кол-во элементов меньше или равно p
          # что равняется количество узлов, смежных с j, которые должны быть присоединены к k
        if neibCount > lenP2 :   # не более чем кол-во соседей j
            neibCount = lenP2  
        vertCount = numEdj - neibCount  # кол-во других узлов для присоединения к узлу k

        neighbours = random.sample(list(vertex1), neibCount) # список вершин из соседних
        
        G.add_edge(j1, k)

        for i in neighbours:
            G.add_edge(i, k)
            j = vertex.index(i) # индекс i в списке всех узлов
            del vertex[j]    # удалить i и его вес из списков
            del weights [j]
            lenP1 -= 1

        for _ in range(0, vertCount):
            [i] = random.choices(range(0, lenP1), weights)
            G.add_edge(vertex[i], k)
            del vertex[i]
            del weights[i]
            lenP1 -= 1


        # сохранить статистику отслеживаемых узлов
        if k % focus_period == 0:
            update_s_a_b(G, focus_indices, s_a_b_focus, k)

        progress_bar_update_period = 50
        if k % progress_bar_update_period == 0 and progress_bar is not None:
            progress_bar['value'] += 100 * (1 / number_of_experiments / n * progress_bar_update_period)
            progress_bar.master.master.update()


    if not save_data and len(focus_indices) > 0:
        plot_s_a_b(s_a_b_focus)

    return (G, s_a_b_focus)


def experiment_triadic():
    filename = f"output/out_tri_{n}_{m}_{p}"

    if save_data:
        start_time = time.time()

        files = init_focus_indices_files(filename)
        filenames_analyze_value = []
        for _ in range(number_of_experiments):
            graph, result = create_triadic(n, m, p, focus_indices, focus_period)
            filenames_analyze_value = process_simulated_network(graph, result, files, filename)
            print("Elapsed time: ", round(time.time() - start_time, 2))
        print("Finished")
        process_dynamics.process_s_a_b_dynamics(files)
        obtain_value_distribution(filenames_analyze_value)
    else:
        graph, result = create_triadic(n, m, p, focus_indices, focus_period)
        analyze_mult_val_graph(graph, "output/test.txt")
        
    

# 3 Test data
def print_node_values(graph, node_i):
    print("Summary degree of neighbors of node %s (si) is %s" % (node_i, get_neighbor_summary_degree(graph, node_i)))
    print("Average degree of neighbors of node %s (ai) is %s" % (node_i, get_neighbor_average_degree(graph, node_i)))
    print("Friendship index of node %s (bi) is %s" % (node_i, get_friendship_index(graph, node_i)))


def experiment_test():
    filename = "test_graph.txt"

    graph = nx.read_edgelist(filename)
    print_node_values(graph, '1')

    analyze_mult_val_graph(graph, "output/test_out.txt")
    
    nx.draw(graph, with_labels=True)
    plt.title("Тестовый граф (см. консоль для доп. информации)")
    plt.show()


def run_external(**params):
    global experiment_type_num, number_of_experiments, n, m, p, focus_indices
    global focus_period, save_data, value_to_analyze, values_to_analyze, value_log_binning
    global progress_bar
    global filename, real_directed

    experiment_type_num = params.get('experiment_type_num', 1)
    
    number_of_experiments = params.get('number_of_experiments', 1)
    n = params.get('n', 100)
    m = params.get('m', 1)
    p = params.get('p', 1)
    focus_indices = params.get('focus_indices', [])
    focus_period = params.get('focus_period', 50)
    save_data = params.get('save_data', False)
    
    value_to_analyze = params.get('value_to_analyze', NONE)
    values_to_analyze = params.get('values_to_analyze', list())
    value_log_binning = params.get('value_log_binning', False)

    progress_bar = params.get('progress_bar', None)
    if progress_bar is not None:
        progress_bar['value'] = 0

    filename = params.get('filename', 'default-filename.txt')
    real_directed = params.get('real_directed', False)

    if False:
        threading.Thread(target=run_internal).start() 
    else:
        run_internal()

def run_internal():
    input_type = experiment_types[experiment_type_num]
    print("Doing %s experiment" % input_type)
    if input_type == "from_file":
        experiment_file()
    elif input_type == "barabasi-albert":
        experiment_ba()
    elif input_type == "triadic":
        experiment_triadic()
    elif input_type == "test":
        experiment_test()


if __name__ == "__main__":
    run_internal()
