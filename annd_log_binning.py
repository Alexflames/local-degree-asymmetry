# Фрагмент кода программы для моделирования и анализа сетей
# 
# Код вспомогательной функции  
# Возвращает два словаря:
# 1. Словарь, ключ - степень узла, значение - кортеж (сумма средних степеней соседей, количество узлов)    
# 2. Словарь, ключ - степень узла, значение - список средних степеней узлов с данной степенью 
# 
# Из полученных данных можно вычислить распределение ANND и дисперсий по степеням 
#
# Для корректной работы функции нужно определить функцию get_neighbor_average_degree(graph, node).
# Для получения визуализации результат функции подается в функцию visualize_deg_alpha_distribution.

from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
import math

# константы для функции получения данных для распределения
log_binning_base = 1.5
apply_log_binning = True

# Константы для функции визуализации результатов
log_value = True
visualization_size = 20
filename = "input_graph.txt"

# На вход:
#   graph - граф              
#   node - узел/номер узла
# На выход:
#   средняя степень соседей узла node
def get_neighbor_average_degree(graph, node):
    # .....
    pass 

# получение распределения ANND и дисперсии средней степени
# 
# Вход: граф
# Возвращает два отображения:
#   1. степень -> (сумма средних степеней, количество средних степеней)
#   2. степень -> [средняя_степень1, средняя_степень2, ...] (напр. для вычисления дисперсии)
# Имеется поддержка log-binning
def acquire_deg_alpha(graph):
    graph_nodes = graph.nodes()
    deg_alpha = dict()              # отображение { степень -> (сумма средних степеней, количество средних степеней) }
    deg_alphas = defaultdict(list)  # отображение { степень -> [средняя_степень1, средняя_степень2, ...] (напр. для вычисления дисперсии) }

    # данный блок кода получает сумму степеней и количество узлов с каждой степенью 
    for node in graph_nodes:
        degree = graph.degree(node)
        alpha = get_neighbor_average_degree(graph, node)
        deg_alpha_cur = deg_alpha.get(degree, (0, 0))
        deg_alpha[degree] = (deg_alpha_cur[0] + alpha, deg_alpha_cur[1] + 1)
        deg_alphas[degree].append(alpha) # список значений средней степени узлов для вычисления дисперсии
    
    max_degree = max(x[1] for x in graph.degree)

    if apply_log_binning:   # если 
        log_max = math.log(max_degree + 0.01, log_binning_base) # Увеличим на 0.01, чтобы это число было недостижимо  
        bins = np.logspace(0, log_max, num=math.ceil(log_max), base=log_binning_base) # последовательность степеней логарифма с основанием base
        bins = [round(bin, 3) for bin in bins] # для использования в качестве ключа словаря округлим вещественное число

        bin_deg_alpha = dict()
        bin_deg_alphas = defaultdict(list)

        degrees_s = sorted(deg_alpha.keys())    # отсортируем ключи словаря, чтобы степени шли по возрастанию
        k = 1   # индекс правой границы корзины log-binning
        for degree in degrees_s:
            if degree > bins[k]: # если текущая степень больше правой границы "корзины"            
                k += 1
            bin = (bins[k - 1] + bins[k]) / 2 # на графике точка будет в центре "корзины"
            bin_deg_alpha_cur = bin_deg_alpha.get(bin, (0, 0))
            bin_deg_alpha[bin] = ( bin_deg_alpha_cur[0] + deg_alpha[degree][0]   # алгоритм как в случае
                                 , bin_deg_alpha_cur[1] + deg_alpha[degree][1] ) # линейного биннинга
            bin_deg_alphas[bin] = bin_deg_alphas[bin] + deg_alphas[degree] # Конкатенация списков
        
        deg_alpha = bin_deg_alpha # результат функции - списки для каждой корзины "log-binning"
        deg_alphas = bin_deg_alphas 

        print("Log binning bins:", bins, sep="\n")
        print(bin_deg_alpha)

    return deg_alpha, deg_alphas


# Стандартный код визуализации распределения на log-log графике. 
# На вход подавать результат выполнения функции acquire_deg_alpha
#
# deg_alpha - отображение 
#   { степень -> ( сумма средних степеней узлов с заданной степенью
#                , количество узлов с заданной степенью )
#   }
# deg_alphas - отображение { степень -> [список средних степеней узлов с заданной степенью] }
def visualize_deg_alpha_distribution(deg_alpha, deg_alphas):
    degrees = deg_alpha.keys()
    alphas = []
    coefs_variation = []
    log_alphas = []
    log_degs = [math.log(deg, 10) for deg in degrees]
    for key in degrees:
        alpha = deg_alpha[key][0] / deg_alpha[key][1]
        deg_alpha[key] = (alpha, deg_alpha[key][1])
        alphas.append(alpha)
        log_alphas.append(math.log(alpha, 10))
    #plt.scatter(degrees, alphas, s = 3)
    if log_value:
        plt.scatter(log_degs, log_alphas, s = visualization_size)
        plt.xlabel("log10(k)")
        plt.ylabel("log ANND")
    else:
        plt.scatter(degrees, alphas, s = visualization_size)
        plt.xlabel("k")
        plt.ylabel("ANND")
    plt.title('Degree to ANND for ' + filename)

    plt.show()

    sigmas = []
    log_sigmas = []
    for degree in deg_alphas.keys():
        sigma2 = 0
        for alpha in deg_alphas[degree]:
            sigma2 += math.pow((alpha - deg_alpha[degree][0]), 2)
        sigma2 /= len(deg_alphas[key])
        sigma = math.sqrt(sigma2)
        sigmas.append(sigma2)
        log_sigmas.append(0 if sigma2 <= 0 else math.log(sigma2, 10))
        coefs_variation.append(sigma / deg_alpha[degree][0])

    #plt.scatter(degrees, sigmas, s = 3)
    if log_value:
        plt.scatter(log_degs, log_sigmas, s = visualization_size)
        plt.xlabel("log10(k)")
        plt.ylabel("log10(дисперсия)")
    else:
        plt.scatter(degrees, sigmas, s = visualization_size)
        plt.xlabel("k")
        plt.ylabel("дисперсия")
    plt.title('Deg. to avg deg. std for ' + filename)
    plt.show()

    plt.scatter(degrees, coefs_variation, s = visualization_size)
    plt.xlabel("k")
    plt.ylabel("коэф. вариации")
    plt.title('Deg. to CV for ' + filename)
    plt.show()