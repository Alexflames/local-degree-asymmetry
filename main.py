import networkx as nx
import matplotlib.pyplot as plt
import random
import numpy as np
import time
import math
from bisect import bisect_left
from datetime import datetime
from sklearn.linear_model import LinearRegression

### To run the experiments
# a) Change input_type_num variable below
# b1) For real graph change filename in experiment_file() function
# b2) For BA model change parameters in experiment_ba() function
# b3) For TC model change parameters in experiment_triadic() function
# focus_indices array allows to record trajectory of nodes with selected indices, i.e [10, 50, 100, 1000]
# To average results go to process_output.py
# To calculate ratio of friendship paradox go to analyze_hist.py
# hist_ files contain histograms on linear and log-log scale as well as linreg approximation
# out_ files contain raw results for nodes in focus_indices array
# please, do not rename output files for further processing to avoid errors

# Works on Python 3.7.6
### Full instructions in Readme.md


input_types = ["from_file", "barabasi-albert", "triadic", "test"]
# Change value below to run experiment
input_type_num = 2

def get_neighbor_summary_degree(graph, node):
    neighbors_of_node = graph.neighbors(node)
    return sum(graph.degree(neighbor) for neighbor in neighbors_of_node)


def get_neighbor_average_degree(graph, node, si=None):
    if not si:
        si = get_neighbor_summary_degree(graph, node)
    return si / graph.degree(node)


def get_friendship_index(graph, node, ai=None):
    if not ai:
        ai = get_neighbor_average_degree(graph, node)
    return ai / graph.degree(node)


# Acquires histograms for friendship index 
def analyze_fi_graph(graph, filename):
    graph_nodes = graph.nodes()

    # b (beta) = friendship index 
    maxb = 0
    bs = []
    # get all values of friendship index
    for node in graph_nodes:
        new_b = get_friendship_index(graph, node)
        if new_b > maxb:
            maxb = new_b
        bs.append(new_b)
    
    # n=values, bins=edges of bins
    n, bins, _ = plt.hist(bs, bins=range(int(maxb)), rwidth=0.85)

    # leave only non-zero
    n_bins = zip(n, bins)
    n_bins = list(filter(lambda x: x[0] > 0, n_bins))
    n, bins = [ a for (a,b) in n_bins ], [ b for (a,b) in n_bins ]
    
    # get log-log scale distribution
    lnt, lnb = [], []
    for i in range(len(bins) - 1):
        if (n[i] != 0):
            lnt.append(math.log(bins[i]+1))
            lnb.append(math.log(n[i]) if n[i] != 0 else 0)

    # prepare for linear regression
    np_lnt = np.array(lnt).reshape(-1, 1)
    np_lnb = np.array(lnb)

    # linear regression to get power law exponent
    model = LinearRegression()
    model.fit(np_lnt, np_lnb)
    linreg_predict = model.predict(np_lnt)

    [directory, filename] = filename.split('/')
    with open(directory + "/hist_" + filename, "w") as f:
        f.write("t\tb\tlnt\tlnb\tlinreg\t k=" + str(model.coef_) + ", b=" + str(model.intercept_) + "\n")

        for i in range(len(lnb)):
            f.write(str(bins[i]) + "\t" + str(int(n[i])) + "\t" + str(lnt[i]) + "\t" + str(lnb[i]) + "\t" + str(linreg_predict[i]) + "\n")


# 0 - From file
def experiment_file():
    filename = "phonecalls.edgelist.txt"
    graph = nx.read_edgelist(filename)
    analyze_fi_graph(graph, filename)
    

# 1 Barabasi-Albert
def create_ba(n, m, focus_indices):
    G = nx.complete_graph(m)

    # get node statistics
    s_a_b_focus = []
    for focus_ind in focus_indices:
        s_a_b_focus.append(([], [], []))

    for k in range(m, n + 1):
        deg = dict(G.degree)  
        G.add_node(k) 
          
        vertex = list(deg.keys()) 
        weights = list(deg.values())

        # preferential attachment 
        nodes_to_connect = random.choices(vertex, weights, k=m)        
        for node in nodes_to_connect: # TODO: same node twice
            G.add_edge(k, node)

        # save focus node statistics
        if k % 50 == 0:
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


    should_plot = False
    if should_plot:
        s_a_b = s_a_b_focus[0]
        s_focus_xrange = [x / len(s_a_b[0]) for x in range(len(s_a_b[0]))]
        plt.plot(s_focus_xrange, s_a_b[0])
        plt.show()
        s_focus_xrange = [x / len(s_a_b[1]) for x in range(len(s_a_b[1]))]
        plt.plot(s_focus_xrange, s_a_b[1])
        plt.show()
        s_focus_xrange = [x / len(s_a_b[2]) for x in range(len(s_a_b[2]))]
        plt.plot(s_focus_xrange, s_a_b[2])
        plt.show()

    #print(G.degree)
    return (G, s_a_b_focus)


def experiment_ba():
    ### Change these parameters ###
    n = 10000
    m = 3
    number_of_experiments = 3
    focus_indices = [50, 100]
    ###  
    filename = f"output/out_ba_{n}_{m}"

    start_time = time.time()
    now = datetime.now()
    should_write = True
    if should_write:
        files = []
        for ind in focus_indices:
            f_s = open(f"{filename}_{ind}_s.txt", "a")
            f_a = open(f"{filename}_{ind}_a.txt", "a")
            f_b = open(f"{filename}_{ind}_b.txt", "a")
            files.append((f_s, f_a, f_b))
        now = datetime.now()
        for i in range(len(focus_indices)):
            for f in files[i]:
                f.write("> n=" + str(n) + " m=" + str(m) + " " + now.strftime("%d/%m/%Y %H:%M:%S") + "\n")
        for _ in range(number_of_experiments):
            graph, result = create_ba(n, m, focus_indices)
            for i in range(len(focus_indices)):
                for j in range(len(result[i])):
                    files[i][j].write(" ".join(str(x) for x in result[i][j]) + "\n")
            analyze_fi_graph(graph, filename + ".txt")
    else:
        graph, result = create_ba(n, m, focus_indices)
        #analyze_fi_graph(graph, "test.txt")
    print(("Elapsed time: %s", time.time() - start_time))
        

# 2 Triadic Closure
def create_triadic(n, m, p, focus_indices):
    G = nx.complete_graph(m)

    s_a_b_focus = []
    for focus_ind in focus_indices:
        s_a_b_focus.append(([], [], []))

    # k - index of added node
    for k in range(m, n + 1):
        deg = dict(G.degree)  
        G.add_node(k) 
          
        vertex = list(deg.keys()) 
        weights = list(deg.values())
            
        [j] = random.choices(range(0, k), weights) # choose first node
        j1 = vertex[j]
        del vertex[j]
        del weights[j]

        lenP1 = k - 1  # length of list of vertices 

        vertex1 = G[j1]
        lenP2 = len(vertex1)
        
        numEdj = m - 1  # number of additional edges

        if numEdj > lenP1: # not more than size of the graph
            numEdj = lenP1

        randNums = np.random.rand(numEdj)   # list of random numbers
        neibCount = np.count_nonzero(randNums <= p) # number of elements less or equal than p
          # which is equal to the number of nodes adjacent to j, which should be connected to k
        if neibCount > lenP2 :   # not more than neighbors of j
            neibCount = lenP2  
        vertCount = numEdj - neibCount  # number of arbitrary nodes of the graph to connect with k

        neibours = random.sample(list(vertex1), neibCount) # список вершин из соседних
        
        G.add_edge(j1, k)

        for i in neibours:
            G.add_edge(i, k)
            j = vertex.index(i) # index of i in the list of all vertices
            del vertex[j]    # delete i and its weight from lists
            del weights [j]
            lenP1 -= 1

        for _ in range(0, vertCount):
            [i] = random.choices(range(0, lenP1), weights)
            G.add_edge(vertex[i], k)
            del vertex[i]
            del weights[i]
            lenP1 -= 1


        # save focus node statistics
        if k % 50 == 0:
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


    should_plot = False
    if should_plot:
        s_a_b = s_a_b_focus[0]
        s_focus_xrange = [x / len(s_a_b[0]) for x in range(len(s_a_b[0]))]
        plt.plot(s_focus_xrange, s_a_b[0])
        plt.show()
        s_focus_xrange = [x / len(s_a_b[1]) for x in range(len(s_a_b[1]))]
        plt.plot(s_focus_xrange, s_a_b[1])
        plt.show()
        s_focus_xrange = [x / len(s_a_b[2]) for x in range(len(s_a_b[2]))]
        plt.plot(s_focus_xrange, s_a_b[2])
        plt.show()

    return (G, s_a_b_focus)


def experiment_triadic():
    n = 10000
    m = 3
    p = 0.75
    number_of_experiments = 3
    focus_indices = [10, 50, 100]
    filename = f"output/out_tri_{n}_{m}_{p}"

    should_write = True
    if should_write:
        files = []
        for ind in focus_indices:
            f_s = open(f"{filename}_{ind}_s.txt", "a")
            f_a = open(f"{filename}_{ind}_a.txt", "a")
            f_b = open(f"{filename}_{ind}_b.txt", "a")
            files.append((f_s, f_a, f_b))
        now = datetime.now()
        start_time = time.time()
        for i in range(len(focus_indices)):
            for f in files[i]:
                f.write("> n=" + str(n) + " m=" + str(m) + " " + now.strftime("%d/%m/%Y %H:%M:%S") + "\n")
        for _ in range(number_of_experiments):
            graph, result = create_triadic(n, m, p, focus_indices)
            for i in range(len(focus_indices)):
                for j in range(len(result[i])):
                    files[i][j].write(" ".join(str(x) for x in result[i][j]) + "\n")
            analyze_fi_graph(graph, filename + ".txt")
        print(("Elapsed time: %s", time.time() - start_time))
    else:
        graph, result = create_triadic(n, m, p, focus_indices)
        analyze_fi_graph(graph, "test.txt")
    

# 3 Test data
def print_node_values(graph, node_i):
    print("Summary degree of neighbors of node %s (si) is %s" % (node_i, get_neighbor_summary_degree(graph, node_i)))
    print("Average degree of neighbors of node %s (ai) is %s" % (node_i, get_neighbor_average_degree(graph, node_i)))
    print("Friendship index of node %s (bi) is %s" % (node_i, get_friendship_index(graph, node_i)))


def experiment_test():
    filename = "test_graph.txt"

    graph = nx.read_edgelist(filename)
    print_node_values(graph, '1')
    
    nx.draw(graph, with_labels=True)
    plt.show()


if __name__ == "__main__":
    input_type = input_types[input_type_num]
    print("Doing %s experiment" % input_type)
    if input_type == "from_file":
        experiment_file()
    elif input_type == "barabasi-albert":
        experiment_ba()
    elif input_type == "triadic":
        experiment_triadic()
    elif input_type == "test":
        experiment_test()
