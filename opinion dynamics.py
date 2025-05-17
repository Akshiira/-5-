import numpy as np
import matplotlib.pyplot as plt
INF = 1000000

d = 2 #размерность
n = 100 #число агентов
n_stubb = 0 #число упрямых агентов
t = INF #число итераций
p = 2
h = 0.5 #коэффициент инерции
pressure = 0.05 #коэффициент давления
n_pub = n #число агентов, участующих в формировании публичного мнения
eps = 0.01
eps_pow = eps**p
threshold = 1/(n * 100.0)
triangle = np.zeros((3, 2))

def Lp_norm_pow(x, y):
    return ((np.abs(x - y))**p).sum()

def Lp_neighbor(x, y):
    if Lp_norm_pow(x, y) <= eps_pow:
        return 1
    return 0

def L1_norm(x, y):
    return np.abs(x - y).sum()

def L1_neighbor(x, y):
    if L1_norm(x, y) <= eps:
        return 1
    return 0

def Linf_neighbor(x, y):
    for i in range(d):
        if abs(x[i] - y[i]) >= eps:
            return 0
    return 1

def Linf_norm(x, y):
    res = -1
    for i in range(d):
        res = max(abs(x[i] - y[i]), res)
    return res

def min_neighbor(x, y):
    for i in range(d):
        if abs(x[i] - y[i]) <= eps:
            return 1
    return 0

def min_dist(x, y):
    res = INF
    for i in range(d):
        res = min(res, abs(x[i] - y[i]))
    return res

def triangle_neighbor(x, y): # y сосед для x
    p = y - x
    if (np.cross(triangle[0] - p, triangle[1] - triangle[0]) > 0 or
        np.cross(triangle[1] - p, triangle[2] - triangle[1]) > 0 or
        np.cross(triangle[2] - p, triangle[0] - triangle[2]) > 0):
        return 0
    return 1

from numpy.linalg import norm
def cos_dist(x, y):
    return np.dot(x, y) / (norm(x) * norm(y))

def cos_neighbor(x, y): #угол меньше 90
    if cos_dist(x, y) >= 1 - eps:
        return 1
    return 0

color = np.random.rand(n, 3)

def traj_seg(graph):
    ax = plt.figure(figsize=(10, 10), constrained_layout=True).add_subplot(projection='3d')
    for i in range(n):
        ax.plot(graph[:t, i, 0], graph[:t, i, 1], np.arange(t), color=color[i], marker='.', linewidth=1)
    #ax.set_xlabel('X1(t)', fontsize=16, labelpad=16)
    #ax.set_ylabel('X2(t)', fontsize=16, labelpad=16)
    ax.set_zlabel('t', fontsize=12, labelpad=0)
    ax.set_zticks(np.arange(0, t, 1)) #регулируется разметка по оси z
    plt.tick_params(axis='both', which='major', labelsize=12)
    plt.show()

def proj_seg(graph, axis):
    for i in range(n):
        plt.plot(np.arange(t), graph[:t,i,axis], marker='.', linewidth=1)
    if axis == 0:
        plt.text(7.4, 0.9, r"$\xi_{1}(t)$", fontsize=24)
    if axis == 1:
        plt.text(7.4, 0.9, r"$\xi_{2}(t)$", fontsize=24)

    plt.tick_params(axis='both', which='major', labelsize=24)
    plt.ylim(-1.1, 1.1)
    plt.show()

import random
def k_random_agents(k): #генерирует k неповторяющихся рандомных агентов из n
    random.seed(9001)  # главное его не трогать :)
    random_set = set()
    while len(random_set) < k:
        random_set.add(random.randint(0, n-1))
    return random_set
def form_pub_opinion(n_pub, x): #формируем публичное мнение из n_pub случайных агентов
    pub_sum = 0
    if n_pub == n: #все агенты участвуют в формировании публичного мнения с одинаковым весом
        for i in range(n):
            pub_sum += x[i]
        return pub_sum/n
    random_agents = k_random_agents(n_pub)
    for i in random_agents:
        pub_sum += x[i]
    return pub_sum/n_pub

def multi_hk_model(x, neighbor_func):
    w = np.zeros((n, d))
    adj_matrix = np.zeros((n, n))  #матрица смежности
    neigh_sum = np.zeros((n, d))
    neigh_count = np.zeros(n)
    pub_opinion = form_pub_opinion(n_pub, x)

    def add_neighbor(i, j): #у агента i новый сосед j
        neigh_sum[i] += x[j]
        neigh_count[i] += 1
        adj_matrix[i][j] = 1
        return

    for i in range(n):
        add_neighbor(i, i)
        for j in range(i + 1, n):
            if i >= n - n_stubb: #упрямым агентам соседей не ищем
                break
            if neighbor_func(x[i], x[j]):
                add_neighbor(i, j)
            if neighbor_func(x[j], x[i]):
                if j >= n - n_stubb: #упрямый себе соседей не ищет
                    break
                add_neighbor(j, i)
        w[i] = (neigh_sum[i]/neigh_count[i]) * (1 - pressure) + pub_opinion * pressure
    return w, adj_matrix


def multi_inertia_hk_model(x, neighbor_func):
    w = np.zeros((n, d))
    adj_matrix = np.zeros((n, n))  #матрица смежности
    neigh_sum = np.zeros((n, d))
    neigh_count = np.zeros(n)

    def add_neighbor(i, j): #у агента i новый сосед j
        neigh_sum[i] += x[j]
        neigh_count[i] += 1
        adj_matrix[i][j] = 1
        return

    for i in range(n):
        add_neighbor(i, i)
        for j in range(i + 1, n):
            if neighbor_func(x[i], x[j]):
                add_neighbor(i, j)
            if neighbor_func(x[j], x[i]):
                add_neighbor(j, i)
        w[i] = (1-h) * x[i] + h * neigh_sum[i]/neigh_count[i]
    return w, adj_matrix

cluster_rep = []
def extract_clusters(adj_matrix):
    global cluster_rep
    cluster_rep = []

    is_equilibrium = 1
    used_in_cluster = np.zeros(n) #был ли задействован уже агент i в каком-то кластере

    for i in range(n):
        if used_in_cluster[i]:
            continue
        used_in_cluster[i] = 1
        cluster_rep.append(i)
        for j in range(n):
            if adj_matrix[i][j]:
                if (adj_matrix[i] == adj_matrix[j]).all():
                    used_in_cluster[j] = 1
                else:
                    is_equilibrium = 0
                    break
        if is_equilibrium == 0:
            break

    return is_equilibrium

def write_data_to_file(data, file):
    for row in data:
        for i in range(len(row)):
            if i < len(row) - 1:
                file.write(str(row[i]) + ' ')
            else:
                file.write(str(row[i]) + '\n')
def res_clusters(x):
    data = []
    for i in cluster_rep:
        data.append(x[i])
    res_corpus_file = open("res_corpus", "w")
    write_data_to_file(data, res_corpus_file)
    res_corpus_file.close()

def get_graph(neighbor_func, x):
    prev_x = np.random.sample((n, d))
    prev_adj_matrix = np.zeros((n, n))
    global t
    graph = np.zeros((t, n, d))
    graph[0] = x
    for i in range(1, t):
        if i > 100:
            t = i
            break

        y, x_adj_matrix = multi_hk_model(x, neighbor_func)#multi_inertia_hk_model(x, neighbor_func)
        graph[i] = y
        if (x_adj_matrix == prev_adj_matrix).all():
            if (np.abs(x - prev_x) < threshold).all():
                is_equilibrium = extract_clusters(x_adj_matrix)
                if is_equilibrium:
                    t = i
                    break

        prev_x = x
        x = y
        prev_adj_matrix = x_adj_matrix

    res_clusters(graph[t-1])
    traj_seg(graph)
    #proj_seg(graph, axis=0)
    #proj_seg(graph, axis=1)
    return x

def make_triangle(eps):
    triangle[0] = np.array([-1 / 2.0, -np.sqrt(3.0) / 2]) * eps
    triangle[1] = np.array([-1 / 2.0, np.sqrt(3.0) / 2]) * eps
    triangle[2] = np.array([1.0, 0.0]) * eps

def read_file(file):
    data = []
    for line in file:
        data.append([float(x) for x in line.split()])
    return np.array(data)

def make_equal_stubb(x):
    for i in range(n - n_stubb, n):
        x[i] = [0,0]
    return x

def diam(x):
    max = -1
    for i in range(n):
        for j in range(n):
            dist = np.sqrt(Lp_norm_pow(x[i], x[j]))
            if dist > max:
                max = dist
    return max



import math

p = 2
neighbor_func = Lp_neighbor

np.random.seed(100)
x = 2 * np.random.sample((n, d)) - 1
t_exp = math.ceil(math.log(eps/diam(x), 1 - pressure)) + 1
make_equal_stubb(x)

x = get_graph(neighbor_func, x)

print("Real time:", t-2)
print("Expected time:", t_exp)














