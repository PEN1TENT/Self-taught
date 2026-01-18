# Travelling Salesman Problem (TSP) by Genetic ALgorithm 
import numpy as np 
from matplotlib import pyplot as plt 

def distance(city, path):
    d = city[path] - city[np.roll(path, 1, axis = 0).astype(int)]
    return np.sum(np.sqrt(np.sum(d ** 2, axis = 1)))
    
def crossover(p1, p2):
    o1 = p1.copy()
    j = [p2[0]]
    s, t = np.array([p1[0]]), np.array([0])
    while not np.any(s == j):
        s = np.concatenate((s, j))
        i = np.where(p1 == j)[0]
        t = np.concatenate((t, i))
        j = p2[i]
        i = p1[i]
    idx = np.setdiff1d(np.arange(0, len(p1)), t)
    o1[idx] = p2[idx]
    o2 = np.max(o1) - o1
    return o1, o2
    
def mutate(p):
    pair = np.random.permutation(len(p))[:2]
    p[pair] = p[pair[::-1]]
    
def GA(city, n_iter = 500, n_sel = 50, n_xover = 50, p_mutate = 0.2):
    code_size = len(city)
    n_pop = n_sel + n_xover 
    pop = np.zeros((n_pop, code_size), int)
    for i in range(n_pop):
        pop[i] = np.random.permutation(code_size)
    n_delta = 0
    N_delta = 50 
    Fitness = []
    i = 0
    while i <= n_iter and n_delta <= N_delta:
        i += 1
        # selection 
        fitness = []
        for p in pop:
            fitness.append(distance(city, p))
        idx = np.argsort(fitness)
        pop = pop[idx]
        Fitness.append(fitness[idx[0]])
        if i == 1:
            min_fitness = Fitness[-1]
        elif Fitness[-1] < min_fitness:
            min_fitness = Fitness[-1]
            print('{} : {} [fitness = {}]'.format(i, pop[0], min_fitness))
            n_delta = 0
        else:
            n_delta += 1
        # cross over 
        for k in range(n_sel, n_pop, 2):
            parent = np.random.randint(0, n_sel, 2)
            pop[k], pop[k + 1] = crossover(pop[parent[0]], pop[parent[1]])
        # mutation 
        for k in range(1, n_pop):
            if np.random.rand() < p_mutate:
                mutate(pop[k])
        # remove duplivated data 
        _, idx = np.unique(pop, axis = 0, return_index = True)
        pop = pop[np.sort(idx)]
        if len(pop) < n_pop:
            newpop = np.zeros((n_pop - len(pop), code_size), int)
            for i in range(len(newpop)):
                newpop[i] = np.random.permutation(code_size)
            pop = np.vstack((pop, newpop))
        # display 
        plt.clf()
        plt.subplot(1, 2, 1)
        xticks = range(1, len(Fitness) + 1)
        plt.plot(xticks, Fitness)
        plt.xlabel('Iterations')
        plt.ylabel('Fitness')
        plt.subplot(1, 2, 2)
        idx = np.concatenate((pop[0], [pop[0, 0]]))
        plt.plot(city[idx, 0], city[idx, 1], ':o')
        plt.plot(city[idx[0], 0], city[idx[0], 1], 'or')
        plt.pause(1e-10)
    plt.show()
    return pop[0]
    
if __name__ == '__main__':
    N = 30
    city = np.random.rand(N, 2)
    soln = GA(city)
            