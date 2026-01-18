# Word Guess by Genetic Algorithm
import numpy as np
from matplotlib import pyplot as plt 

def guess(target = 'ANT', n_iter = 100, n_sel = 50, n_xover = 50, p_mutate = 0.2):
    target = [ord(t) for t in target]
    code_size = len(target) 
    n_pop = n_sel + n_xover 
    data = np.arange(65, 91)
    pop = np.random.choice(data, (n_pop, code_size))
    Fitness = []
    min_fitness = 0
    i = 0
    while i <= n_iter and min_fitness < code_size:
        i += 1
        # selection 
        fitness = []
        for p in pop:
            fitness.append(np.sum(p == target))
        idx = np.argsort(fitness)[::-1]
        pop = pop[idx]
        Fitness.append(fitness[idx[0]])
        if Fitness[-1] > min_fitness:
            min_fitness = Fitness[-1]
            print('{} : {} [fitness = {}]'.format(i, ''.join([chr(p) for p in pop[0]]), min_fitness))
        # cross over 
        for k in range(n_sel, n_pop, 2):
            parent = np.random.randint(0, n_sel, 2)
            crosspoint = np.random.randint(0, code_size)
            offspring1 = np.concatenate((pop[parent[0]][:crosspoint], pop[parent[1]][crosspoint:]))
            offspring2 = np.concatenate((pop[parent[1]][:crosspoint], pop[parent[0]][crosspoint:]))
            pop[k] = offspring1
            pop[k + 1] = offspring2
        # mutation 
        for k in range(1, n_pop):
            if np.random.rand() < p_mutate:
                mutatepoint = np.random.randint(0, code_size)
                pop[k][mutatepoint] = np.random.choice(data)
        # remove duplicated data 
        _, idx = np.unique(pop, axis = 0, return_index = True)
        pop = pop[np.sort(idx)]
        if len(pop) < n_pop:
            newpop = np.random.choice(data, (n_pop - len(pop), code_size))
            pop = np.vstack((pop, newpop))
        
        # display 
        plt.clf()
        xticks = range(1, len(Fitness) + 1)
        plt.plot(xticks, Fitness)
        plt.xticks(xticks)
        plt.xlabel('Iterations')
        plt.ylabel('Fitness')
        plt.pause(1e-10)
    plt.show()
    return pop[0]
    
if __name__ == '__main__':
    guess('THAILAND')