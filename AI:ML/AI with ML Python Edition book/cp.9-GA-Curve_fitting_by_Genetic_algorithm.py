# Curve fitting by Genetic Algorithm 
import numpy as np
from matplotlib import pyplot as plt 

def curve_fit(x, y, degree = 5, step = 5, error = 1e-2, n_iter = 1e2, n_sel = 50, n_xover = 50, p_mutate = 0.2):
    code_size = degree + 1
    n_pop = n_sel + n_xover 
    pop = np.random.rand(n_pop, code_size)
    Fitness = []
    min_fitness = 100 
    i = 0 
    while i <= n_iter and min_fitness > error:
        i += 1
        # selection 
        fitness = []
        for p in pop:
            z = np.polyval(p, x)
            if len(z) == 1:
                print(len(z), len(p), pop.shape) 
            mse = np.mean((z - y) ** 2) 
            fitness.append(mse)
        idx = np.argsort(fitness) 
        pop = pop[idx]
        Fitness.append(fitness[idx[0]])
        if Fitness[-1] < min_fitness:
            min_fitness = Fitness[-1]
            print('{} : {} [fitness = {}]'.format(i, pop[0], min_fitness))
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
                pop[k][mutatepoint] += np.random.rand() * step * (-1) **np.random.randint(2)
        # remove duplicated data 
        _, idx = np.unique(pop, axis = 0, return_index = True)
        pop = pop[np.sort(idx)]
        if len(pop) < n_pop:
            newpop = np.random.rand(n_pop -len(pop), code_size)
            pop = np.vstack((pop, newpop))
        # display
        plt.clf()
        xticks = range(1, len(Fitness) + 1)
        plt.plot(xticks, Fitness)
        #plt.xticks(xticks) 
        plt.xlabel('Iterations')
        plt.ylabel('Fitness')
        plt.pause(1e-10)
    plt.show()
    return pop[0]
    
if __name__ == '__main__':
    solution = [1, 4, 3, 2, 0, 1]
    N = 50
    x = np.random.rand(N)
    y = np.polyval(solution, x)
    
    soln = curve_fit(x, y, degree = 5)
    plt.figure()
    plt.plot(x, y, 'or', mfc = 'none')
    ax = plt.gca()
    xlim = ax.get_xlim()
    x = np.arange(xlim[0], xlim[1], 0.01)
    z = np.polyval(soln, x)
    plt.plot(x, z)
    plt.legend(['Actual', 'GA'])
    plt.show()
    
