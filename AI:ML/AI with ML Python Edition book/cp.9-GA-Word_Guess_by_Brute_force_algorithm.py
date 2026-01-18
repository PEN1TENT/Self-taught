# Word Guess by Brute force algorithm
import numpy as np 
from matplotlib import pyplot as plt 

def guess(target = 'ANT'):
    target = [ord(t) for t in target]
    code_size = len(target)
    data = np.arange(65, 91) # ASCII CODE (A-Z)
    i = 1
    idx_data = np.zeros(code_size, int) 
    fitness = []
    while True:
        letter_idx = code_size - 1
        pop = data[idx_data]
        fitness.append(np.sum(pop == target)) 
        print('%d : %s [fitness = %d]' % (i, ''.join([chr(p) for p in pop]), fitness[-1]))
        if fitness[-1] == code_size or letter_idx == -1:
            break
        i += 1
        idx_data[-1] += 1
        while idx_data[letter_idx] > len(data) - 1:
            idx_data[letter_idx] = 0
            letter_idx -= 1
            if letter_idx == 0:
                break
            idx_data[letter_idx] += 1
    plt.plot(fitness)
    plt.xlabel('Iterations')
    plt.ylabel('Fitness')
    plt.show()

if __name__ == '__main__':
    guess('THAILAND')