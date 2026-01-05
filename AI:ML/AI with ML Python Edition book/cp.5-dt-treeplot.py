# Treeplot
import numpy as np
from matplotlib import pyplot as plt
from pandas._libs.ops import vec_binop


def treeplot(
    tree, vlabel=[], elabel=[], show=True, fig_no=None, node_style="ob", edge_style="c"
):
    num_nodes = len(tree)
    num_children = np.zeros(num_nodes + 1, int)
    x_r = np.zeros(num_nodes, int)
    y = np.zeros(num_nodes, int)
    x_l = np.zeros(num_nodes, int)

    for i in range(num_nodes):
        num_children[tree[i]] += 1

    pos = 0
    start = np.zeros(num_nodes + 1, int)
    i_x = np.zeros(num_nodes + 1, int)
    stop = np.zeros(num_nodes + 1, int)
    for i in range(num_nodes + 1):
        start[i] = pos
        i_x[i] = pos
        pos += num_children[i]
        stop[i] = pos

    vec_of_child = np.zeros(num_nodes, int)
    for i in range(num_nodes):
        vec_of_child[i_x[tree[i]]] = i
        i_x[tree[i]] += 1

    parent_idx = -1
    left_most = 0
    depth = num_nodes
    min_depth = num_nodes

    t = np.array([[-2, 1]], int)
    
    max_iterations = num_nodes * 10  
    iterations = 0
    
    while parent_idx != -2 and iterations < max_iterations:
        iterations += 1
        if t.size == 0:
            break
        if start[parent_idx + 1] < stop[parent_idx + 1]:
            idx = vec_of_child[start[parent_idx + 1] : stop[parent_idx + 1]]
            temp = np.vstack((idx, np.full(idx.shape, parent_idx))).T
            t = np.vstack((t, temp))
        if t[-1, -1] != parent_idx:
            left_most += 1
            x_r[parent_idx] = left_most
            min_depth = min([min_depth, depth])
            if len(t) > 1:
                temp = np.roll(t, 1, axis=0) - t
                if np.any(temp.flatten()[1:] == 0) and t[-1, -1] != t[-2, -1]:
                    position = np.where(temp[:, -1] == 0)[0][-1] + 1
                    parent_idx_vec = t[position:, -1]
                    depth += len(parent_idx_vec)
                    x_r[parent_idx_vec] = left_most
                    t = t[:position, :]

            t = t[:-1, :]
            if t.size > 0:
                parent_idx = t[-1, 0]
            else:
                parent_idx = -2
            if parent_idx != -2:
                y[parent_idx] = depth
                x_l[parent_idx] = left_most + 1
        else:
            depth -= 1
            parent_idx = t[-1, 0]
            y[parent_idx] = depth
            x_l[parent_idx] = left_most + 1

    x = (x_l + x_r) / -2  # - for flip left-right

    # Plot tree
    if fig_no is None:
        plt.figure()
    else:
        plt.figure(fig_no)
    plt.axis("off")

    # Plot nodes
    plt.plot(x, y, node_style)
    # Plot node's values
    for i, t in enumerate(vlabel):
        plt.text(x[i], y[i], t)

    # Plot edges
    for i in range(len(tree)):
        if tree[i] > 0:
            j = tree[i] - 1
            X = [x[i], x[j]]
            Y = [y[i], y[j]]
            plt.plot(X, Y, edge_style)
            if len(elabel) == len(tree) - 1:
                plt.text(
                    (X[0] + X[1]) / 2, (Y[0] + Y[1]) / 2, elabel[i - 1]
                )  # Plot edge's values

    if show:
        plt.show()

    return x, y
