import numpy as np


def is_consistent(h, instance):
    """Returns True if hypothesis h matches the data instance."""
    for i in range(len(h)):
        if h[i] != "?" and h[i] != instance[i]:
            return False
    return True


def candidate_elimination(X, T):
    n = len(X[0])

    S = [list(X[0])]

    G = [["?" for _ in range(n)]]

    for i in range(len(X)):
        instance, label = X[i], T[i]

        if label == "Yes":
            G = [g for g in G if is_consistent(g, instance)]

            for s in S:
                for j in range(n):
                    if s[j] != instance[j]:
                        s[j] = "?"

        else:
            S = [s for s in S if not is_consistent(s, instance)]

            new_G = []
            for g in G:
                if not is_consistent(g, instance):
                    new_G.append(g)
                else:
                    for j in range(n):
                        if g[j] == "?":
                            for s in S:
                                if s[j] != "?" and s[j] != instance[j]:
                                    special_h = list(g)
                                    special_h[j] = s[j]
                                    if special_h not in new_G:
                                        new_G.append(special_h)
            G = new_G

    G = [g for g in G if any(is_consistent(g, s) for s in S)]

    return S, G


if __name__ == "__main__":
    X = [
        ["Sunny", "Warm", "Normal", "Strong", "Warm", "Same"],
        ["Sunny", "Warm", "High", "Strong", "Warm", "Same"],
        ["Rainy", "Cold", "High", "Strong", "Warm", "Change"],
        ["Sunny", "Warm", "High", "Strong", "Cool", "Change"],
    ]
    T = ["Yes", "Yes", "No", "Yes"]

    S_res, G_res = candidate_elimination(X, T)

    print("S Final:", S_res)
    print("G Final:", G_res)
