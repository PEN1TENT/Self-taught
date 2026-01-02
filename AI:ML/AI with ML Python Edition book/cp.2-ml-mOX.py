import random
import numpy as np

O = []
X = []


def checkWin(P):
    if not P:
        return False
    last = P[-1]
    r, c = last // 3, last % 3

    row_count = sum(1 for m in P if m // 3 == r)
    col_count = sum(1 for m in P if m % 3 == c)
    diag1 = sum(1 for m in P if m in [0, 4, 8]) if last in [0, 4, 8] else 0
    diag2 = sum(1 for m in P if m in [2, 4, 6]) if last in [2, 4, 6] else 0

    return 3 in (row_count, col_count, diag1, diag2)


def get_lines():
    rows = [set(range(i * 3, (i + 1) * 3)) for i in range(3)]
    cols = [set(range(i, 9, 3)) for i in range(3)]
    diags = [{0, 4, 8}, {2, 4, 6}]
    return rows + cols + diags


def evalOX(O_set, X_set):
    SO = SX = 0
    critical = []

    for line in get_lines():
        o_in_line = line.intersection(O_set)
        x_in_line = line.intersection(X_set)

        if not x_in_line:
            SO += len(o_in_line)
            if len(o_in_line) == 2:
                critical.extend(list(line - o_in_line))

        if not o_in_line:
            SX += len(x_in_line)

    return 1 + SX - SO, critical


def AI():
    valid = list(set(range(9)) - set(O + X))
    V = {m: -100 for m in valid}

    for m in valid:
        score, critical = evalOX(set(O), set(X + [m]))
        if critical:
            move = [c for c in critical if c in valid]
            if move:
                return random.choice(move)
        V[m] = score

    max_val = max(V.values())
    best_moves = [m for m, v in V.items() if v == max_val]
    return random.choice(best_moves)


def displayOX():
    board = np.array([" "] * 9)
    for i in O:
        board[i] = "O"
    for i in X:
        board[i] = "X"
    print(board.reshape(3, 3), "\n")


while True:
    try:
        move = int(input("Choose [1-9]: ")) - 1
        if move not in range(9) or move in O + X:
            print("Invalid move.")
            continue
    except ValueError:
        continue

    O.append(move)
    displayOX()
    if checkWin(O):
        print("O wins!")
        break
    if len(O) + len(X) == 9:
        print("Draw")
        break
    X.append(AI())
    displayOX()
    if checkWin(X):
        print("X wins!")
        break
