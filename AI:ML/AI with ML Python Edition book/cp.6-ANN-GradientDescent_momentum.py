# Gradient Descent with Momentum
import numpy as np
from matplotlib import pyplot as plt


def grad_desc_momentum(x0, f, df, lr=1e-2, eps=1e-2, alpha=1e-2, show=True):
    xOld = x0
    xNew = xOld + 2 * eps
    dw = 0
    i = 0
    while abs(xNew - xOld) > eps:
        xOld = xNew
        dw = lr * df(xNew) + alpha * dw
        xNew -= dw
        if show:
            plt.plot(xNew, f(xNew), ".r")
            plt.text(xNew, f(xNew), str(i))
            plt.draw()
            plt.pause(0.2)
            print(i)
            i += 1
    return xNew


if __name__ == "__main__":

    def f(x):
        return 2 * x**4 + 5 * x**3 - 30 * x**2 + x - 1

    def df(x):
        return 8 * x**3 + 15 * x**2 - 60 * x + 1

    x = np.arange(-5, 5.1, 0.1)
    y = f(x)
    plt.plot(x, y)
    plt.xlabel("x")
    plt.ylabel("y")

    local_min = grad_desc_momentum(5, f, df, lr=1e-3, eps=1e-3, alpha=0.8)

    exact_min = -3.83775470916525
    print("Gradient Descent's Local minimum occurs at", local_min)
    print("Exact minimum occurs at", exact_min)
    print("Error rate is", abs((local_min - exact_min) / exact_min * 100), "%")
    plt.show()
