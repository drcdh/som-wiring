import warnings

warnings.filterwarnings("error")

import math
import matplotlib.pyplot as plt
import numpy as np
import time

from matplotlib import cm


SEED = 1959

D = 2
K = 15
C = D

SIGMA_A = 0.2
SIGMA_B = 0.2

ANNEAL_N = 1
ANNEAL_D = 100

LR = 0.1
LR_DECAY = 1.0

ITERATIONS = 1000
PLOT_ITERATIONS = 100

def main():
    np.random.seed(SEED)

    f, g = None, None

    f = np.random.rand(K**D, C)  # X->Y or line->square
    Y = np.array(np.meshgrid(*(D*(range(K),)))).T.reshape(-1, D)

    g = np.random.rand(K, K)  # Y->X or square->line

    #print(f); print(Y)

    start_time = time.time()
    lr = LR
    summarize(start_time, -1, Y, f, g)
    try:
        if g is None:
            for i in range(ITERATIONS):
                df = np.zeros_like(f)
                for y in Y:
                    nf_y = best_match_f(f, y)
                    for x in range(K**C):  # Assumes map is on a line
                        df[x] += df_x(lr, i, y, nf_y, f, x)
                print("f(x):  {:3d}, {:.3f} : {:.3f} +- ({:.6f} +- {:.6f})".format(i+1, lr, np.mean(np.abs(f)), np.mean(np.abs(df)), np.std(np.abs(df))))
                f += df
                lr *= LR_DECAY
                if (i+1)%PLOT_ITERATIONS == 0:
                    summarize(start_time, i, Y, f, g)
        elif f is None:
            for i in range(ITERATIONS):
                dg = np.zeros_like(g)
                for x in range(K**C):
                    ng_x = best_match_g(g, x)
                    for y in Y:
                        dg[tuple(y)] += dg_y(lr, i, x, g, y, ng_x)
                print("g(y):  {:3d}, {:.3f} : {:.3f} +- ({:.6f} +- {:.6f})".format(i+1, lr, np.mean(np.abs(g)), np.mean(np.abs(dg)), np.std(np.abs(dg))))
                g += dg
                lr *= LR_DECAY
                if (i+1)%PLOT_ITERATIONS == 0:
                    summarize(start_time, i, Y, f, g)
        else:
            for i in range(ITERATIONS):
                df = np.zeros_like(f)
                dg = np.zeros_like(g)
                for y in Y:
                    g_y = g[tuple(y)]
                    for x in range(K**C):
                        df[x] += df_x(lr, i, y, g_y*(K**D-1), f, x)
                        dg[tuple(y)] += dg_y(lr, i, x, g, y, f[x]*(K-1))
                print("f(x):  {:3d}, {:.3f} : {:.3f} +- ({:.6f} +- {:.6f})".format(i+1, lr, np.mean(np.abs(f)), np.mean(np.abs(df)), np.std(np.abs(df))))
                print("g(y):  {:3d}, {:.3f} : {:.3f} +- ({:.6f} +- {:.6f})".format(i+1, lr, np.mean(np.abs(g)), np.mean(np.abs(dg)), np.std(np.abs(dg))))
                f += df
                g += dg
                lr *= LR_DECAY
                if (i+1)%PLOT_ITERATIONS == 0:
                    summarize(start_time, i, Y, f, g)
    except KeyboardInterrupt:
        pass
    if (i+1)%PLOT_ITERATIONS != 0:
        summarize(start_time, i, Y, f, g)
    #print(f)

def summarize(start_time, i, Y, f, g):
    end_time = time.time()
    print(f"Training done in {end_time - start_time:0.3f} seconds")
    plot(i, Y, f, g)

def best_match_f(f, y):
    y = y/(K-1)  # map from [[0, K-1], [0, K-1]] to [[0, 1], [0, 1]]
    return np.argmin(np.sum((f-y)**2, axis=1))

def best_match_g(g, x):
    x = x/(K**D-1)  # map from [0, K**D-1] to [0, 1]
    return np.array(np.unravel_index(np.argmin(np.abs(g-x)), g.shape))


def anneal(i):
    return 1.0/(1.0 + ANNEAL_N*(i//ANNEAL_D))

def A(i, x, x_):
    sigma_A = SIGMA_A*anneal(i)
    return math.exp(-np.sum((x - x_)**2)/(2*sigma_A**2))

def NN(y, y0):
    if y[0] == y0[0] and y[1] == y0[1]:
        return 1
    if y[0] == y0[0] and abs(y[1] - y0[1]) == 1:
        return 1
    if y[1] == y0[1] and abs(y[0] - y0[0]) == 1:
        return 1
    return 0

def B(i, y, y_):
    #return 1  # Normal Kohonen
    sigma_B = SIGMA_B*anneal(i)
    return math.exp(-np.sum((y - y_)**2)/(2*sigma_B**2))

def df_x(lr, i, y, nf_y, f, x):
    f_x = f[x]
    y = y/(K-1)

    x = x/(K**D-1)
    nf_y = nf_y/(K**D-1)

    Bp = (y - f_x)*B(i, y, f_x)
    return lr * A(i, x, nf_y) * Bp

def dg_y(lr, i, x, g, y, ng_x):
    g_y = g[tuple(y)]
    x = x/(K**D-1)

    y = y/(K-1)
    ng_x = ng_x/(K-1)

    Ap = (x - g_y)*A(i, x, g_y)
    dg_y_ = lr * Ap * B(i, y, ng_x)
    return dg_y_

def plot(i, Y, f, g):
    rows = 1 if f is None or g is None else 2
    fig = plt.figure(figsize=plt.figaspect(.5))
    fig.suptitle(f"{K=} {SEED=} {SIGMA_A=} {SIGMA_B=}\n{ANNEAL_N=} {ANNEAL_D=} {LR=} {LR_DECAY=} iterations={i+1}")
    if f is not None:
        ax = fig.add_subplot(rows, 2, 1, projection="3d")
        ax.set_xlabel("$x$")
        ax.set_ylabel("$y_0$")
        ax.set_zlabel("$y_1$")
        for x, fx in enumerate(f):
            ax.plot([x, x], [0, fx[0]*(K-1)], [0, fx[1]*(K-1)], "b-")
        ax = fig.add_subplot(rows, 2, 2)
        #ax.set_aspect("equal")
        ax.set_xlabel("$y_0$")
        ax.set_ylabel("$y_1$")
        ax.grid()
        path = Y[np.argsort(np.fromiter((best_match_f(f, y) for y in Y), dtype=float))]
        path = path[:,0], path[:,1]
        ax.plot(*path, ".r-")
        ax.scatter(f[:,0]*(K-1), f[:,1]*(K-1), c="b", marker="o")
    if g is not None:
        ax = fig.add_subplot(rows, 2, 3 if rows == 2 else 1, projection="3d")
        ax.set_xlabel("$y_0$")
        ax.set_ylabel("$y_1$")
        ax.set_zlabel("$x$")
        X_, Y_ = np.arange(K), np.arange(K)
        X_, Y_ = np.meshgrid(X_, Y_)
        ax.plot_surface(X_, Y_, g, cmap=cm.coolwarm)
        ax = fig.add_subplot(rows, 2, 4 if rows == 2 else 2)
        ax.set_xlabel("$y_0$")
        ax.set_ylabel("$y_1$")
        ax.grid()
        path = np.array([best_match_g(g, _x) for _x in range(K**C)])
        path = path[:,1], path[:,0]
        ax.plot(*path, ".g-")
    plt.show()


if __name__ == "__main__":
    main()