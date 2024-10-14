# 3D projections might not work with the system install of matplotlib (say, from ROS)
# Use a virtual environment in this case. This filter escalates the warning to an error
PLOT_3D = False
import warnings; warnings.filterwarnings("error"); PLOT_3D = True  # comment me if thou must
try:
    import matplotlib.pyplot as plt
except UserWarning:
    print("See comment about matplotlib. Exiting")
    exit(0)

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
ANNEAL_SMOOTH = True

LR = 0.1
LR_DECAY = 1.0

ITERATIONS = 1000
PLOT_ITERATIONS = 250

def main():
    np.random.seed(SEED)

    f, g = None, None

    f = np.random.rand(K**D, C)  # X->Y or line->square
    #g = np.random.rand(K, K)  # Y->X or square->line

    Y = np.array(np.meshgrid(*(D*(range(K),)))).T.reshape(-1, D)
    X = np.arange(K**C)

    Y_EXP = np.expand_dims(Y, axis=1)

    start_time = time.time()
    lr = LR
    nnd_f, nnd_g = [], []
    #summarize(start_time, -1, Y, f, g, nnd_f, nnd_g)
    try:
        if g is None:
            for i in range(ITERATIONS):
                nf_Y = best_match_f(f, Y_EXP)
                df = df_X(lr, i, Y, nf_Y, f, X)
                print("f(x):  {:3d}, {:.3f} : {:.3f} +- ({:.6f} +- {:.6f})".format(i+1, lr, np.mean(np.abs(f)), np.mean(np.abs(df)), np.std(np.abs(df))))
                f += df
                lr *= LR_DECAY
                f_nn_distance_stats(nnd_f, f)
                if (i+1)%PLOT_ITERATIONS == 0:
                    summarize(start_time, i, Y, f, g, nnd_f, nnd_g)
        elif f is None:
            for i in range(ITERATIONS):
                ng_X = best_match_g(g, X)
                dg = dg_Y(lr, i, X, g, Y, ng_X)
                print("g(y):  {:3d}, {:.3f} : {:.3f} +- ({:.6f} +- {:.6f})".format(i+1, lr, np.mean(np.abs(g)), np.mean(np.abs(dg)), np.std(np.abs(dg))))
                g += dg
                lr *= LR_DECAY
                g_nn_distance_stats(nnd_g, g)
                if (i+1)%PLOT_ITERATIONS == 0:
                    summarize(start_time, i, Y, f, g, nnd_f, nnd_g)
        else:
            for i in range(ITERATIONS):
                df = df_X(lr, i, Y, g*(K**D-1), f, X)
                dg = dg_Y(lr, i, X, g, Y, f*(K-1))
                print("f(x):  {:3d}, {:.3f} : {:.3f} +- ({:.6f} +- {:.6f})".format(i+1, lr, np.mean(np.abs(f)), np.mean(np.abs(df)), np.std(np.abs(df))))
                print("g(y):  {:3d}, {:.3f} : {:.3f} +- ({:.6f} +- {:.6f})".format(i+1, lr, np.mean(np.abs(g)), np.mean(np.abs(dg)), np.std(np.abs(dg))))
                f += df
                g += dg
                lr *= LR_DECAY
                f_nn_distance_stats(nnd_f, f)
                g_nn_distance_stats(nnd_g, g)
                if (i+1)%PLOT_ITERATIONS == 0:
                    summarize(start_time, i, Y, f, g, nnd_f, nnd_g)
    except KeyboardInterrupt:
        pass
    if (i+1)%PLOT_ITERATIONS != 0:
        summarize(start_time, i, Y, f, g, nnd_f, nnd_g)
    #print(f)

#def nn_distance_stats(nnd, map_):
#    diff = np.subtract.outer(map_, map_)
#    # don't use inter-channel differences
#    dist = np.sqrt(diff[:, :, 0, :, :, 0]**2 + diff[:, :, 1, :, :, 1]**2)
#    dist = np.fill_diagonal(dist.reshape(K**D, K**2), np.inf).reshape(K, K, K, K)
#    min_dist = np.min(dist)
#    mean, std  = np.mean(dist), np.std(dist)
#    nnd.append([mean, std])

def f_nn_distance_stats(nnd, f):
    min_d = []
    for i, w in enumerate(f):
        d = np.sqrt(np.sum((w - f)**2, axis=1))
        d[i] = np.inf
        min_d.append(np.min(d))
    min_d = np.array(min_d)
    nnd.append([np.mean(min_d), np.std(min_d)])

def g_nn_distance_stats(nnd, g):
    min_d = []
    for i, w in enumerate(g):
        d = np.abs(w - g)
        d[i] = np.inf
        min_d.append(np.min(d))
    min_d = np.array(min_d)
    nnd.append([np.mean(min_d), np.std(min_d)])

def summarize(start_time, i, Y, f, g, nnd_f, nnd_g):
    end_time = time.time()
    print(f"Training done in {end_time - start_time:0.3f} seconds")
    plot(i, Y, f, g, nnd_f, nnd_g)

def best_match_f(f, Y):
    Y = Y/(K-1)  # map from [[0, K-1], [0, K-1]] to [[0, 1], [0, 1]]
    return np.argmin(np.sum((f-Y)**2, axis=2), axis=1)

def best_match_g(g, X):
    X = X/(K**D-1)  # map from [0, K**D-1] to [0, 1]
    return np.array(np.unravel_index(np.argmin(np.abs(g.reshape(-1)-X[:, np.newaxis]), axis=1), g.shape)).T.reshape(K**D, C)


def anneal(i):
    if ANNEAL_SMOOTH:
        return 1.0/(1.0 + ANNEAL_N*(i/ANNEAL_D))
    return 1.0/(1.0 + ANNEAL_N*(i//ANNEAL_D))

def A(i, x, x_):
    sigma_A = SIGMA_A*anneal(i)
    return np.exp(-(x - x_)**2/(2*sigma_A**2))

def A_outer(i, x, x_):
    sigma_A = SIGMA_A*anneal(i)
    return np.exp(-np.subtract.outer(x, x_)**2/(2*sigma_A**2)).T

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
    return np.exp(-np.sum((y - y_)**2, axis=-1)/(2*sigma_B**2))

def df_X(lr, i, Y, nf_Y, f, X):
    Y = Y/(K-1)

    X = X/(K**D-1)
    nf_Y = nf_Y/(K**D-1)

    A_ = A(i, X, nf_Y[:, np.newaxis])  # A_[i, j] == A(X, nf_Y[i])[j]
    Bp = (Y[:, np.newaxis] - f)*B(i, Y[:, np.newaxis], f)[:, :, np.newaxis]  # Bp[i, j] == (Y[i] - f[j])*B(Y[i], f[j])
    df_X_Y = A_[:, :, np.newaxis]*Bp
    return lr * np.sum(df_X_Y, axis=0)

def dg_Y(lr, i, X, g, Y, ng_X):
    X = X/(K**D-1)

    Y = Y/(K-1)
    ng_X = ng_X/(K-1)

    Ap = np.subtract.outer(X, g.reshape(-1)).T*A_outer(i, X, g.reshape(-1))
    B_ = B(i, Y, ng_X[:, np.newaxis, :]).T
    dg_X_Y = Ap * B_
    return lr * np.sum(dg_X_Y, axis=1).reshape(g.shape)

def plot(i, Y, f, g, nnd_f, nnd_g):
    Y_EXP = np.expand_dims(Y, axis=1)
    rows = 2 if f is None or g is None else 3
    cols = 2 if PLOT_3D else 1
    index = 1
    fig = plt.figure(figsize=plt.figaspect(.5))
    fig.suptitle(f"{K=} {SEED=} {SIGMA_A=:.3} {SIGMA_B=:.3}\n{ANNEAL_N=} {ANNEAL_D=} {LR=} {LR_DECAY=} iterations={i+1}")
    if f is not None:
        if PLOT_3D:
            ax = fig.add_subplot(rows, cols, index, projection="3d")
            ax.set_xlabel("$x$")
            ax.set_ylabel("$y_0$")
            ax.set_zlabel("$y_1$")
            for x, fx in enumerate(f):
                ax.plot([x, x], [0, fx[0]*(K-1)], [0, fx[1]*(K-1)], "b-")
            index += 1
        ax = fig.add_subplot(rows, cols, index)
        #ax.set_aspect("equal")
        ax.set_xlabel("$y_0$")
        ax.set_ylabel("$y_1$")
        ax.grid()
        path = Y[np.argsort(best_match_f(f, Y_EXP))]
        path = path[:,0], path[:,1]
        ax.plot(*path, ".r-")
        ax.scatter(f[:,0]*(K-1), f[:,1]*(K-1), c="b", marker="o")
        index += 1
    if g is not None:
        X = np.arange(K**2)
        if PLOT_3D:
            ax = fig.add_subplot(rows, cols, index, projection="3d")
            ax.set_xlabel("$y_0$")
            ax.set_ylabel("$y_1$")
            ax.set_zlabel("$x$")
            X_, Y_ = np.arange(K), np.arange(K)
            X_, Y_ = np.meshgrid(X_, Y_)
            ax.plot_surface(X_, Y_, g, cmap=cm.coolwarm)
            index += 1
        ax = fig.add_subplot(rows, cols, index)
        ax.set_xlabel("$y_0$")
        ax.set_ylabel("$y_1$")
        ax.grid()
        path = best_match_g(g, X)
        path = path[:,1], path[:,0]
        ax.plot(*path, ".g-")
        index += 1
    ax = fig.add_subplot(rows, cols, index)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Distance")
    ax.grid()
    if nnd_f or nnd_g:
        iterations = np.arange(len(nnd_f or nnd_g))
        if nnd_f:
            nnd_f = np.array(nnd_f)
            ax.errorbar(iterations, nnd_f[:, 0], nnd_f[:, 1], label="$\\overline{f_{min}(x)}$")
            ax.plot(iterations, [SIGMA_A*anneal(i) for i in iterations], label="$\\sigma_A$")
            ax.hlines(1/K, 0, iterations[-1], linestyles="dashed", label=f"$1/K = 1/{K}$")
        if nnd_g:
            nnd_g = np.array(nnd_g)
            ax.errorbar(iterations, nnd_g[:, 0], nnd_g[:, 1], label="$\\overline{g_{min}(y)}$")
            ax.plot(iterations, [SIGMA_B*anneal(i) for i in iterations], label="$\\sigma_B$")
            ax.hlines(1/K**2, 0, iterations[-1], linestyles="dashed", label=f"$1/K^2 = 1/{K**2}$")
        ax.legend()
    index += 1
    plt.show()


if __name__ == "__main__":
    main()