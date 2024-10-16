# 3D projections might not work with the system install of matplotlib (say, from ROS)
# Use a virtual environment in this case. This filter escalates the warning to an error
PLOT_3D = False
import warnings; warnings.filterwarnings("error"); PLOT_3D = True  # comment me if thou must
try:
    import matplotlib.pyplot as plt
except UserWarning:
    print("See comment about matplotlib. Exiting")
    exit(0)

from matplotlib.gridspec import GridSpec
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

    start_time = time.time()
    lr = LR
    nnd_f, nnd_g = [], []
    #summarize(start_time, -1, Y, f, g, nnd_f, nnd_g)
    try:
        if g is None:
            for i in range(ITERATIONS):
                nf_Y = best_match_f(f, Y)
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

def f_nn_distance_stats(nnd, f):
    min_d = []
    for i, w in enumerate(f):
        d = np.sqrt(np.sum((w - f)**2, axis=1))
        d[i] = np.inf
        min_d.append(np.min(d))
    min_d = np.array(min_d)
#    nnd.append([np.mean(min_d), np.std(min_d)])
    nnd.append([np.median(min_d), np.min(min_d), np.max(min_d)])

def g_nn_distance_stats(nnd, g):
    min_d = []
    for i, w in enumerate(g):
        d = np.abs(w - g)
        d[i] = np.inf
        min_d.append(np.min(d))
    min_d = np.array(min_d)
#    nnd.append([np.mean(min_d), np.std(min_d)])
    nnd.append([np.median(min_d), np.min(min_d), np.max(min_d)])

def summarize(start_time, i, Y, f, g, nnd_f, nnd_g):
    end_time = time.time()
    print(f"Training done in {end_time - start_time:0.3f} seconds")
    plot(i, Y, f, g, nnd_f, nnd_g)

def best_match_f(f, Y):
    Y = np.expand_dims(Y, axis=1)
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

def B_outer(i, y, y_):
    sigma_B = SIGMA_B*anneal(i)
    diff = np.diagonal(np.subtract.outer(y, y_), axis1=1, axis2=3)
    return np.exp(-np.sum(diff**2, axis=-1)/(2*sigma_B**2))

def df_X(lr, i, Y, nf_Y, f, X):
    Y = Y/(K-1)

    X = X/(K**D-1)
    nf_Y = nf_Y/(K**D-1)

    A_ = A_outer(i, X, nf_Y.reshape(-1))  # A_[i, j] == A(X, nf_Y[i])[j]
    # A_[i, :] is Gaussian, centered at (K**2-1)*nf_Y.reshape(-1)[i]
    Ymf = np.diagonal(np.subtract.outer(Y, f), axis1=1, axis2=3)
    B_ = B_outer(i, Y, f)
    # B_[:, x] is a 2D Gaussian, centered at f[x]
    Bp = Ymf*np.expand_dims(B_, axis=-1)
    df_X_Y = A_[:, :, np.newaxis]*Bp
    df_X = np.sum(df_X_Y, axis=0)
    if False:#i > 250:
        j = 100
        plt.plot(A_[j, :], label=f"A({j})"); plt.vlines(nf_Y.reshape(-1)[j]*(K**2-1), 0, 1); plt.legend(); plt.show()
        plt.imshow(B_[:, j].reshape(K, K)); plt.title(f"B({j})"); plt.show()
        plt.plot(Bp[:, j, 0].reshape(K, K)[:, K//2], label=f"B'({j}, c=0)"); plt.legend(); plt.show()
        print(df_X_Y.shape); plt.plot(df_X_Y[:, j, 0], label=f"df_X_Y({j}, c=0)"); plt.plot(df_X_Y[:, j, 1]); plt.legend(); plt.show()
        print(df_X.shape); plt.plot(df_X[:, 0], label="df"); plt.plot(df_X[:, 1]); plt.legend(); plt.show()
    return lr * df_X

def dg_Y(lr, i, X, g, Y, ng_X):
    X = X/(K**D-1)

    Y = Y/(K-1)
    ng_X = ng_X/(K-1)

    A_ = A_outer(i, X, g.reshape(-1))
    Ap = np.subtract.outer(X, g.reshape(-1)).T*A_
    B_ = B(i, Y, ng_X[:, np.newaxis, :]).T
    dg_X_Y = Ap * B_
    dg_Y = np.sum(dg_X_Y, axis=1).reshape(g.shape)
    if False:#i > 250:
        j = 100
        plt.plot(A_[j, :], label=f"A({j})"); plt.legend(); plt.show()
        plt.plot(Ap[j, :], label=f"A'({j})"); plt.legend(); plt.show()
        plt.imshow(B_[:, j].reshape(K, K)); plt.title(f"B({j})"); plt.show()
        print(dg_X_Y.shape); plt.imshow(dg_X_Y[:, j].reshape(K, K)); plt.colorbar(); plt.title(f"df_X_Y({j})"); plt.show()
        print(dg_Y.shape); plt.imshow(dg_Y[:, :]); plt.colorbar(); plt.title("dg"); plt.show()
    return lr * dg_Y

def plot(i, Y, f, g, nnd_f, nnd_g):
    PLOT_F = f is not None
    PLOT_G = g is not None
    PLOT_BOTH = PLOT_F and PLOT_G

    fig = plt.figure(layout="constrained")
    fig.suptitle(f"{K=} {SEED=} {SIGMA_A=:.3} {SIGMA_B=:.3}\n{ANNEAL_N=} {ANNEAL_D=} {LR=} {LR_DECAY=} iterations={i+1}")

    gs = GridSpec(2, 4 if PLOT_BOTH else 2, figure=fig)

    if PLOT_F:
        if PLOT_3D:
            ax = fig.add_subplot(gs[0, 0], projection="3d")
            plot_f_3d(ax, f)
        ax = fig.add_subplot(gs[0, 1])
        plot_f_path(ax, f, Y)
        if PLOT_BOTH:
            ax = fig.add_subplot(gs[0, 2:])
        else:
            ax = fig.add_subplot(gs[1, :])
        if nnd_f:
            plot_f_wstats(ax, nnd_f)

    if PLOT_G:
        G_ROW = 1 if PLOT_F else 0
        if PLOT_3D:
            ax = fig.add_subplot(gs[G_ROW, 0], projection="3d")
            plot_g_3d(ax, g)
        ax = fig.add_subplot(gs[G_ROW, 1])
        plot_g_path(ax, g)
        if PLOT_BOTH:
            ax = fig.add_subplot(gs[G_ROW, 2:])
        else:
            ax = fig.add_subplot(gs[G_ROW+1, :])
        if nnd_g:
            plot_g_wstats(ax, nnd_g)

    plt.show()

def plot_f_3d(ax, f):
    ax.set_xlabel("$x$")
    ax.set_ylabel("$y_0$")
    ax.set_zlabel("$y_1$")
    for x, fx in enumerate(f):
        ax.plot([x, x], [0, fx[0]*(K-1)], [0, fx[1]*(K-1)], "b-")

def plot_f_path(ax, f, Y):
    ax.set_aspect("equal")
    ax.set_xlabel("$y_0$")
    ax.set_ylabel("$y_1$")
    ax.grid()
    path = Y[np.argsort(best_match_f(f, Y))]
    path = path[:,0], path[:,1]
    ax.plot(*path, ".r-", alpha=0.8)
    ax.scatter(f[:,0]*(K-1), f[:,1]*(K-1), c="b", marker="o", alpha=0.8)

def plot_f_wstats(ax, wstats):
    iterations = np.arange(len(wstats))
    wstats = np.array(wstats)
    ax.grid()
    ax.plot(iterations, [SIGMA_A*anneal(i) for i in iterations], label="$\\sigma_A$")
    ax.plot(iterations, wstats[:, 2], label="Max inter-node closest distance")
    ax.plot(iterations, wstats[:, 0], label="Median inter-node closest distance")
    ax.plot(iterations, wstats[:, 1], label="Min inter-node closest distance")
    ax.hlines(1/K, 0, iterations[-1], linestyles="dashed", label=f"$1/K = 1/{K}$")
    ax.set_yscale("log")
    ax.legend()

def plot_g_3d(ax, g):
    ax.set_xlabel("$y_0$")
    ax.set_ylabel("$y_1$")
    ax.set_zlabel("$x$")
    X_, Y_ = np.arange(K), np.arange(K)
    X_, Y_ = np.meshgrid(X_, Y_)
    ax.plot_surface(X_, Y_, g, cmap=cm.coolwarm)

def plot_g_path(ax, g):
    X = np.arange(K**2)
    ax.set_aspect("equal")
    ax.set_xlabel("$y_0$")
    ax.set_ylabel("$y_1$")
    ax.grid()
    path = best_match_g(g, X)
    path = path[:,1], path[:,0]
    ax.plot(*path, ".g-")

def plot_g_wstats(ax, wstats):
    iterations = np.arange(len(wstats))
    wstats = np.array(wstats)
    ax.grid()
    ax.plot(iterations, [SIGMA_B*anneal(i) for i in iterations], label="$\\sigma_B$")
    ax.plot(iterations, wstats[:, 2], label="Max inter-node closest distance")
    ax.plot(iterations, wstats[:, 0], label="Median inter-node closest distance")
    ax.plot(iterations, wstats[:, 1], label="Min inter-node closest distance")
    ax.hlines(1/K**2, 0, iterations[-1], linestyles="dashed", label=f"$1/K^2 = 1/{K**2}$")
    ax.set_yscale("log")
    ax.legend()


if __name__ == "__main__":
    main()