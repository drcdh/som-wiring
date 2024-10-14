import math
import numpy as np


SEED = 1959

D = 2
K = 15
C = D

SIGMA_A = 0.2
SIGMA_B = 0.2

def best_match_Y(f, Y):
    Y = Y/(K-1)  # map from [[0, K-1], [0, K-1]] to [[0, 1], [0, 1]]
    return np.argmin(np.sum((f-Y)**2, axis=2), axis=1)

def best_match_y(f, y):
    y = y/(K-1)  # map from [[0, K-1], [0, K-1]] to [[0, 1], [0, 1]]
    return np.argmin(np.sum((f-y)**2, axis=1))

def best_match_X(g, X):
    X = X/(K**D-1)  # map from [0, K**D-1] to [0, 1]
    return np.array(np.unravel_index(np.argmin(np.abs(g.reshape(-1)-X[:, np.newaxis]), axis=1), g.shape)).T.reshape(K**D, C)

def best_match_x(g, x):
    x = x/(K**D-1)  # map from [0, K**D-1] to [0, 1]
    return np.array(np.unravel_index(np.argmin(np.abs(g-x)), g.shape))

def A_vec(_, x, x_):
    return np.exp(-(x - x_)**2/(2*SIGMA_A**2))

def A_vec_outer(_, x, x_):
    return np.exp(-np.subtract.outer(x, x_)**2/(2*SIGMA_A**2)).T

def B_vec(_, y, y_):
    return np.exp(-np.sum((y - y_)**2, axis=-1)/(2*SIGMA_B**2))

def A(_, x, x_):
    return math.exp(-np.sum((x - x_)**2)/(2*SIGMA_A**2))

def B(_, y, y_):
    return math.exp(-np.sum((y - y_)**2)/(2*SIGMA_B**2))

#####

def df_x(lr, i, y, nf_y, f, x):
    f_x = f[x]
    y = y/(K-1)

    x = x/(K**D-1)
    nf_y = nf_y/(K**D-1)

    Bp = (y - f_x)*B(i, y, f_x)
    return lr * A(i, x, nf_y) * Bp

def df_X(lr, i, Y, nf_Y, f, X):
    Y = Y/(K-1)

    X = X/(K**D-1)
    nf_Y = nf_Y/(K**D-1)

    A_ = A_vec(i, X, nf_Y[:, np.newaxis])  # A_[i, j] == A(X, nf_Y[i])[j]
    Bp = (Y[:, np.newaxis] - f)*B_vec(i, Y[:, np.newaxis], f)[:, :, np.newaxis]  # Bp[i, j] == (Y[i] - f[j])*B(Y[i], f[j])
    df_X_Y = A_[:, :, np.newaxis]*Bp
    return lr * np.sum(df_X_Y, axis=0)

def dg_y(lr, i, x, g, y, ng_x):
    g_y = g[tuple(y)]
    x = x/(K**D-1)

    y = y/(K-1)
    ng_x = ng_x/(K-1)

    Ap = (x - g_y)*A(i, x, g_y)
    dg_y_ = lr * Ap * B(i, y, ng_x)
    return dg_y_

def dg_Y(lr, i, X, g, Y, ng_X):
    X = X/(K**D-1)

    Y = Y/(K-1)
    ng_X = ng_X/(K-1)

    Ap = np.subtract.outer(X, g.reshape(-1)).T*A_vec_outer(i, X, g.reshape(-1))
    B_ = B_vec(i, Y, ng_X[:, np.newaxis, :]).T
    dg_X_Y = Ap * B_
    return lr * np.sum(dg_X_Y, axis=1).reshape(g.shape)

def test_nf_Y():
    f = np.random.rand(K**D, C)  # X->Y or line->square
    Y = np.array(np.meshgrid(*(D*(range(K),)))).T.reshape(-1, D)
    Y_EXP = np.expand_dims(Y, axis=1)
    nf_Y = []
    for y in Y:
        nf_Y.append(best_match_y(f, y))
    nf_Y_vec = best_match_Y(f, Y_EXP)
    assert np.allclose(nf_Y, nf_Y_vec)

def test_nf_X():
    g = np.random.rand(K, K)  # Y->X or square->line
    X = np.arange(K**C)
    nf_X = []
    for x in range(K**C):
        nf_X.append(best_match_x(g, x))
    nf_X_vec = best_match_X(g, X)
    assert np.allclose(nf_X, nf_X_vec)

#####

def test_f_A():
    f = np.random.rand(K**D, C)  # X->Y or line->square
    Y = np.array(np.meshgrid(*(D*(range(K),)))).T.reshape(-1, D)
    Y_EXP = np.expand_dims(Y, axis=1)
    X = np.arange(K**C)
    A_ = []
    for y in Y:
        nf_y = best_match_y(f, y)
        for x in range(K**C):
            A_.append(A(1, x, nf_y))
            #Ap_.append()
    A_ = np.array(A_).reshape(K**2, K**2)
    nf_Y_vec = best_match_Y(f, Y_EXP)
    A_vec_ = A_vec(1, X, nf_Y_vec[:, np.newaxis])
    assert np.allclose(A_, A_vec_)

def test_f_Bp():
    f = np.random.rand(K**D, C)  # X->Y or line->square
    Y = np.array(np.meshgrid(*(D*(range(K),)))).T.reshape(-1, D)
    Y_EXP = np.expand_dims(Y, axis=1)
    X = np.arange(K**C)
    Bp_ = []
    for y in Y:
        for x in range(K**C):
            f_x = f[x]
            Bp_.append((y - f_x)*B(1, y, f_x))
    Bp_ = np.array(Bp_).reshape(K**2, K**2, 2)
    Bp_vec_ = (Y[:, np.newaxis] - f)*B_vec(1, Y[:, np.newaxis], f)[:, :, np.newaxis]
    assert np.allclose(Bp_, Bp_vec_)

#####

def test_g_A():
    g = np.random.rand(K, K)  # Y->X or square->line
    Y = np.array(np.meshgrid(*(D*(range(K),)))).T.reshape(-1, D)
    X = np.arange(K**C)
    A_, Ap_ = [], []
    Xmg = []
    for y in Y:
        for x in range(K**C):
            g_y = g[tuple(y)]
            Xmg.append(x - g_y)
            A_.append(A(1, x, g_y))
            Ap_.append((x - g_y)*A(1, x, g_y))
    A_ = np.array(A_).reshape(K**2, K**2)
    A_vec_ = A_vec_outer(1, X, g.reshape(-1))
    assert np.allclose(A_, A_vec_)
    Ap_ = np.array(Ap_).reshape(K**2, K**2)
    Xmg = np.array(Xmg).reshape(K**2, K**2)
    Xmg_vec = np.subtract.outer(X, g.reshape(-1)).T
    assert np.allclose(Xmg, Xmg_vec)
    Ap_vec_ = ((Xmg_vec)*A_vec_outer(1, X, g.reshape(-1)))
    assert np.allclose(Ap_, Ap_vec_)

def test_g_B():
    g = np.random.rand(K, K)  # Y->X or square->line
    Y = np.array(np.meshgrid(*(D*(range(K),)))).T.reshape(-1, D)
    X = np.arange(K**C)
    B_ = []
    for y in Y:
        for x in range(K**C):
            ng_x = best_match_x(g, x)
            B_.append(B(1, y, ng_x))
    B_ = np.array(B_).reshape(K**2, K**2)
    ng_X = best_match_X(g, X)
    B_vec_ = B_vec(1, Y, ng_X[:, np.newaxis, :]).T
    assert np.allclose(B_, B_vec_)

#####

def df(f, Y):
    df = np.zeros_like(f)
    for y in Y:
        nf_y = best_match_y(f, y)
        for x in range(K**C):  # Assumes map is on a line
            df[x] += df_x(1, 1, y, nf_y, f, x)
    return df

def df_vec(f, Y, Y_EXP, X):
    nf_Y = best_match_Y(f, Y_EXP)
    df = df_X(1, 1, Y, nf_Y, f, X)
    return df

def dg(g, Y):
    dg = np.zeros_like(g)
    for x in range(K**C):
        ng_x = best_match_x(g, x)
        for y in Y:
            dg[tuple(y)] += dg_y(1, 1, x, g, y, ng_x)
    return dg

def dg_vec(g, Y, X):
    ng_X = best_match_X(g, X)
    dg = dg_Y(1, 1, X, g, Y, ng_X)
    return dg

def test_df():
    f = np.random.rand(K**D, C)  # X->Y or line->square
    Y = np.array(np.meshgrid(*(D*(range(K),)))).T.reshape(-1, D)
    Y_EXP = np.expand_dims(Y, axis=1)
    X = np.arange(K**C)
    df_ = df(f, Y)
    df_vec_ = df_vec(f, Y, Y_EXP, X)
    assert np.allclose(df_, df_vec_)

def test_dg():
    g = np.random.rand(K, K)  # Y->X or square->line
    Y = np.array(np.meshgrid(*(D*(range(K),)))).T.reshape(-1, D)
    X = np.arange(K**C)
    dg_ = dg(g, Y)
    dg_vec_ = dg_vec(g, Y, X)
    assert np.allclose(dg_, dg_vec_)
