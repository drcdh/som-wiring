import math
import numpy as np

from initial import (
    best_match_f as best_match_y,
    best_match_g as best_match_x,
    A,
    B,
    df_x,
    dg_y,
)

from main import (
    best_match_f as best_match_Y,
    best_match_g as best_match_X,
    A as A_vec,
    A_outer as A_vec_outer,
    B as B_vec,
    B_outer as B_vec_outer,
    df_X,
    dg_Y,
)

SEED = 1959

D = 2
K = 15
C = D

SIGMA_A = 0.2
SIGMA_B = 0.2


def test_nf_Y():
    f = np.random.rand(K**D, C)  # X->Y or line->square
    Y = np.array(np.meshgrid(*(D*(range(K),)))).T.reshape(-1, D)
    nf_Y = []
    for y in Y:
        nf_Y.append(best_match_y(f, y))
    nf_Y_vec = best_match_Y(f, Y)
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
    X = np.arange(K**C)
    A_ = []
    for y in Y:
        nf_y = best_match_y(f, y)
        for x in range(K**C):
            A_.append(A(1, x, nf_y))
            #Ap_.append()
    A_ = np.array(A_).reshape(K**2, K**2)
    nf_Y_vec = best_match_Y(f, Y)
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

def df_vec(f, Y, X):
    nf_Y = best_match_Y(f, Y)
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
    X = np.arange(K**C)
    df_ = df(f, Y)
    df_vec_ = df_vec(f, Y, X)
    assert np.allclose(df_, df_vec_)

def test_dg():
    g = np.random.rand(K, K)  # Y->X or square->line
    Y = np.array(np.meshgrid(*(D*(range(K),)))).T.reshape(-1, D)
    X = np.arange(K**C)
    dg_ = dg(g, Y)
    dg_vec_ = dg_vec(g, Y, X)
    assert np.allclose(dg_, dg_vec_)
