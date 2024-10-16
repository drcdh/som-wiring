import math
import numpy as np
import pytest

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


def initialize(batch_size=None):
    f = np.random.rand(K**D, C)  # X->Y or line->square
    g = np.random.rand(K, K)  # Y->X or square->line
    Y = np.array(np.meshgrid(*(D*(range(K),)))).T.reshape(-1, D)
    X = np.arange(K**C)
    S = np.arange(K**D)
    if batch_size:
        np.random.shuffle(S)
        #Y = Y[S[:batch_size]]
        #X = X[S[:batch_size]]
    return f, g, Y, X, S


@pytest.mark.parametrize("batch_size", [None, 1, 64])
def test_nf_Y(batch_size):
    f, _, Y, _, S = initialize(batch_size)
    nf_Y = []
    for y in Y[S]:
        nf_Y.append(best_match_y(f, y))
    nf_Y = np.array(nf_Y)
    nf_Y_vec = best_match_Y(f, Y[S])
    assert nf_Y.shape == nf_Y_vec.shape
    assert np.allclose(nf_Y, nf_Y_vec)

@pytest.mark.parametrize("batch_size", [None, 1, 64])
def test_ng_X(batch_size):
    _, g, _, X, S = initialize(batch_size)
    ng_X = []
    for x in X[S]:
        ng_X.append(best_match_x(g, x))
    ng_X = np.array(ng_X)
    ng_X_vec = best_match_X(g, X[S])
    assert ng_X.shape == ng_X_vec.shape
    assert np.allclose(ng_X, ng_X_vec)

#####

@pytest.mark.parametrize("batch_size", [None, 1, 64])
def test_f_A(batch_size):
    f, _, Y, X, S = initialize(batch_size)
    Y = Y[S]
    A_ = []
    for y in Y:
        nf_y = best_match_y(f, y)
        for x in X:
            A_.append(A(1, x, nf_y))
            #Ap_.append()
    A_ = np.array(A_).reshape(len(X), len(X))
    nf_Y_vec = best_match_Y(f, Y)
    A_vec_ = A_vec(1, X, nf_Y_vec[:, np.newaxis])
    assert A_.shape == A_vec_.shape
    assert np.allclose(A_, A_vec_)

@pytest.mark.parametrize("batch_size", [None, 1, 64])
def test_f_Bp(batch_size):
    f, _, Y, X, S = initialize(batch_size)
    Y = Y[S]
    Y_EXP = np.expand_dims(Y, axis=1)
    Bp_ = []
    for y in Y:
        for x in X:
            f_x = f[x]
            Bp_.append((y - f_x)*B(1, y, f_x))
    #B_ = B_vec(1, Y[:, np.newaxis], f[X])[:, :, np.newaxis]
    Ymf = Y[:, np.newaxis] - f
    B_ = B_vec_outer(1, Y, f)
    Bp_ = np.array(Bp_).reshape(len(X), len(X), 2)
    Bp_vec_ = Ymf*np.expand_dims(B_, axis=-1)
    assert Bp_.shape == Bp_vec_.shape
    assert np.allclose(Bp_, Bp_vec_)

#####

@pytest.mark.parametrize("batch_size", [None, 1, 64])
def test_g_A(batch_size):
    _, g, Y, X, S = initialize(batch_size)
    X = X[S]
    A_, Ap_ = [], []
    Xmg = []
    for y in Y:
        g_y = g[tuple(y)]
        for x in X:
            Xmg.append(x - g_y)
            A_.append(A(1, x, g_y))
            Ap_.append((x - g_y)*A(1, x, g_y))
    A_ = np.array(A_).reshape(len(X), len(X))
    g = g[Y[:, 0], Y[:, 1]]
    A_vec_ = A_vec_outer(1, X, g.reshape(-1))
    assert A_.shape == A_vec_.shape
    assert np.allclose(A_, A_vec_)
    Ap_ = np.array(Ap_).reshape(len(X), len(X))
    Xmg = np.array(Xmg).reshape(len(X), len(X))
    Xmg_vec = np.subtract.outer(X, g.reshape(-1)).T
    assert Xmg.shape == Xmg_vec.shape
    assert np.allclose(Xmg, Xmg_vec)
    Ap_vec_ = ((Xmg_vec)*A_vec_outer(1, X, g.reshape(-1)))
    assert Ap_.shape == Ap_vec_.shape
    assert np.allclose(Ap_, Ap_vec_)

@pytest.mark.parametrize("batch_size", [None, 1, 64])
def test_g_B(batch_size):
    _, g, Y, X, S = initialize(batch_size)
    X = X[S]
    B_ = []
    for y in Y:
        for x in X:
            ng_x = best_match_x(g, x)
            B_.append(B(1, y, ng_x))
    B_ = np.array(B_).reshape(len(X), len(X))
    ng_X = best_match_X(g, X)
    B_vec_ = B_vec(1, Y, ng_X[:, np.newaxis, :]).T
    assert B_.shape == B_vec_.shape
    assert np.allclose(B_, B_vec_)

#####

def df(f, Y, X, g=None):
    df = np.zeros_like(f)
    for y in Y:
        nf_y = best_match_y(f, y)
        if g is not None:
            nf_y = g[tuple(y)]*(K**D-1)
        for x in X:  # Assumes map is on a line
            df[x] += df_x(1, 1, y, nf_y, f, x)
    return df/np.mean(df)

def df_vec(f, Y, X, g=None):
    nf_Y = best_match_Y(f, Y)
    if g is not None:
        nf_Y = g[tuple(Y.T)].reshape(-1)*(K**D-1)
    df = df_X(1, 1, Y, nf_Y, f, X)
    return df/np.mean(df)

def dg(g, Y, X, f=None):
    dg = np.zeros_like(g)
    for x in X:
        ng_x = best_match_x(g, x)
        if f is not None:
            ng_x = f[x]*(K-1)
        for y in Y:
            dg[tuple(y)] += dg_y(1, 1, x, g, y, ng_x)
    return dg/np.mean(dg)

def dg_vec(g, Y, X, f=None):
    ng_X = best_match_X(g, X)
    dg = dg_Y(1, 1, X, g, Y, ng_X if f is None else f[X]*(K-1))
    return dg/np.mean(dg)

@pytest.mark.parametrize("batch_size", [None, 1, 64])
def test_df(batch_size):
    f, _, Y, X, S = initialize(batch_size)
    Y = Y[S]
    df_ = df(f, Y, X)
    df_vec_ = df_vec(f, Y, X)
    assert df_.shape == df_vec_.shape
    assert df_.shape == f.shape
    assert np.allclose(df_, df_vec_)

@pytest.mark.parametrize("batch_size", [None, 1, 64])
def test_dg(batch_size):
    _, g, Y, X, S = initialize(batch_size)
    X = X[S]
    dg_ = dg(g, Y, X)
    dg_vec_ = dg_vec(g, Y, X)
    assert dg_.shape == dg_vec_.shape
    assert dg_.shape == g.shape
    assert np.allclose(dg_, dg_vec_)

@pytest.mark.parametrize("batch_size", [None, 1, 64])
def test_df_dg(batch_size):
    f, g, Y, X, S = initialize(batch_size)

    df_ = df(f, Y[S], X, g)
    df_vec_ = df_vec(f, Y[S], X, g)
    assert df_.shape == df_vec_.shape
    assert df_.shape == f.shape
    assert np.allclose(df_, df_vec_)

    dg_ = dg(g, Y, X[S], f)
    dg_vec_ = dg_vec(g, Y, X[S], f)
    assert dg_.shape == dg_vec_.shape
    assert dg_.shape == g.shape
    assert np.allclose(dg_, dg_vec_)
