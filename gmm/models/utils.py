import numpy as np
import pandas as pd

#--- Simple Price Returns----
def simple_ret(px: pd.DataFrame, freq='D'):
    rets = px.pct_change().dropna()
    return rets

#---Newey-West HAC estimator for Covariance of sample moments---
def nw_hacest(S_t, L):
    T, k = S_t.shape
    S = np.zeros((k, k))
    S += (S_t.T @ S_t) / T
    for l in (1, L + 1):
        w = (L + 1 - l) / (L + 1)
        gamma_l = (S_t[1:].T @ S_t[:-1]) / T
        S += w * (gamma_l + gamma_l.T)
    return S

#---Hansen J-Stat---
def Jtest(T, g, W):
    from scipy.stats import chi2
    b = T * g.T @ W @ g
    J = T * b
    return J
