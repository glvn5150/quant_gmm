import pandas as pd
import numpy as np
from dataclasses import dataclass
from sklearn.base import BaseEstimator
from scipy.optimize import minimize
from gmm.models.utils import nw_hacest

@dataclass
class GMMOutput:
    params: np.ndarray
    se: np.ndarray
    jstat: float
    df: int
    pricing_errors: pd.Series
    fitted: dict

class linear_sdf_gmm(BaseEstimator):
    def __init__(self, nw_lags=4, maxiter=10000, tol=1e-10, verbose=False):
        self.nw_lags = nw_lags
        self.maxiter = maxiter
        self.tol = tol
        self.verbose = verbose
        self.results_ = None

    # g_t(b) = m_t R_t - 1
    def _moments(self, b, R, F):
        m_t = 1 - F @ b           # (T,)
        g_t = m_t[:, None] * R - 1
        return g_t               # (T,N)

    def _gbar(self, b, R, F):
        return self._moments(b, R, F).mean(axis=0)  # (N,)

    def fit(self, R: pd.DataFrame, F: pd.DataFrame):

        df = R.join(F, how="inner")
        R = df[R.columns]
        F = df[F.columns]

        T, N = R.shape
        K = F.shape[1]

        R_n = R.to_numpy()
        F_n = F.to_numpy()

        # ---------- first stage ----------
        W1 = np.eye(N)

        def obj_1(b):
            gbar = self._gbar(b, R_n, F_n)
            return gbar @ W1 @ gbar

        b0 = np.zeros(K)
        res1 = minimize(obj_1, b0, method="BFGS",
                        options=dict(maxiter=self.maxiter, gtol=self.tol))
        b1 = res1.x

        # ---------- second stage ----------
        g_t = self._moments(b1, R_n, F_n)
        S = nw_hacest(g_t, self.nw_lags) / T
        epsilon = 1e-6 * np.identity(S.shape[0])
        S_regularized = S + epsilon
        W2 = np.linalg.inv(S_regularized)

        def obj_2(b):
            gbar = self._gbar(b, R_n, F_n)
            return gbar @ W2 @ gbar

        res2 = minimize(obj_2, b1, method="BFGS",
                        options=dict(maxiter=self.maxiter, gtol=self.tol))
        b2 = res2.x

        # ---------- Inference ----------
        # Jacobian: D = E[-R_t F_t']
        D = np.zeros((N, K))
        for t in range(T):
            D -= np.outer(R_n[t], F_n[t])
        D /= T

        V = np.linalg.inv(D.T @ W2 @ D) / T
        se = np.sqrt(np.diag(V))

        gbar = self._gbar(b2, R_n, F_n)
        jstat = T * gbar @ W2 @ gbar
        df_j = N - K

        pricing_errors = pd.Series(gbar, index=R.columns)

        self.results_ = GMMOutput(
            params=b2,
            se=se,
            jstat=jstat,
            df=df_j,
            pricing_errors=pricing_errors,
            fitted={"W": W2, "D": D}
        )
        return self