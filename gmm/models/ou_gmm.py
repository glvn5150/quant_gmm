import numpy as np
from scipy.optimize import minimize
from sklearn.base import BaseEstimator
from gmm.base_models.general_ou import OU

class OU_GMM(BaseEstimator):
    def __init__(self, dt=1/252, verbose=False):
        self.dt = dt
        self.verbose = verbose
        self.params_ = None
        self.jstat_ = None

    def _moments(self, params, X):
        kappa, theta, sigma = params
        dX = X[1:] - X[:-1]
        Xlag = X[:-1]

        m1 = dX.mean()
        m2 = dX.var()
        m3 = np.mean((Xlag - Xlag.mean()) * dX)

        # theoretical moments
        phi = np.exp(-kappa * self.dt)
        var_theory = sigma**2 * (1 - phi**2) / (2 * kappa)
        g1 = m1 - kappa * (theta - Xlag.mean()) * self.dt
        g2 = m2 - var_theory
        g3 = m3 + kappa * Xlag.var() * self.dt

        return np.array([g1, g2, g3])

    def fit(self, X, x0=(0.5, 0.0, 0.1)):
        X = np.asarray(X)

        def objective(p):
            g = self._moments(p, X)
            return g @ g

        res = minimize(objective, x0=np.asarray(x0), method="BFGS")

        if not res.success:
            raise RuntimeError("OU GMM failed")

        self.params_ = res.x
        self.jstat_ = objective(res.x)

        if self.verbose:
            k, th, s = self.params_
            print("\nOU GMM calibration complete")
            print(f"kappa = {k:.6f}")
            print(f"theta = {th:.6f}")
            print(f"sigma = {s:.6f}")
            print(f"J-stat = {self.jstat_:.6e}")
        return self

    def fitted_model(self, X0):
        k, th, s = self.params_
        return OU(X0=X0, kappa=k, theta=th, sigma=s, dt=self.dt)
