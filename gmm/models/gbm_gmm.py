# gmm/models/gbm_gmm.py
import numpy as np
from scipy.optimize import minimize
from sklearn.base import BaseEstimator

class GBM_GMM(BaseEstimator):
    def __init__(self, dt, verbose=False):
        self.dt = dt
        self.verbose = verbose
        self.params_ = None
        self.jstat_ = None

    def _moments(self, params, logrets):
        mu, sigma = params
        m1 = logrets.mean() - (mu - 0.5 * sigma**2) * self.dt
        m2 = logrets.var() - sigma**2 * self.dt
        return np.array([m1, m2])
    def fit(self, logrets, x0=(0.05, 0.2)):
        logrets = np.asarray(logrets).flatten()
        def obj(p):
            g = self._moments(p, logrets)
            return g @ g
        res = minimize(obj, x0=np.asarray(x0), method="BFGS")
        if not res.success:
            raise RuntimeError("GBM GMM failed")
        self.params_ = res.x
        self.jstat_ = obj(res.x)
        if self.verbose:
            mu, sigma = self.params_
            print("GBM GMM calibration complete")
            print(f"mu    = {mu:.6f}")
            print(f"sigma = {sigma:.6f}")
            print(f"J-stat= {self.jstat_:.6e}")
        return self
