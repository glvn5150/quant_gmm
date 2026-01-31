import numpy as np
from scipy.optimize import minimize
from sklearn.base import BaseEstimator

class ForeignDomestic_GMM(BaseEstimator):
    def __init__(self, r, rf, dt=1/252, verbose=False):
        self.r = r
        self.rf = rf
        self.dt = dt
        self.verbose = verbose
        self.params_ = None
        self.jstat_ = None

    def _moments(self, params, logS, logQ):
        alpha, gamma, sigma1, sigma2 = params
        m1 = logS.mean() - alpha * self.dt
        m2 = logQ.mean() - (self.r - self.rf + gamma) * self.dt
        m3 = logS.var() - sigma1**2 * self.dt
        m4 = logQ.var() - sigma2**2 * self.dt

        return np.array([m1, m2, m3, m4])

    def fit_gmm(self,S_series,Q_series,x0=(0.0, 0.0, 0.2, 0.2)):
        logS = np.log(S_series / S_series.shift(1)).dropna().values
        logQ = np.log(Q_series / Q_series.shift(1)).dropna().values
        def objective(p):
            g = self._moments(p, logS, logQ)
            return g @ g
        res = minimize(objective,x0=np.asarray(x0),method="BFGS")
        if not res.success:
            raise RuntimeError("Foreign–Domestic GMM failed")
        self.params_ = res.x
        self.jstat_ = objective(res.x)
        alpha, gamma, sigma1, sigma2 = self.params_
        if self.verbose:
            print("\nForeign–Domestic GMM calibration complete")
            print(f"alpha  = {alpha:.6f}")
            print(f"gamma  = {gamma:.6f}")
            print(f"sigma1 = {sigma1:.6f}")
            print(f"sigma2 = {sigma2:.6f}")
            print(f"J-stat = {self.jstat_:.6e}")
        return {
            "alpha": alpha,
            "gamma": gamma,
            "sigma1": sigma1,
            "sigma2": sigma2,
            "J": self.jstat_}
