import numpy as np
from scipy.optimize import minimize
from sklearn.base import BaseEstimator
from scipy.linalg import inv

class ForeignDomestic_GMM(BaseEstimator):
    def __init__ (self,r,rf,dt=1 / 252,verbose=False,instruments=None):
        self.r = r
        self.rf = rf
        self.dt = dt
        self.verbose = verbose
        self.Z = instruments
        self.params_ = None
        self.jstat_ = None

    def _moments (self,params,logS,logQ):
        alpha,gamma,sigma1,sigma2 = params
        dS,dQ = logS,logQ
        m1 = dS - alpha * self.dt
        m2 = dQ - (self.r - self.rf + gamma) * self.dt
        m3 = dS ** 2 - sigma1 ** 2 * self.dt
        m4 = dQ ** 2 - sigma2 ** 2 * self.dt
        G = np.column_stack([m1,m2,m3,m4])
        if self.Z is not None:
            if np.isscalar(self.Z):
                Z_t = self.Z * np.ones(len(dS))
            else:
                Z_t = self.Z[:len(dS)]
            G = G * Z_t[:,None]
        return G

    def fit_gmm (self,S_series,Q_series,x0=(0.0,0.0,0.2,0.2)):
        logS = np.log(S_series).diff()
        logQ = np.log(Q_series).diff()
        df = np.column_stack([logS,logQ])
        df = df[~np.isnan(df).any(axis=1)]
        logS = df[:,0]
        logQ = df[:,1]
        T = len(logS)
        def objective (p,W):
            g_obs = self._moments(p,logS,logQ)
            g_avg = g_obs.mean(axis=0)
            quadratic_form = g_avg @ W @ g_avg
            return quadratic_form.item()

        W1 = np.eye(len(x0))
        res1 = minimize(objective,x0=np.asarray(x0),method="BFGS",args=(W1,))
        if not res1.success:
            raise RuntimeError("Foreign–Domestic GMM Step 1 failed")

        params1 = res1.x
        g_obs_step1 = self._moments(params1,logS,logQ)
        S_hat = (g_obs_step1.T @ g_obs_step1) / T
        eps = 1e-6 #for invertibility
        W2 = np.linalg.inv(S_hat + eps * np.eye(S_hat.shape[0]))
        res2 = minimize(objective,x0=params1,method="BFGS",args=(W2,))

        if not res2.success:
            raise RuntimeError("Foreign–Domestic GMM Step 2 failed")
        self.params_ = res2.x
        self.jstat_ = objective(res2.x,W2) * T

        alpha,gamma,sigma1,sigma2 = self.params_
        if self.verbose:
            print("\nForeign–Domestic Two-Step GMM calibration complete")
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

