import numpy as np
from sklearn.base import BaseEstimator
from scipy.optimize import minimize
from gmm.base_models.cir_hull_white import CIR_Hull_White

class CIR_Hull_White_GMM(BaseEstimator):
    def __init__(
        self,
        maturities,
        market_zcb_prices,
        r0,
        T=30.0,
        N=300,
        nw_lags=0,
        verbose=False
    ):
        self.maturities = np.asarray(maturities)
        self.market_prices = np.asarray(market_zcb_prices)
        self.r0 = r0
        self.T = T
        self.N = N
        self.nw_lags = nw_lags
        self.verbose = verbose

        self.params_ = None
        self.model_ = None

    def _moments(self, params):
        a, theta, sigma = params

        model = CIR_Hull_White(
            T=self.T,
            N=self.N,
            r0=self.r0,
            a=a,
            theta=theta,
            sigma=sigma
        )

        errors = []
        for T_i, P_mkt in zip(self.maturities, self.market_prices):
            P_model = model.bond_price_mc(
                t=0.0,
                T_mat=T_i,
                paths=3000,
                method="full_truncation"
            )
            errors.append(P_model - P_mkt)

        return np.asarray(errors)

    def _objective(self, params):
        g = self._moments(params)
        return g @ g

    def fit(self, x0=(0.5, 0.03, 0.05)):
        res = minimize(
            self._objective,
            x0=np.asarray(x0),
            method="BFGS"
        )

        if not res.success:
            raise RuntimeError("CIR–HW GMM failed")

        self.params_ = res.x

        # store fitted model
        a, theta, sigma = self.params_
        self.model_ = CIR_Hull_White(
            T=self.T,
            N=self.N,
            r0=self.r0,
            a=a,
            theta=theta,
            sigma=sigma
        )

        if self.verbose:
            print("CIR–Hull White GMM calibration complete")
            print(f"a     = {a:.4f}")
            print(f"theta = {theta:.4f}")
            print(f"sigma = {sigma:.4f}")

        return self
