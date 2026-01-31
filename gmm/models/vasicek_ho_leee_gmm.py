from sklearn.base import BaseEstimator
import numpy as np
from scipy.optimize import minimize
from gmm.base_models.vasicek_ho_lee import Vasicek_HoLee

class VasicekHoLee_GMM(BaseEstimator):
    def __init__(self, maturities, spot_rates, model_type="vasicek"):
        self.maturities = np.asarray(maturities)
        self.spot_rates = np.asarray(spot_rates)
        self.model_type = model_type.lower()
        self.theta_hat = None
        self.sigma_hat = None

    def _market_prices(self):
        return np.exp(-self.spot_rates * self.maturities)

    def _moments(self, params):
        theta = params[0]
        sigma = params[1] if len(params) > 1 else None
        model = Vasicek_HoLee(
            T=max(self.maturities),
            N=500,
            r0=self.spot_rates[0],
            theta=theta,
            sigma=sigma if sigma is not None else 0.01
        )

        errors = []
        P_mkt = self._market_prices()

        for i, T in enumerate(self.maturities):
            A, B = model.get_A_B(0.0, T, model=self.model_type)
            P_model = A * np.exp(-B * model.r0)
            errors.append(P_model - P_mkt[i])

        return np.array(errors)

    def fit(self, estimate_sigma=False):
        x0 = np.array([0.03, 0.01]) if estimate_sigma else np.array([0.03])
        def obj(p):
            g = self._moments(p)
            return g @ g

        res = minimize(obj, x0=x0, method="BFGS")

        if not res.success:
            raise RuntimeError("GMM optimization failed")

        self.theta_hat = res.x[0]
        if estimate_sigma:
            self.sigma_hat = res.x[1]

        return self