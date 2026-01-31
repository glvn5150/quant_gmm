from gmm.base_models.bdt import BlackDermanToy
from sklearn.base import BaseEstimator
import numpy as np
from scipy.optimize import minimize

class BlackDermanToy_GMM(BaseEstimator):
    def __init__(self, quotes, maturity):
        self.quotes = quotes
        self.maturity = maturity
        self.theta_hat = None

    def _moments(self, alpha):
        alpha = float(alpha)
        scaled_quotes = {
            t:[r, alpha*s]
            for t, (r,s) in self.quotes.items()}
        model = BlackDermanToy(scaled_quotes, self.maturity, verbose=False)
        errors = []
        for t in range(1, self.maturity + 1):
            model_price = model.zcb_prices[t]
            market_price = np.exp(-self.quotes[t][0])
            errors.append(model_price - market_price)

        return np.array(errors)

    def fit(self):
        def obj(alpha):
            g = self._moments(alpha)
            return g@g

        res = minimize(obj, x0=np.array([1.0]), method = "BFGS")

        if not res.success:
            raise RuntimeError("GMM failure")

        self.theta_hat = res.x[0]
        return self



