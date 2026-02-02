import numpy as np
import pandas as pd
from scipy import stats
from numpy import log, exp, sqrt
from sklearn.base import BaseEstimator
from scipy.optimize import minimize


#----basic call and put results---
def call_option_price(S, E, T, rf, sigma, return_d_parameters=False):
    d1 = (log(S / E) + (rf + sigma * sigma / 2.0) * T) / (sigma * sqrt(T))
    d2 = d1 - sigma * sqrt(T)
    price = S*stats.norm.cdf(d1)-E*exp(-rf*T)*stats.norm.cdf(d2)
    if return_d_parameters:
        return price, d1, d2
    return price

def put_option_price(S, E, T, rf, sigma, return_d_parameters=False):
    d1 = (log(S / E) + (rf + sigma * sigma / 2.0) * T) / (sigma * sqrt(T))
    d2 = d1 - sigma * sqrt(T)
    price = -S*stats.norm.cdf(-d1)+E*exp(-rf*T)*stats.norm.cdf(-d2)
    if return_d_parameters:
        return price, d1, d2
    return price

#----GMM and BSM ---
class BSM_GMM_estimator(BaseEstimator):
    def __init__(self, maxiter=1000, tol=1e-6):
        self.maxiter = maxiter
        self.tol = tol
        self.results = None

    def _moments(self, sigma, data):
        errors = []
        for index, row in data.iterrows():
            S,E,T,rf,market_price, option_type = row[['S', 'E', 'T', 'rf', 'market_price', 'type']]

            if option_type == 'call':
                model_price = call_option_price(S,E,T,rf, sigma[0])
            elif option_type == 'put':
                model_price = put_option_price(S,E,T,rf,sigma[0])
            else:
                continue
            errors.append(market_price - model_price)

        return np.array(errors)

    def _gbar(self, sigma, data):
        return self._moments(sigma, data).mean(axis=0)

    def fit(self, data):
        def obj_1(sigma_guess, data):
            gbar = self._gbar(sigma_guess, data)
            return gbar**2

        initial_sigma = np.array([0.20])

        res = minimize(fun=obj_1, x0=initial_sigma, method='BFGS',args=(data,),
                       options=dict(maxiter=self.maxiter, gtol=self.tol))

        self.estimated_sigma = res.x[0]
        self.results = res

        return self

    def compute_d1_d2(self, data):
        if not hasattr(self, "estimated_sigma"):
            raise RuntimeError("model needs to be fitted first")

        rows = []
        sigma = self.estimated_sigma


        for index, row in data.iterrows():
            S, E, T, rf, option_type = row[['S', 'E', 'T', 'rf', 'type']]
            sigma = self.estimated_sigma

            if option_type == 'call':
                price, d1,d2 = call_option_price(S,E,T,rf,sigma,return_d_parameters=True)
            elif option_type == 'put':
                price, d1,d2 = put_option_price(S,E,T,rf,sigma,return_d_parameters=True)
            else:
                continue

            rows.append({
                "index": index,
                "type": option_type,
                "S": S,
                "E": E,
                "T": T,
                "rf": rf,
                "sigma_hat": sigma,
                "d1": d1,
                "d2": d2,
                "price": price
            })

        return pd.DataFrame(rows).set_index("index")











