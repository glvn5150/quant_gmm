import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from numpy import sqrt, log, exp
import pandas as pd
import statsmodels.api as sm

class ForexModel:
    '''Mathematics orginated from Shreeve Volume II'''
    #define time variables
    def __init__(self, T=1, N=1000):
        # only time-related setup in __init__
        self.N = N
        self.T = T
        self.dt = T / N
        self.t_axis = np.linspace(0, T, N + 1)

    #simulate brownian motion
    def _generate_shocks(self, rho):
        dW1 = sqrt(self.dt) * np.random.randn(self.N)
        dW2 = sqrt(self.dt) * np.random.randn(self.N)
        dW3 = rho * dW1 + sqrt(1 - rho ** 2) * dW2
        return dW1, dW2, dW3

    #simulate domestic money markets perspective
    def simulate_domestic(self, S0, Q0, r, rf, alpha, gamma, sigma1, sigma2, rho):
        M0 = exp(r)
        D0 = 1 / M0
        dW1, dW2, dW3 = self._generate_shocks(rho)
        S, Q, x, y = [np.zeros(self.N + 1) for _ in range(4)]
        S[0], Q[0], y[0], x[0] = S0, Q0, D0 * S0, M0 * Q0

        for t in range(1, self.N + 1):
            S[t] = S[t - 1] + alpha * self.dt + sigma1 * dW1[t - 1]
            Q[t] = Q[t - 1] + (r - rf + gamma) * self.dt + sigma2 * dW3[t - 1]
            y[t] = y[t - 1] * exp(sigma1 * dW1[t - 1])
            x[t] = x[t - 1] * exp(sigma2 * dW3[t - 1])
        return S, Q, x, y

    #simulate foreign money markets perspective
    def simulate_foreign(self, S0, Q0, rf, alpha, gamma, sigma1, sigma2, rho):
        M0 = exp(rf)
        D0 = 1 / M0
        dW1, dW2, dW3 = self._generate_shocks(rho)

        Sf, Qf, xf, yf = [np.zeros(self.N + 1) for _ in range(4)]
        Sf[0], Qf[0] = S0, Q0
        xf[0] = D0 * M0 / Q0
        yf[0] = D0 * S0 / Q0

        for t in range(1, self.N + 1):
            Sf[t] = Sf[t - 1] + alpha * self.dt + sigma1 * dW1[t - 1]
            Qf[t] = Qf[t - 1] + gamma * self.dt + sigma2 * dW3[t - 1]
            # foreign measure dynamics
            xf[t] = xf[t - 1] * exp(-sigma2 * dW3[t - 1])
            yf[t] = yf[t - 1] * exp(sigma1 * dW1[t - 1] - sigma2 * dW3[t - 1])
        return Sf, Qf, xf, yf

    #simulate sharpe ratio
    def simulate_risk(self, r, rf, sigma2, rho):
        dW1, _, _ = self._generate_shocks(rho)
        theta = np.zeros(self.N + 1)
        rets = (r - rf - 0.5 * sigma2 ** 2)
        vols = sigma2
        for t in range(1, self.N + 1):
            dTheta = (rets / vols) * self.dt + dW1[t - 1]
            theta[t] = theta[t - 1]+dTheta
        return theta, rets, vols

    #call put via Garman-Kohlagen model
    def Garman_Kohlage_call(self, rf, r, sigma2, Q0, K, return_d_parameters=False):
        T = self.T
        d1 = (log(Q0 / K) + (r - rf - 0.5 * sigma2 ** 2) * T) / (sigma2 * sqrt(T))
        d2 = d1  - sigma2 * sqrt(T)
        price = exp(-rf) * Q0 * stats.norm.cdf(-d1) - K * exp(-r * T) * stats.norm.cdf(-d2)
        d_data = [price,d1,d2]
        index = ['price : ', 'd1 : ', 'd2 : ']
        data = price
        if return_d_parameters:
            return pd.DataFrame(data=d_data,index=index)
        return pd.DataFrame(data=data)

    #plotting and results
    def plot_forex_analysis(self, S0, Q0, r, rf, gamma, alpha, sigma1, sigma2, rho):
        S, Q, x, y = self.simulate_domestic(S0, Q0, r, rf, alpha, gamma, sigma1, sigma2, rho)
        Sf, Qf, xf, yf = self.simulate_foreign(S0, Q0, rf, alpha, gamma, sigma1, sigma2, rho)
        spread = log(S) - log(Q)
        fx_logrets = np.diff(np.log(S / Q))
        t_idx = np.arange(len(fx_logrets))
        X = sm.add_constant(t_idx)
        ols = sm.OLS(fx_logrets,X).fit(cov_type="HAC",cov_kwds={"maxlags": 5})
        fx_hat = ols.fittedvalues
        resid = ols.resid
        rmse = np.sqrt(np.mean(resid ** 2))
        r2 = ols.rsquared

        theta, rets_base, vols_base = self.simulate_risk(r, rf, sigma2, rho)
        rets, vols = [], []
        for _ in range(self.N):
            _, ri, vi = self.simulate_risk(r, rf, np.random.uniform(0.01, 0.2), np.random.uniform(0.1, 0.9))
            rets.append(ri); vols.append(vi)

        fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(14, 10),)
        ax = axes.flatten()

        ax[0].plot(self.t_axis,S,label='Domestic (S)')
        ax[0].plot(self.t_axis,Q,label='Foreign (Q)')
        ax[0].set_title("Individual Price Processes")
        ax[0].legend()

        ax[1].plot(self.t_axis,y,label='DMM-Dom')
        ax[1].plot(self.t_axis,x,label='DMM-For')
        ax[1].plot(self.t_axis,yf,label='FMM-Dom',linestyle='dashdot')
        ax[1].plot(self.t_axis,xf,label='FMM-For',linestyle='dashdot')
        ax[1].set_title("Discounted Dynamics")
        ax[1].legend()

        ax[2].plot(self.t_axis,spread,color='black',label='Log-Spread')
        ax[2].axhline(np.mean(spread),color='red',linestyle='--',alpha=0.5)
        ax[2].set_title("Currency Spread")
        ax[2].legend()

        ax[3].scatter(t_idx,resid,alpha=0.5,color="purple")
        ax[3].axhline(0,color="black",linestyle=":")
        ax[3].axhline(np.mean(resid),color="black",linestyle="-")
        ax[3].set_title("OLS Residuals")
        ax[3].set_xlabel("Time index")
        ax[3].set_ylabel("Residual")

        ax[4].plot(self.t_axis, theta, color='purple', label='Theta (Risk-Adj)')
        ax[4].set_title("FOREX Logarithmic Risk Time-Series")
        ax[4].legend()

        sc = ax[5].scatter(rets, vols, c=np.array(rets) / np.array(vols),
                           cmap='viridis', marker='o')
        ax[5].set_title("FOREX Frontier")
        ax[5].set_xlabel("Returns")
        ax[5].set_ylabel("Volatility")
        fig.colorbar(sc, ax=ax[5], label='Log Risk', shrink=0.8, pad=0.02)

        ax[6].hist(rets, bins=20, alpha=0.5, label='Returns', color='blue')
        ax[6].set_title("Distribution of Returns")
        ax[6].legend()

        ax[7].hist(vols, bins=20, alpha=0.5, label='Volatility', color='red')
        ax[7].set_title("Distribution of  Volatility")
        ax[7].legend()

        for a in ax: a.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
