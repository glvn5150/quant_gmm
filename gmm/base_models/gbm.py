# gmm/base_models/gbm.py
import numpy as np
import matplotlib.pyplot as plt
from numpy import log, exp, sqrt

class GeometricBrownianMotion:
    def __init__(self, T=1.0, N=252, S0=1.0, mu=0.05, sigma=0.2):
        self.T = T
        self.N = N
        self.dt = T / N
        self.t_axis = np.linspace(0, T, N + 1)
        self.S0 = float(S0)
        self.mu = float(mu)
        self.sigma = float(sigma)

    def simulate(self, paths=1000, seed=None):
        if seed is not None:
            np.random.seed(seed)

        S = np.zeros((paths, self.N + 1))
        S[:, 0] = self.S0

        for t in range(self.N):
            z = np.random.randn(paths)
            S[:, t + 1] = S[:, t] * np.exp((self.mu - 0.5 * self.sigma**2) * self.dt + self.sigma * sqrt(self.dt) * z)
        return S

    def log_returns(self, S):
        return np.diff(np.log(S), axis=1)

    def plot_gbm_analysis(self, paths=1000, seed=42):
        np.random.seed(seed)

        S = self.simulate(paths=paths, seed=seed)
        logrets = self.log_returns(S)
        vols = np.std(logrets, axis=1) / sqrt(self.dt)
        mean_path = S.mean(axis=0)
        low = np.percentile(S, 5, axis=0)
        high = np.percentile(S, 95, axis=0)
        fig, ax = plt.subplots(2, 3, figsize=(18, 10))
        axs = ax.flatten()
        axs[0].plot(self.t_axis, S.T, lw=0.6, alpha=0.25, color="gray")
        axs[0].plot(self.t_axis, mean_path, lw=2.5, color="black", label="Mean")
        axs[0].set_title("GBM Price Paths")
        axs[0].legend()
        axs[1].hist(logrets.flatten(), bins=60, density=True,color="steelblue", edgecolor="black")
        axs[1].set_title("Distribution of Log Returns")
        axs[2].hist(vols, bins=40,color="salmon", edgecolor="black")
        axs[2].set_title("Realized Volatility Distribution")
        axs[3].plot(self.t_axis, mean_path, lw=2, color="black")
        axs[3].fill_between(self.t_axis, low, high,alpha=0.3, color="orange")
        axs[3].set_title("Mean Price ± 90% CI")
        # --- Expected log-price
        theo_mean = self.S0 * np.exp(self.mu * self.t_axis)
        axs[4].plot(self.t_axis, theo_mean, lw=2,color="purple", label="E[S_t]")
        axs[4].legend()
        axs[4].set_title("Expected GBM Path")
        axs[5].axis("off")
        for a in axs:
            a.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
