import numpy as np
import matplotlib.pyplot as plt

class OU:
    """
    General Ornstein–Uhlenbeck process:
        dX_t = kappa (theta - X_t) dt + sigma dW_t
    """
    def __init__(self, X0=0.0, kappa=0.5, theta=0.0, sigma=0.1, dt=1/252):
        self.X0 = float(X0)
        self.kappa = float(kappa)
        self.theta = float(theta)
        self.sigma = float(sigma)
        self.dt = float(dt)

    def conditional_mean(self, Xt):
        phi = np.exp(-self.kappa * self.dt)
        return self.theta + phi * (Xt - self.theta)

    def conditional_var(self):
        phi = np.exp(-self.kappa * self.dt)
        return self.sigma**2 * (1 - phi**2) / (2 * self.kappa)

    def simulate(self, T, paths=1, seed=None):
        if seed is not None:
            np.random.seed(seed)

        phi = np.exp(-self.kappa * self.dt)
        var = self.conditional_var()
        std = np.sqrt(var)

        X = np.zeros((paths, T))
        X[:, 0] = self.X0

        for p in range(paths):
            for t in range(1, T):
                X[p, t] = (self.theta + phi * (X[p, t - 1] - self.theta) + std * np.random.randn())
        return X if paths > 1 else X[0]

    def plot_diagnostics(self, T=500, paths=500, seed=42):
        np.random.seed(seed)

        X = self.simulate(T=T, paths=paths, seed=seed)
        if X.ndim == 1:
            X = X[None, :]

        dt = self.dt
        t_axis = np.arange(T) * dt

        dX = np.diff(X, axis=1)
        vols = np.std(dX, axis=1) / np.sqrt(dt)

        mean_path = X.mean(axis=0)
        low = np.percentile(X, 5, axis=0)
        high = np.percentile(X, 95, axis=0)

        fig, ax = plt.subplots(2, 3, figsize=(18, 10))
        axs = ax.flatten()

        # --- diffusion paths ---
        axs[0].plot(t_axis, X.T, lw=0.6, alpha=0.25, color="gray")
        axs[0].plot(t_axis, mean_path, lw=3, color="black", label="Mean")
        axs[0].axhline(self.theta, color="red", linestyle="--", label="θ")
        axs[0].set_title("OU Diffusion Paths")
        axs[0].legend()

        axs[1].hist(dX.flatten(), bins=60, density=True,
                    color="steelblue", edgecolor="black")
        axs[1].set_title("Distribution of ΔX")

        axs[2].hist(vols, bins=40, color="salmon", edgecolor="black")
        axs[2].set_title("Realized Volatility Distribution")

        axs[3].scatter(X[:, :-1].flatten(), dX.flatten(),alpha=0.15, color="purple")
        axs[3].axhline(0, color="black", linestyle=":")
        axs[3].set_title("Mean-Reversion Signature")

        axs[4].plot(t_axis, mean_path, lw=2.5, color="black")
        axs[4].fill_between(t_axis, low, high, alpha=0.3,
                            color="orange", label="90% CI")
        axs[4].legend()
        axs[4].set_title("OU Mean Path ± CI")

        axs[5].axis("off")

        for a in axs:
            a.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()
