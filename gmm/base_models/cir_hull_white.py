import numpy as np
import matplotlib.pyplot as plt
from numpy import sqrt, exp, log
import statsmodels.api as sm

class CIR_Hull_White:
    def __init__(self, T=1.0, N=100, r0=0.03, sigma=0.05, a=0.5,
                 theta=0.03, theta_fn=None, lambda0=0.0):
        self.N = N
        self.T = T
        self.dt = T / N
        self.t_axis = np.linspace(0, T, N + 1)
        self.r0 = float(r0)
        self.sigma = float(sigma)
        self.a = float(a)
        self.theta = float(theta)
        self.theta_fn = theta_fn
        self.lambda0 = lambda0

    def theta_at(self, t_idx):
        t = self.t_axis[t_idx]
        if self.theta_fn is not None:
            return float(self.theta_fn(t))
        else:
            return self.theta

    def risk_neutral_params(self):
        kappa_q = self.a
        theta_q = self.theta - (self.sigma * self.lambda0) / self.a if self.a != 0 else self.theta
        return kappa_q, theta_q

    def simulate(self, model='cir-hw', paths=1000, method='full_truncation', random_seed=None):
        if random_seed is not None:
            np.random.seed(random_seed)

        dt = self.dt
        N = self.N
        a = self.a
        sigma = self.sigma

        r = np.zeros((paths, N + 1))
        r[:,0] = self.r0

        if method == 'exact' and self.theta_fn is None:
            df = 4.0 * a * self.theta / (sigma ** 2)
            for t in range(N):
                exp_kdt = np.exp(-a * dt)
                c = (sigma**2*(1 - exp_kdt)) / (4.0 * a)
                lam = (4.0*a*exp_kdt) / (sigma**2*(1 - exp_kdt))*r[:, t]
                x = np.random.noncentral_chisquare(df, lam, size=paths)
                r[:,t+1] = c*x
            r[r < 0] = 0.0
            return r
        for t in range(N):
            theta_t = self.theta_at(t)
            z = np.random.normal(0.0, 1.0, size=paths)
            sqrt_rt = np.sqrt(np.maximum(r[:, t], 0.0))
            drift = a * (theta_t - r[:, t]) * dt
            diffusion = sigma * sqrt_rt * np.sqrt(dt) * z
            r_next = r[:, t] + drift + diffusion
            r[:, t + 1] = np.maximum(r_next, 0.0)
        return r

    def get_A_B(self, t, T_mat):
        tau = float(T_mat - t)
        if tau <= 0:
            return 1.0, 0.0

        if self.theta_fn is not None:
            raise NotImplementedError("closed forms A/B for time-dependent theta's not used. use bond_price_mc to price numerically")
        kappa_q, theta_q = self.risk_neutral_params()
        kappa = kappa_q
        sigma = self.sigma
        if sigma == 0:
            B = tau
            A = exp(-kappa * theta_q * (tau ** 2) / 2.0)
            return A, B

        gamma = np.sqrt(kappa ** 2 + 2.0 * sigma ** 2)
        exp_gt = np.exp(gamma * tau)

        numerator_B = 2.0 * (exp_gt - 1.0)
        denom_B = (gamma + kappa) * (exp_gt - 1.0) + 2.0 * gamma
        B = numerator_B / denom_B
        term = (2.0 * gamma * np.exp((kappa + gamma) * tau / 2.0)) / denom_B
        power = 2.0 * kappa * theta_q / (sigma ** 2)
        A = term ** power
        return A, B

    def bond_price_mc(self, t, T_mat, paths=5000, method='full_truncation', random_seed=None):
        if T_mat <= t:
            return 1.0

        if self.theta_fn is None:
            A, B = self.get_A_B(t, T_mat)
            return A * np.exp(-B * self.r0)

        if random_seed is not None:
            np.random.seed(random_seed)

        horizon = T_mat - t
        N_steps = max(2, int(np.ceil(horizon / self.dt)))
        dt_local = horizon / N_steps

        # simulate with local euler (full truncation)
        r = np.ones((paths, N_steps + 1)) * self.r0
        for i in range(N_steps):
            theta_i = self.theta_fn(t + i * dt_local)
            z = np.random.normal(0.0, 1.0, size=paths)
            sqrt_rt = np.sqrt(np.maximum(r[:, i], 0.0))
            drift = self.a * (theta_i - r[:, i]) * dt_local
            diffusion = self.sigma * sqrt_rt * np.sqrt(dt_local) * z
            r_next = r[:, i] + drift + diffusion
            r[:, i + 1] = np.maximum(r_next, 0.0)

        # approximate integral by trapezoid / rectangle (simple sum)
        integral_approx = np.sum(r[:, :-1] * dt_local, axis=1)  # left Riemann
        discounts = np.exp(-integral_approx)
        price = discounts.mean()
        return float(price)

    def yield_curve(self, r_t=None, model='cir', tenors=None, mc_for_hw=True):
        if r_t is None:
            r_t = self.r0
        if tenors is None:
            tenors = np.linspace(0.25, self.T + 2.0, self.N)
        yields = np.zeros_like(tenors)
        for i, tau in enumerate(tenors):
            T_mat = tau
            if (self.theta_fn is None) and (model.lower() == 'cir'):
                A, B = self.get_A_B(0.0, T_mat)
                P = A * np.exp(-B * r_t)
                yields[i] = -log(P) / tau
            else:
                P = self.bond_price_mc(0.0, T_mat, paths=2000, method='full_truncation')
                yields[i] = -log(P) / tau
        return tenors, yields

    def plot_combined_view (self,spots_df=None,paths=1000,method="full_truncation",seed=42):
        np.random.seed(seed)
        r_paths = self.simulate(paths=paths,method=method,random_seed=seed)
        r_paths = r_paths.T
        dt = self.dt
        t_axis = self.t_axis
        dr = np.diff(r_paths,axis=0)
        vols = np.std(dr,axis=0) / np.sqrt(dt)
        tenors,y_model = self.yield_curve()
        avg_yields = np.cumsum(r_paths,axis=0) / np.arange(
            1,r_paths.shape[0] + 1
        )[:,None]

        low = np.percentile(avg_yields,5,axis=1)
        high = np.percentile(avg_yields,95,axis=1)
        fig,ax = plt.subplots(2,3,figsize=(18,10))
        axs = ax.flatten()
        axs[0].plot(t_axis,r_paths,lw=0.6,alpha=0.25,color="gray")
        axs[0].plot(t_axis,r_paths.mean(axis=1),lw=2.5,color="black",label="Mean")
        axs[0].set_title("CIR–Hull–White Short-Rate Paths")
        axs[0].set_xlabel("Time")
        axs[0].set_ylabel("r(t)")
        axs[0].legend()
        axs[1].hist(dr.flatten(),bins=60,density=True,
                    color="steelblue",edgecolor="black")
        axs[1].set_title("Distribution of Δr")
        axs[2].hist(vols,bins=40,
                    color="salmon",edgecolor="black")
        axs[2].set_title("Realized Volatility Distribution")
        axs[3].plot(tenors,y_model,lw=3,label="CIR–HW Model")
        if spots_df is not None:
            axs[3].scatter(spots_df.index,spots_df.values,color="red",zorder=5,label="Market")
            axs[3].legend()
        axs[3].set_title("Yield Curve")
        axs[4].plot(t_axis,r_paths.mean(axis=1),lw=2,color="black",label="Mean Short Rate")
        axs[4].fill_between(t_axis,low,high,alpha=0.3,color="orange",label="90% CI")
        axs[4].legend()
        axs[4].set_title("Forward Rate Uncertainty")
        axs[5].axis("off")

        plt.tight_layout()
        plt.show()
