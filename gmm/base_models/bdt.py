import time
from scipy import optimize as opt
import matplotlib.pyplot as plt
import numpy as np

#---Tree---
class Vertex:
    def __init__(self, t: int, i: int, r: float = None):
        self.t = t
        self.i = i
        self.r = r
        self.left = None
        self.right = None

#--- Black-Derman-Toy Model---
class BlackDermanToy:
    def __init__(self, quotes: dict[int, list], maturity: int, verbose=True):
        self.quotes = quotes
        self.maturity = maturity
        self.p = 0.5
        self.tree: list[list[Vertex]] = []
        self.verbose = verbose
        self.zero_rates = {t: v[0] for t, v in quotes.items()}
        self.sigmas = {t: v[1] for t, v in quotes.items()}
        self.zcb_prices = {t: self.zcb_price(v[0], t) for t, v in quotes.items()}

        if self.verbose:
            print("\nTerm structure input:")
            print("T | Rate(%) | ZCB | Vol(%)")
            for t, (r, s) in quotes.items():
                print(f"{t} | {r * 100:.2f} | {self.zcb_prices[t]:.4f} | {s * 100:.2f}")

            start = time.time()
            self.build_tree()
            end = time.time()

            print(f"\nBDT tree built in {(end - start) * 1000:.2f} ms")

    def simulate_lognormal_diffusion (self,r0,theta_fn=None,paths=200,seed=None):
        if seed is not None:
            np.random.seed(seed)
        T = self.maturity
        dt = 1.0
        times = np.arange(T + 1)
        rates = np.zeros((paths,T + 1))
        rates[:,0] = r0
        for p in range(paths):
            ln_r = np.log(r0)
            for t in range(T):
                sigma_t = self.sigmas[t + 1]
                if theta_fn is None:
                    theta_t = 0.5 * sigma_t ** 2  # risk-neutral BDT drift
                else:
                    theta_t = theta_fn(t)
                z = np.random.normal()
                ln_r = ln_r + theta_t * dt + sigma_t * np.sqrt(dt) * z
                rates[p,t + 1] = np.exp(ln_r)
        return times,rates

    def bdt_diffusion_step (self,ln_r,theta,sigma,dt,z):
        return ln_r + theta * dt + sigma * np.sqrt(dt)*z

    def build_tree(self):
        self.tree.append([Vertex(0, 0, self.zero_rates[1])])
        for t in range(1, self.maturity):
            layer = [Vertex(t, i) for i in range(t + 1)]
            prev = self.tree[t - 1]
            self.tree.append(layer)
            for i, v in enumerate(prev):
                v.left = layer[i]
                v.right = layer[i + 1]

            r0 = self.zero_rates[t + 1]
            sigma = self.sigmas[t + 1]
            target_price = self.zcb_prices[t + 1]

            res = opt.root(self._zcb_constraint,r0,args=(t, sigma, target_price),method="lm")
            if not res.success:
                raise RuntimeError(f"Calibration failed at t={t}")

            base_rate = res.x[0]
            dt = 1.0
            sigma_t = sigma
            theta_t = sigma_t ** 2 / 2.0  # standard BDT no-arbitrage drift

            ln_r0 = np.log(base_rate)

            for i,v in enumerate(layer):
                z = (2 * i - t) / np.sqrt(t + 1)  # binomial normalisation
                ln_r = self.bdt_diffusion_step(
                    ln_r0,
                    theta=theta_t,
                    sigma=sigma_t,
                    dt=dt,
                    z=z
                )
                v.r = np.exp(ln_r)

    def _zcb_constraint (self,r0,t,sigma,target):
        if r0 <= 0:
            return 1e6
        dt = 1.0
        theta = sigma ** 2 / 2.0
        ln_r0 = np.log(r0)
        rates = []
        for i in range(t + 1):
            z = (2 * i - t) / np.sqrt(t + 1)
            ln_r = self.bdt_diffusion_step(
                ln_r0,
                theta=theta,
                sigma=sigma,
                dt=dt,
                z=z
            )
            rates.append(np.exp(ln_r))
        prices = np.ones(t + 1)
        for k in range(t,-1,-1):
            for i in range(k + 1):
                if k == t:
                    prices[i] = np.exp(-rates[i])
                else:
                    prices[i] = np.exp(-self.tree[k][i].r) * (
                            self.p * prices[i] + (1 - self.p) * prices[i + 1]
                    )

        return prices[0] - target

    @staticmethod
    def zcb_price(rate, t):
        return np.exp(-rate * t)

    def simulate_paths (self,paths=500,seed=None):
        if seed is not None:
            np.random.seed(seed)
        T = len(self.tree)
        mc = np.zeros((T,paths))
        mc[0,:] = self.tree[0][0].r
        for col in range(paths):
            node_idx = 0
            for t in range(1, self.maturity):
                if np.random.rand() > self.p:
                    node_idx += 1
                mc[t,col] = self.tree[t][node_idx].r

        return mc

    def yield_curve (self):
        tenors = sorted(self.zcb_prices.keys())
        yields = [-np.log(self.zcb_prices[T]) / T for T in tenors]
        return np.array(tenors),np.array(yields)

    def enumerate_tree_paths (self,max_paths=500):
        T = len(self.tree)
        total_paths = 2 ** (T - 1)
        if total_paths > max_paths:
            path_ids = np.linspace(0,total_paths - 1,max_paths).astype(int)
        else:
            path_ids = np.arange(total_paths)
        paths = []
        for pid in path_ids:
            ups = 0
            path = [self.tree[0][0].r]
            for t in range(1,T):
                move = (pid >> (t - 1)) & 1  # 0=down, 1=up
                ups += move
                path.append(self.tree[t][ups].r)
            paths.append(path)
        return np.array(paths).T

    def plot_tree (self,paths=500,dt=1.0,spots=None,seed=42):
        tree_paths = self.enumerate_tree_paths()
        max_paths = 500
        if tree_paths.shape[1] > max_paths:
            idx = np.linspace(0,tree_paths.shape[1] - 1,max_paths).astype(int)
            tree_paths = tree_paths[:,idx]
        x = np.arange(tree_paths.shape[0])
        np.random.seed(seed)
        paths_mc = self.simulate_paths(paths=paths,seed=seed)
        returns = np.diff(np.log(paths_mc), axis=0).flatten()
        vols = returns / np.sqrt(dt)
        tenors,y_model = self.yield_curve()
        path_yields = np.cumsum(paths_mc,axis=0) / np.arange(
            1,len(paths_mc) + 1
        )[:,None]

        low = np.percentile(path_yields,5,axis=1)[tenors - 1]
        high = np.percentile(path_yields,95,axis=1)[tenors - 1]
        fig,ax = plt.subplots(2,3,figsize=(18,10))
        axs = ax.flatten()
        times,diff_paths = self.simulate_lognormal_diffusion( r0=self.tree[0][0].r,paths=paths,seed=seed)
        axs[0].plot(times,diff_paths.T,lw=0.6,alpha=0.3,color="gray")
        axs[0].plot(times,diff_paths.mean(axis=0),lw=2,color="black",label="Mean")
        axs[0].set_title("Lognormal Short-Rate Diffusion")
        axs[0].set_xlabel("Time")
        axs[0].set_ylabel("Short Rate")
        axs[0].legend()
        axs[1].hist(returns,bins=50,density=True,color="skyblue",edgecolor="black")
        axs[1].set_title("distribution of log returns")
        axs[2].hist(vols,bins=20,color="salmon",edgecolor="black")
        axs[2].set_title("Realized Volatility Distribution")
        axs[3].plot(tenors,y_model,lw=3,label="BDT Model")
        if spots is not None:
            axs[3].scatter(spots.index,spots.values,color="red",zorder=5,label="Market")
            axs[3].legend()
        axs[3].set_title("Yield Curve (Tree-Implied)")
        axs[4].plot(tenors,y_model,lw=2,color="black",label="Current Curve")
        axs[4].fill_between(tenors,low,high,alpha=0.3,
                            color="orange",label="90% CI")
        axs[4].legend()
        axs[4].set_title("Forward Yield Uncertainty")

        for t,layer in enumerate(self.tree):
            for i,node in enumerate(layer):
                x0 = t
                y0 = -i + t / 2
                axs[5].scatter(x0,y0,color="black",s=5,zorder=3)
                if t % 2 == 0:
                    axs[5].text(x0,y0 + 0.1,f"{node.r:.2%}",
                                ha="center",va="bottom",
                                fontsize=6.5,rotation=45)
        axs[5].set_title("Black–Derman–Toy Binomial Short-Rate Tree")
        axs[5].set_xlabel("Time Step")
        axs[5].set_ylabel("State")
        axs[5].axis("off")
        axs[5].set_aspect("equal")
        axs[5].invert_yaxis()
        for a in axs:
            a.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()


