import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from numpy import sqrt,log,exp
from scipy.interpolate import PchipInterpolator

class Vasicek_HoLee:
    def __init__ (self,T=1,N=100,r0=0.05,sigma=0.02,a=0.5,theta=0.05,lambda0=0,lambda1=0):
        self.N = N
        self.T = T
        self.dt = T / N
        self.t_axis = np.linspace(0,T,N + 1)
        self.r0 = r0
        self.sigma = sigma
        self.a = a
        self.theta = theta
        self.lambda0 = lambda0
        self.lambda1 = lambda1

    def risk_neutral_params (self):
        # Adjusted for Vasicek Risk Neutrality
        kappa_q = self.a
        theta_q = self.theta - (self.sigma * self.lambda0) / self.a if self.a != 0 else self.theta
        return kappa_q,theta_q

    def fit_vasicek_ols (self,historical_rates):
        y = np.diff(historical_rates)
        x = historical_rates[:-1]
        x_with_constant = sm.add_constant(x)
        res = sm.OLS(y,x_with_constant).fit()
        alpha,beta = res.params
        self.a = -beta / self.dt
        self.theta = alpha / (self.a * self.dt)
        return {"a": self.a,"theta": self.theta,"half_life": log(2) / self.a}

    def get_A_B (self,t,T_mat,model="vasicek"):
        tau = T_mat - t
        if tau <= 0: return 1.0,0.0
        kappa_q,theta_q = self.risk_neutral_params()
        if model.lower() == "ho-lee":
            B = tau
            ln_A = (self.theta * (tau ** 2) / 2) - (self.sigma ** 2 * (tau ** 3) / 6)
            return exp(ln_A),B
        elif model.lower() == "vasicek":
            B = (1 - exp(-self.a * tau)) / self.a
            term1 = (B - tau) * (self.a ** 2 * theta_q - self.sigma ** 2 / 2) / (self.a ** 2)
            term2 = (self.sigma ** 2 * B ** 2) / (4 * self.a)
            return exp(term1 - term2),B
        else:
            raise ValueError("Model type not recognized")

    def simulate(self,model="vasicek",paths=10):
        shocks = np.random.normal(0,sqrt(self.dt),(paths,self.N))
        r = np.zeros((paths,self.N + 1))
        r[:,0] = self.r0

        for t in range(self.N):
            if model.lower() == "ho-lee":
                drift = self.theta * self.dt
            else:
                drift = self.a * (self.theta - r[:,t]) * self.dt
            r[:,t + 1] = r[:,t] + drift + self.sigma * shocks[:,t]
        return r

    def yield_curve (self,r_t,model="vasicek"):
        tenors = np.linspace(0.1,self.T + 2,self.N)  # Extended for visualization
        yields = np.zeros_like(tenors)
        for i,tau in enumerate(tenors):
            A,B = self.get_A_B(0,tau,model)
            P = A * exp(-B * r_t)
            yields[i] = -log(P) / tau
        return tenors,yields

    def plot_combined_view (self,spots_df=None):
        ho_paths = self.simulate("ho-lee",paths=50)
        va_paths = self.simulate("vasicek",paths=50)
        returns = np.diff(va_paths,axis=1).flatten()
        vol = np.std(np.diff(va_paths,axis=1),axis=1) / sqrt(self.dt)
        tenors,y_model = self.yield_curve(self.r0,"vasicek")

        fig,ax = plt.subplots(2,3,figsize=(18,10))
        axs = ax.flatten()
        axs[0].plot(self.t_axis,ho_paths.T,lw=0.6,alpha=0.3,color="gray")
        axs[0].plot(self.t_axis,ho_paths.mean(axis=0),lw=2,color="blue",label="Mean")
        axs[0].set_title("Ho–Lee Paths")
        axs[1].plot(self.t_axis,va_paths.T,lw=0.6,alpha=0.3,color="gray")
        axs[1].plot(self.t_axis,va_paths.mean(axis=0),lw=2,color="red",label="Mean Diffusion")
        axs[1].set_title("Vasicek Paths (Mean Reverting)")
        axs[2].hist(returns,bins=50,density=True,color='skyblue',edgecolor='black')
        axs[2].set_title("Distribution of dR (Vasicek)")
        axs[3].hist(vol,bins=20,color='salmon',edgecolor='black')
        axs[3].set_title("Realized Volatility Distribution")

        # 5. Yield curve comparison
        axs[4].plot(tenors,y_model,lw=3,label="Vasicek Model")
        if spots_df is not None:
            market_interp = PchipInterpolator(spots_df.index,spots_df.values)
            market_yields_smooth = market_interp(tenors)
            axs[4].plot(tenors,market_yields_smooth,lw=2,linestyle='--',
                       color='red',label="Market (Interpolated)")
            axs[4].scatter(spots_df.index,spots_df.values,color='red',s=30,zorder=5)
        axs[4].set_title("Yield Curve (Analytical)")
        axs[4].legend()

        mc_yields = []
        for r_end in va_paths[:,-1]:
            _,y = self.yield_curve(r_end,"vasicek")
            mc_yields.append(y)
        mc_yields = np.array(mc_yields)

        low = np.percentile(mc_yields,5,axis=0)
        high = np.percentile(mc_yields,95,axis=0)

        axs[5].plot(tenors,y_model,lw=2,color='black',label="Current Curve")
        axs[5].fill_between(tenors,low,high,alpha=0.3,color='orange',label="90% Forward CI")
        axs[5].set_title("Yield Curve Forward Uncertainty")
        axs[5].legend()

        plt.tight_layout()
        plt.show()

