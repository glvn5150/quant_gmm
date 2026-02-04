from gmm.base_models.foreign_domestic_pricing import ForexModel as fm
from gmm.models.foreign_domestic_gmm import ForeignDomestic_GMM as fm_gmm
import yfinance as yf
import pandas as pd
import numpy as np
from numpy import sqrt,log

tickers = ["IDR=X", "EURUSD=X"]
df = yf.download(tickers, start="2020-01-01", end="2025-12-01")['High']
df_norm = df/df.iloc[0]
ret= log(df/df.shift(1)).dropna()
S0, Q0 = df_norm["IDR=X"].iloc[-1],df_norm["EURUSD=X"].iloc[-1]
sigma1 = ret["IDR=X"].std()*sqrt(252)
sigma2 = ret["EURUSD=X"].std()*sqrt(252)
rho = ret.corr().iloc[0,1]
mu1 = ret["IDR=X"].mean()*252
mu2 = ret["EURUSD=X"].mean()*252
alpha = mu1-0.5*(sigma1**2)
gamma = mu2-0.5*(sigma2**2)
r = 0.0475
rf = 0.0375
K = 1

model = fm(T=1, N=len(df))
model_gmm = fm_gmm(r=r, rf=rf, dt=1/252, verbose=True)
params = model_gmm.fit_gmm(S_series=df["IDR=X"], Q_series=df["EURUSD=X"])
call = model.Garman_Kohlage_call(rf, r, sigma2, Q0, K, True)
print(call)
model.plot_forex_analysis(S0=S0,Q0=Q0,r=r,rf=rf,gamma=gamma,alpha=alpha,sigma1=sigma1,sigma2=sigma2,rho=rho)
