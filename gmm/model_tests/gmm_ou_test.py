import numpy as np
import pandas as pd
import yfinance as yf
from gmm.base_models.general_ou import OU
from gmm.models.ou_gmm import OU_GMM

# -------------------------------------------------
# Download equity data
# -------------------------------------------------
tickers = """ADRO.JK ANTM.JK ASII.JK BBCA.JK BBNI.JK BBRI.JK BBTN.JK BMRI.JK
BRPT.JK GGRM.JK
CPIN.JK EMTK.JK EXCL.JK ICBP.JK INCO.JK INDF.JK INKP.JK KLBF.JK
MDKA.JK MIKA.JK
PGAS.JK PTBA.JK SMGR.JK TBIG.JK TINS.JK TLKM.JK TOWR.JK UNTR.JK
UNVR.JK WSKT.JK"""

df = yf.download(tickers,start="2020-01-01",end="2026-01-01",progress=False)["Close"]

df = df.dropna()
px = df["ADRO.JK"]
logP = np.log(px)
rets = logP.diff().dropna()

X = logP - logP.mean()

dt = 1 / 252
kappa = 0.5
theta = 0.0
sigma = rets.std() * np.sqrt(252)

ou = OU(X0=X.iloc[-1],kappa=kappa,theta=theta,sigma=sigma,dt=dt)
ou_gmm = OU_GMM(dt=dt, verbose=True)
ou_gmm.fit(X.values)
ou.plot_diagnostics(T=len(X),paths=600,seed=42)

