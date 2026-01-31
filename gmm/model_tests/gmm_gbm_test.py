# gmm/model_tests/gmm_gbm_test.py

import numpy as np
import pandas as pd
import yfinance as yf

from gmm.base_models.gbm import GeometricBrownianMotion
from gmm.models.gbm_gmm import GBM_GMM

tickers = [
    "ADRO.JK","ANTM.JK","ASII.JK","BBCA.JK","BBNI.JK","BBRI.JK","BBTN.JK","BMRI.JK",
    "BRPT.JK","GGRM.JK","CPIN.JK","EMTK.JK","EXCL.JK","ICBP.JK","INCO.JK","INDF.JK",
    "INKP.JK","KLBF.JK","MDKA.JK","MIKA.JK","PGAS.JK","PTBA.JK","SMGR.JK","TBIG.JK",
    "TINS.JK","TLKM.JK","TOWR.JK","UNTR.JK","UNVR.JK","WSKT.JK"]

start = "2020-01-01"
end   = "2026-01-01"
dt = 1.0 / 252.0
prices = yf.download(
    tickers,
    start=start,
    end=end,
    progress=False
)["Close"]

logrets = np.log(prices / prices.shift(1)).dropna(how="all")
results = []

for ticker in tickers:
    r = logrets[ticker].dropna()
    if len(r) < 300:
        continue
    try:
        gmm = GBM_GMM(dt=dt)
        gmm.fit(r.values)
        mu_hat, sigma_hat = gmm.params_
        results.append({
            "ticker": ticker,
            "mu_hat": mu_hat,
            "sigma_hat": sigma_hat,
            "J_stat": gmm.jstat_,
            "obs": len(r)
        })
    except Exception as e:
        print(f"[skip] {ticker}: {e}")

df_results = (
    pd.DataFrame(results)
    .set_index("ticker")
    .sort_values("sigma_hat", ascending=False)
)
print("\n=== GBM–GMM results (IDX equities) ===")
print(df_results.round(6))
rep = df_results["sigma_hat"].sort_values().index[len(df_results)//2]
print(f"\nDiagnostic plot for representative ticker: {rep}")

S = prices[rep].dropna()
S0 = S.iloc[-1] / S.iloc[0]

model = GeometricBrownianMotion(
    T=1.0,
    N=252,
    S0=S0,
    mu=df_results.loc[rep, "mu_hat"],
    sigma=df_results.loc[rep, "sigma_hat"])
model.plot_gbm_analysis(
    paths=1500,
    seed=42)
