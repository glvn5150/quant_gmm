import yfinance as yf
import pandas as pd

from gmm.models.gmm_base import linear_sdf_gmm


tickers = ["^JKSE", "BBCA.JK", "TLKM.JK", "ASII.JK", "UNVR.JK"]
prices = yf.download(tickers, start="2018-01-01", interval="1mo", auto_adjust=True)['Close']

R = prices.pct_change().dropna()

market = R["^JKSE"]
vol = R.rolling(6).std()["^JKSE"]
F = pd.DataFrame({"MKT": market, "VOL":vol}).dropna()

R = R.loc[F.index]

model = linear_sdf_gmm(nw_lags=4)
model.fit(R, F)

print("b_hat:", model.results_.params)
print("J-stat:", model.results_.jstat)
print("pricing errors:")
print(model.results_.pricing_errors)