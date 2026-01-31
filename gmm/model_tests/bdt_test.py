import numpy as np
import yfinance as yf
from gmm.base_models.bdt import BlackDermanToy


tickers = {1: "^IRX",2: "2YY=F",5: "^FVX",10: "^TNX",30: "^TYX"}
prices = yf.download(list(tickers.values()),start="2020-01-01",end="2025-01-11",progress=False,auto_adjust=True)["Close"]
yields_raw = {T: float(prices[ticker].iloc[-1])/100 for T, ticker in tickers.items()}
vols_raw = {T: float(prices[ticker].pct_change(fill_method=None).std()) for T, ticker in tickers.items()}
all_maturities = np.arange(1, 31)
interp_yields = np.interp(all_maturities,list(yields_raw.keys()),list(yields_raw.values()))
interp_vols = np.interp(all_maturities,list(vols_raw.keys()),list(vols_raw.values()))
quotes = {int(m): [y, v] for m, y, v in zip(all_maturities, interp_yields, interp_vols)}
bdt_model = BlackDermanToy(quotes=quotes,maturity=30,verbose=True)
bdt_model.plot_tree(paths=300, dt=1.0,spots=None,seed=42,)
