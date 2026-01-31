import numpy as np
import pandas as pd
import yfinance as yf
from gmm.models.cir_hull_white_gmm import CIR_Hull_White_GMM
from gmm.base_models.cir_hull_white import CIR_Hull_White

tickers_map = {"^IRX": 0.25,"^FVX": 5.0,"^TNX": 10.0,"^TYX": 30.0}
data = yf.download(list(tickers_map.keys()),start="2020-01-01",end="2026-01-20",progress=False)["Close"]
spot_rates = {maturity: data[ticker].dropna().iloc[-1] / 100.0 for ticker, maturity in tickers_map.items()}
spot_rates_series = pd.Series(spot_rates).sort_index()

spots = pd.Series(spot_rates).sort_index()
maturities = spots.index.values
zcb_prices = np.exp(-spots.values * maturities)
r0 = spots.iloc[0]

model = CIR_Hull_White(T=30,r0=r0)
gmm_model = CIR_Hull_White_GMM(
    maturities=maturities,
    market_zcb_prices=zcb_prices,
    r0=r0,
    T=30,
    N=300,
    verbose=True
)

gmm_model.fit(x0=(0.5, 0.04, 0.08))
model.plot_combined_view(spots_df=spot_rates_series,paths=1500)


model = gmm_model.model_
tenors, y_model = model.yield_curve()
print("\nModel-implied yields:")
for T, y in zip(tenors, y_model):
    print(f"{T:5.2f}y : {y:.4%}")
