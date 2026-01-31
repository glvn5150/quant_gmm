import numpy as np
import pandas as pd
import yfinance as yf
from gmm.base_models.vasicek_ho_lee import Vasicek_HoLee

tickers_map = {
    "^IRX": 0.25,   # 13-week (~0.25y)
    "^FVX": 5.0,    # 5Y
    "^TNX": 10.0,   # 10Y
    "^TYX": 30.0    # 30Y
}

data = yf.download(
    list(tickers_map.keys()),
    start="2020-01-01",
    end="2026-01-20",
    progress=False
)

prices = data["Close"]

spot_rates = {
    maturity: prices[ticker].dropna().iloc[-1] / 100.0
    for ticker, maturity in tickers_map.items()
}

spot_rates_series = pd.Series(spot_rates).sort_index()

historical_rates = prices["^TNX"].dropna().values / 100.0
current_r0 = historical_rates[-1]

model = Vasicek_HoLee(
    T=30,
    N=300,
    r0=current_r0
)
model.dt = 1.0 / 252.0
params = model.fit_vasicek_ols(historical_rates)
model.dt = model.T / model.N
model.t_axis = np.linspace(0, model.T, model.N + 1)

if model.a <= 0:
    print("Warning: negative mean reversion detected. Forcing a = 0.1")
    model.a = 0.1

print("Calibration complete:")
for k, v in params.items():
    print(f"  {k}: {v:.4f}")

model.plot_combined_view(
    spots_df=spot_rates_series
)
