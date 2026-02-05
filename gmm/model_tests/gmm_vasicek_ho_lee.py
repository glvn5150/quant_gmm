import numpy as np
import pandas as pd
import yfinance as yf
from gmm.base_models.vasicek_ho_lee import Vasicek_HoLee
from gmm.models.vasicek_ho_lee_gmm import VasicekHoLee_GMM

tickers_map = {"^IRX": 0.25,"^FVX": 5.0,"^TNX": 10.0,"^TYX": 30.0}

data = yf.download(list(tickers_map.keys()),start="2020-01-01",end="2026-01-20",progress=False)
prices = data["Close"]
spot_rates = {maturity: prices[ticker].dropna().iloc[-1] / 100.0 for ticker, maturity in tickers_map.items()}

spot_rates_series = pd.Series(spot_rates).sort_index()

maturities = spot_rates_series.index.values
spot_yields = spot_rates_series.values

historical_rates = prices["^TNX"].dropna().values / 100.0
r0 = historical_rates[-1]

vas_ols = Vasicek_HoLee(T=30, N=300, r0=r0)
vas_ols.dt = 1.0 / 252.0
ols_params = vas_ols.fit_vasicek_ols(historical_rates)

vas_ols.dt = vas_ols.T / vas_ols.N
vas_ols.t_axis = np.linspace(0, vas_ols.T, vas_ols.N + 1)

if vas_ols.a <= 0:
    print("negative mean reversion detected — forcing a = 0.1")
    vas_ols.a = 0.1

print("\n=== OLS Vasicek calibration ===")
for k, v in ols_params.items():
    print(f"{k:>12}: {v:.4f}")

gmm = VasicekHoLee_GMM(
    maturities=maturities,
    spot_rates=spot_yields,
    model_type="vasicek"
)

gmm.fit(estimate_sigma=True)

print("\n=== GMM Vasicek–Ho–Lee calibration ===")
print(f"{'theta_hat':>12}: {gmm.theta_hat:.6f}")
print(f"{'sigma_hat':>12}: {gmm.sigma_hat:.6f}")

tenors = np.linspace(0.25, 30.0, 200)

# OLS-implied curve
y_ols = []
for T in tenors:
    A, B = vas_ols.get_A_B(0.0, T, model="vasicek")
    P = A * np.exp(-B * r0)
    y_ols.append(-np.log(P) / T)

# gmm_implied_curve
vas_gmm = Vasicek_HoLee(
    T=30,
    N=300,
    r0=r0,
    theta=gmm.theta_hat,
    sigma=gmm.sigma_hat
)

y_gmm = []
for T in tenors:
    A, B = vas_gmm.get_A_B(0.0, T, model="vasicek")
    P = A * np.exp(-B * r0)
    y_gmm.append(-np.log(P) / T)

df_curves = pd.DataFrame({
    "tenor": tenors,
    "ols": y_ols,
    "gmm": y_gmm
})

print("\n=== Curve diagnostics ===")
print("mean |ols − gmm|:",
      np.mean(np.abs(df_curves["ols"] - df_curves["gmm"])))

print("Max  |ols − gmm|:",
      np.max(np.abs(df_curves["ols"] - df_curves["gmm"])))

vas_ols.plot_combined_view(
    spots_df=spot_rates_series
)
