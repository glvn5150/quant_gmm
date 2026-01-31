import numpy as np
import pandas as pd
from gmm.models.bsm_gmm import BSM_GMM_estimator
from gmm.models.bsm_gmm import call_option_price, put_option_price
import plotly.graph_objects as go

#--sample data
S0 = 100
E = 100
T = 1
rf = 0.05
sigma =0.2

call = call_option_price(S0,E,T,rf,sigma,return_d_parameters=True)
put = put_option_price(S0,E,T,rf,sigma,return_d_parameters=True)


if __name__ == '__main__':

    data = pd.DataFrame({
        'S': [100, 100, 100, 105, 95],
        'E': [100, 105, 95, 100, 100],
        'T': [1.0, 1.0, 1.0, 0.5, 0.5],
        'rf': [0.05, 0.05, 0.05, 0.05, 0.05],
        'market_price': [10.45, 8.12, 14.50, 8.50, 4.20],  # Example market prices
        'type': ['call', 'call', 'put', 'call', 'put']
    })

    gmm_bsm = BSM_GMM_estimator()
    gmm_bsm.fit(data)

    data_no_market = data.drop(columns=['market_price'])
    print("data - excld. market:")
    print(data_no_market)

    estimated_sigma = gmm_bsm.estimated_sigma
    print("estimated volatility : ", estimated_sigma)

    d_table = gmm_bsm.compute_d1_d2(data_no_market)
    print("diagnostics:")
    print(d_table)

















