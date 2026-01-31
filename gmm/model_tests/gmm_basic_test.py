from gmm.models import gmm_base as md
import pandas as pd
import numpy as np

np.random.seed(0)

T = 500
N = 5
K=2

F = pd.DataFrame(np.random.randn(T,K), columns=["f1","f2"])
true_b = np.array([0.4,-0.2])

m = 1- F@ true_b
R = pd.DataFrame(
    m.values[:, None] + 0.1*np.random.randn(T,N),
    columns=[f"R(i)" for i in range(N)])

model = md.linear_sdf_gmm(nw_lags=4)
model.fit(R, F)

print("estimated b :", model.results_.params)
print("std err :", model.results_.se)
print("std err :", model.results_.jstat, "df", model.results_.df)
print("pricing errors:", model.results_.pricing_errors)
