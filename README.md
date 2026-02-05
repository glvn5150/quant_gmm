## Quant GMM
# Outline of Rep
This rep is a personal project, for the applications of Generalized Methods of Moments, in the style of John Cochrane to financial data and models. The model stems from his book _Asset Pricing_, and this rep samples some well known diffusion models in finance and applied GMM methods to the moment statistics to those diffusion moments. The folder [GMM](gmm) makes each base models in the folder named [base_models](gmm/base_models) as python class as a GMM model of each of the models moments via the [models](gmm/models) folder. A simple test of the models has been implemented by the use of YahooFinance plugin in python in the folder [model_tests](gmm/model_tests). Example applications to empirical data are present in the jupyter notebooks with initials ex. in the [main directory](https://github.com/glvn5150/quant_gmm). The intentions of the model is thus, to have a unify GMM framework from sample models that has essential diagnostic plots for each model tested. The rep has its limitations, by which it:
- Moment conditions are hand-specified and model-dependent.
- Weighting matrices rely on simple identity or HAC estimators.
- Inference assumes standard asymptotic GMM conditions.
- Monte Carlo–based moments introduce simulation noise.
- No explicit identification or rank diagnostics are implemented.
- Diffusion models rely on discretized dynamics.
- Instrument selection is manual and model-specific.
- Numerical stability is handled pragmatically rather than optimally.
- The framework is not optimized for large-scale or production use.
# First Principles of GMM
GMM starts with a vector of moments condition via function $g_t(\theta) \in \mathbb{R}^k$ in k-dimensions and parameters $\theta \in \mathbb{R}^p$ in p-dimensions, such that
```math
\mathbb{E}[g_t(\theta)] = 0
```
Cochrane's Asset Pricing defines the price of an asset as $p = \mathbb{E}[m_t x_t]$, where $m$ is the discount factor of the asset price. The $g$ is defined as the expected value of the sample mean of $u_t$ errors such that : 
```math
g_T(b) = \mathbb{E}[u_t(b)] = \mathbb{E}[m_{t+1}(b) x_{t+1} - p_t]
```
The estimator is via matrix weight $W$: 
```math
\hat{\theta} = \arg\min_\theta \; \bar g_T(\theta)^\top W \bar g_T(\theta)
```
GMM has 1-step GMM and 2-step GMM, where the matrix weight $W=I$ for the 1-step, and the 2-step has $W = S^{-1}$ where $S = \mathbb{E}[g_t g_t^\top]$. The tests of significance of this statistical moments is via the J-statistic, which is:
```math
J = T \cdot \bar g_T(\hat\theta)^\top \hat W \bar g_T(\hat\theta)
```
