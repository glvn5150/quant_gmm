from setuptools import setup, find_packages

setup(
    name='gmm',
    version="0.1",
    packages = find_packages(),
    install_requires=[
        "numpy",
        "pandas",
        "scipy",
        "yfinance",
        "matplotlib",
        "scikit-learn",
        "seaborn",
        "plotly",
        "mstarpy",
        "investpy",
        "tradingeconomics",
        "bs4",
    ],
)