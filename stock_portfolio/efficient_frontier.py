import numpy as np
from matplotlib import axes
from matplotlib import pyplot as plt
from pypfopt import risk_models
from pypfopt import expected_returns
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt import plotting
from pandas_datareader import data as pdr
import yfinance as yf

yf.pdr_override()
import pandas as pd
stocks = "AAPL MSFT AMZN GOOGL FB JPM JNJ"


def plot_ef(ef):
    fig, ax = plt.subplots()
    plotting.plot_efficient_frontier(ef, ax=ax)
    ef.add_constraint(lambda w: w[0]+w[1]+w[2]+w[3]+w[4]+w[5]+w[6] == 1)
    weights = ef.max_sharpe()
    optimal, volatility, _ = ef.portfolio_performance(verbose=True)
    ax.plot(volatility, optimal, marker="*", label="optimal")
    ax.legend()
    plt.savefig("ef.png")
    plt.show()

def plot_covariance(S):
    co_v = plotting.plot_covariance(S, show_tickers=True, show=True)
    axes.Axes.plot(co_v)
    plt.savefig("covariance_fig.png")
    plt.show()  

def random_portfolios(mu, S):
    ef = EfficientFrontier(mu, S)
    fig, ax = plt.subplots()
    plotting.plot_efficient_frontier(ef, ax=ax, show_assets=False)

    # Find the tangency portfolio
    ef.max_sharpe()
    ret_tangent, std_tangent, _ = ef.portfolio_performance()
    ax.scatter(std_tangent, ret_tangent, marker="*", s=100, c="r", label="Max Sharpe")

    # Generate random portfolios
    n_samples = 10000
    w = np.random.dirichlet(np.ones(len(mu)), n_samples)
    rets = w.dot(mu)
    stds = np.sqrt(np.diag(w @ S @ w.T))
    sharpes = rets / stds
    ax.scatter(stds, rets, marker=".", c=sharpes, cmap="viridis_r")

    # Output
    ax.set_title("Efficient Frontier with random portfolios")
    ax.legend()
    plt.tight_layout()
    plt.savefig("ef_scatter.png", dpi=200)
    plt.show()

df = pdr.get_data_yahoo(tickers=stocks, start="2017-01-01", end="2017-04-30")['Adj Close']
returns = df.pct_change().dropna()
mu = expected_returns.mean_historical_return(df)
S = risk_models.sample_cov(df)
ef = EfficientFrontier(mu, S)
plot_ef(ef)
plot_covariance(S)
random_portfolios(mu,S)
cleaned_weights = ef.clean_weights()
ef.save_weights_to_file("weights.txt")