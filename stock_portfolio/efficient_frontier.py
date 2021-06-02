import numpy as np
from matplotlib import axes
from matplotlib import pyplot as plt
from pypfopt import risk_models
from pypfopt import expected_returns
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt import plotting
from pypfopt import objective_functions
from pypfopt.discrete_allocation import DiscreteAllocation, get_latest_prices
from pandas_datareader import data as pdr
import yfinance as yf

yf.pdr_override()
import pandas as pd
stocks = "AAPL MSFT AMZN GOOGL FB TSLA BRK-A JPM V JNJ"


def plot_ef(ef, method=None):
    fig, ax = plt.subplots()
    plotting.plot_efficient_frontier(ef, ax=ax)
    ef.add_constraint(lambda w: w[0]+w[1]+w[2]+w[3]+w[4]+w[5]+w[6]+w[7]+w[8]+w[9] == 1)
    weights = ef.max_sharpe()
    cleaned_weights = ef.clean_weights()
    print(cleaned_weights)
    optimal, volatility, _ = ef.portfolio_performance(verbose=True)
    ax.plot(volatility, optimal, marker="*", label="optimal")
    ax.legend()
    plt.savefig("ef_{}.png".format(method))
    plt.show()

def plot_covariance(S):
    co_v = plotting.plot_covariance(S, show_tickers=True, show=True, plot_correlation=True)
    axes.Axes.plot(co_v)
    plt.savefig("correlation_fig.png")
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
#How many shares of each asset you should purchase
def allocate(df, ef):
    ef.add_objective(objective_functions.L2_reg, gamma=0.1)
    w = ef.max_sharpe()
    latest_prices = get_latest_prices(df)
    da = DiscreteAllocation(w, latest_prices, total_portfolio_value=20000)
    allocation, leftover = da.lp_portfolio()
    print(allocation)

def run(method):
    if method == "NN":
        df = pd.read_pickle("../NN/expected_returns_NN.pkl")
    else:
        #2019-12-12 to now
        df = pdr.get_data_yahoo(tickers=stocks, start="2014-01-01", end="2021-05-29")['Adj Close']
        returns = df.pct_change().dropna()
    mu = expected_returns.mean_historical_return(df)
    S = risk_models.sample_cov(df)
    ef = EfficientFrontier(mu, S)
    plot_ef(ef, method=method)
# plot_covariance(S)
# random_portfolios(mu,S)
if __name__ == '__main__':
    run("NN")
