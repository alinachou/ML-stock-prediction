import yfinance as yf
import pandas as pd
from datasets.linear_reg import *
from datasets.logistic_reg import *
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
import stock_portfolio.data as data

TICKERS = ["AAPL", "MSFT", "AMZN", "GOOGL", "FB", "TSLA", "JPM", "V", "JNJ"]


def find_correlation(stock_a, stock_b):
    joint = pd.merge(stock_a, stock_b, left_index=True, right_index=True, how='outer').dropna()
    returns = (joint - joint.shift(1)) / joint.shift(1)

    # stock_a from yesterday
    X = returns.shift(1)['Open_y'].dropna().tolist()
    # stock_b for today
    Y = returns['Open_y'].dropna().tolist()

    # Getting the data of the same size in X and Y vectors
    Y = Y[1:]

    correlation = np.abs(np.corrcoef(X, Y)[0, 1])
    return correlation, X, Y


def compare_correlation():
    correlation_lst = []
    for i in range(len(TICKERS)):
        stock_a = yf.download(tickers=TICKERS[i], period='10y')
        for j in range(i, len(TICKERS)):
            stock_b = yf.download(tickers=TICKERS[j], period='10y')
            correlation, X, Y = find_correlation(stock_a, stock_b)
            correlation_lst.append((X, Y, correlation))

    # Getting the tuple with highest correlation
    max_correlation = sorted(correlation_lst, key=lambda x: x[-1])[0]

    # Return most correlated two stocks
    print(max_correlation[2])
    return max_correlation[0], max_correlation[1]


"""
def logistic_predict(stock_a, stock_b):
    stock_a = np.asarray(stock_a).reshape(-1, 1)
    stock_b = np.asarray(stock_b)
    clf = LogisticRegression().fit(stock_a, stock_b)
    y_pred = clf.predict(stock_a)
    # score = clf.score(stock_a, stock_b)
    print(y_pred)
"""


def linear_predict(stock_a, stock_b):
    stock_a = np.asarray(stock_a).reshape(-1, 1)
    stock_b = np.asarray(stock_b)
    clf = LinearRegression().fit(stock_a, stock_b)
    y_pred = clf.predict(stock_a)
    score = clf.score(stock_a, stock_b)
    return y_pred, score


def main():
    stock_a, stock_b = compare_correlation()
    y_pred, score = linear_predict(stock_a, stock_b)
    print(y_pred, score)


if __name__ == '__main__':
    main()
