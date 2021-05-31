import yfinance as yf
import pandas as pd
from datasets.linear_reg import *
from datasets.logistic_reg import *
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
import stock_portfolio.data as data
import matplotlib.pyplot as plt

TICKERS = ["AAPL", "MSFT", "AMZN", "GOOGL", "FB", "TSLA", "JPM", "V", "JNJ", "AAL", "C"]


def find_correlation(stock_a, stock_b):
    joint = pd.merge(stock_a, stock_b, left_index=True, right_index=True, how='outer').dropna()
    returns = (joint - joint.shift(1)) / joint.shift(1)

    # stock_a from yesterday
    X = returns['Open_x'].dropna().tolist()
    # stock_b for today
    Y = returns.shift(1)['Open_y'].dropna().tolist()

    # Getting the data of the same size in X and Y vectors
    size = min(len(X), len(Y))
    X = X[:size]
    Y = Y[:size]

    correlation = np.abs(np.corrcoef(X, Y)[0, 1])
    return correlation, X, Y


def compare_correlation():
    correlation_lst = []
    for i in range(len(TICKERS)):
        stock_a = yf.download(tickers=TICKERS[i], period='10y')
        for j in range(i, len(TICKERS)):
            stock_b = yf.download(tickers=TICKERS[j], period='10y')
            correlation, X, Y = find_correlation(stock_a, stock_b)
            correlation_lst.append((X, Y, TICKERS[i], TICKERS[j], correlation))

    # Getting the tuple with highest correlation
    max_correlation = sorted(correlation_lst, key=lambda x: x[-1])[0]

    # Return most correlated two stocks
    print(max_correlation[-1], max_correlation[2], max_correlation[3])
    stock_a_name = max_correlation[2]
    stock_b_name = max_correlation[3]
    return max_correlation[0], max_correlation[1], stock_a_name, stock_b_name


def linear_predict(stock_a, stock_b, stock_b_name):
    stock_a = np.asarray(stock_a).reshape(-1, 1)
    stock_b = np.asarray(stock_b)
    clf = LinearRegression().fit(stock_a, stock_b)
    y_pred = clf.predict(stock_a)
    base = yf.download(tickers=stock_b_name, period='10y').dropna()['Open']
    #y_pred = [base[i] + base[i] * y_pred[i] for i in range(len(y_pred))]
    score = clf.score(stock_a, stock_b)

    return y_pred, score

def plot(stock_b, y_pred, time):
    plt.plot(time[2:], stock_b)
    # plt.plot(time, stock_a)
    plt.plot(time[2:], y_pred)
    plt.xlabel("Date")
    plt.ylabel("Returns")
    plt.legend
    plt.show()


"""
def logistic_predict(stock_a, stock_b):
    stock_a = np.asarray(stock_a).reshape(-1, 1)
    stock_b = np.asarray(stock_b)
    clf = LogisticRegression().fit(stock_a, stock_b)
    y_pred = clf.predict(stock_a)
    # score = clf.score(stock_a, stock_b)
    print(y_pred)
"""


def main():
    stock_a, stock_b, stock_a_name, stock_b_name = compare_correlation()
    y_pred, score = linear_predict(stock_a, stock_b, stock_b_name)
    # plot(yf.download(tickers=stock_b_name, period='10y').dropna()['Open'], y_pred, yf.download(tickers=stock_b_name, period='10y').index)
    plot(stock_b, y_pred, yf.download(tickers=stock_b_name, period='10y').index)
    print(y_pred, score)


if __name__ == '__main__':
    main()