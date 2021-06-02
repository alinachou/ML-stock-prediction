import yfinance as yf
import pandas as pd
from datasets.linear_reg import *
from datasets.logistic_reg import *
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
import stock_portfolio.data as data
import matplotlib.pyplot as plt


TICKERS = ["AAPL", "MSFT", "AMZN", "GOOGL", "FB", "TSLA", "BRK-A", "JPM", "V", "JNJ"]


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
    correlation_all_stock = []
    for i in range(len(TICKERS)):
        print(TICKERS[i])
        correlation_lst = []
        stock_a = yf.download(tickers=TICKERS[i], start="2014-01-01", end="2021-05-29")
        for j in range(len(TICKERS)):
            if i == j and j != len(TICKERS) - 1:
                j += 1
            elif i == j and j == len(TICKERS) - 1:
                break
            stock_b = yf.download(tickers=TICKERS[j], start="2014-01-01", end="2021-05-29")
            correlation, X, Y = find_correlation(stock_a, stock_b)
            # correlation_lst.append((TICKERS[i], TICKERS[j], correlation))
            correlation_lst.append((X, Y, TICKERS[i], TICKERS[j], correlation))
        # Getting the tuple with highest correlation
        max_correlation = sorted(correlation_lst, key=lambda x: x[-1])[-1]
        correlation_all_stock.append(max_correlation)

    """
    # Return most correlated two stocks
    print(max_correlation[-1], max_correlation[2], max_correlation[3])
    stock_a_name = max_correlation[2]
    stock_b_name = max_correlation[3]
    return max_correlation[0], max_correlation[1], stock_a_name, stock_b_name
    """
    return correlation_all_stock


def linear_predict(stock_a, stock_b, stock_b_name):
    stock_a = np.asarray(stock_a).reshape(-1, 1)
    stock_b = np.asarray(stock_b)
    clf = LinearRegression().fit(stock_a, stock_b)
    y_pred = clf.predict(stock_a)
    # Convert to dataframe:
    df_linear = pd.DataFrame(y_pred, columns=['Linear_Reg_Returns'])
    base = yf.download(tickers=stock_b_name, start="2014-01-01", end="2021-05-29").dropna()['Open']
    # y_pred = [base[i] + base[i] * y_pred[i] for i in range(len(y_pred))]
    score = clf.score(stock_a, stock_b)

    return y_pred, df_linear, score


def logistic_predict(stock_a, stock_b):
    stock_a = np.asarray(stock_a).reshape(-1, 1)
    stock_b = np.asarray(stock_b)
    for i in range(stock_a.shape[0]):
        stock_a[i] = 1 if stock_a[i] > 0 else 0
    for i in range(stock_b.shape[0]):
        stock_b[i] = 1 if stock_b[i] > 0 else 0

    clf = LogisticRegression().fit(stock_a, stock_b)
    y_pred = clf.predict(stock_a)
    # Convert to dataframe:
    df_logistic = pd.DataFrame(y_pred, columns=['Linear_Reg_Returns'])
    score = clf.score(stock_a, stock_b)

    return y_pred, df_logistic, score


def plot(stock_b, y_pred, time, stock_a_name, stock_b_name):
    plt.plot(time[2:], stock_b)
    # plt.plot(time, stock_a)
    plt.plot(time[2:], y_pred)
    plt.xlabel("Date")
    plt.ylabel("Returns")
    plt.title("Using " + stock_a_name + " to Predict " + stock_b_name)
    plt.savefig(stock_b_name + "_shifted_correlation.png")
    plt.show()


def main():
    # stock_a, stock_b, stock_a_name, stock_b_name = compare_correlation()
    correlation_all_stock = compare_correlation()
    df_lst = []
    for stock in correlation_all_stock:
        stock_a = stock[0]
        stock_b = stock[1]
        stock_a_name = stock[2]
        stock_b_name = stock[3]

        y_pred_linear, df_linear, score_linear = linear_predict(stock_a, stock_b, stock_b_name)
        y_pred_logistic, df_logistic, score_logistic = logistic_predict(stock_a, stock_b)

        y_pred_linear = np.asarray(y_pred_linear).reshape(-1, 1)
        print(y_pred_linear)
        print(y_pred_linear.shape)
        df_lst.append(y_pred_linear)

        plot(stock_b, y_pred_linear, yf.download(tickers=stock_b_name, start="2014-01-01", end="2021-05-29").index, stock_a_name, stock_b_name)
        print(stock_a_name, stock_b_name, "Linear Regression: ", y_pred_linear, score_linear)
        print("Logistic Regression: ", y_pred_logistic, score_logistic)
        
    df = pd.DataFrame(np.hstack(tuple(df_lst)),
                          columns=["AAPL", "MSFT", "AMZN", "GOOGL", "FB", "TSLA", "BRK-A", "JPM", "V", "JNJ"])
    print(df)


if __name__ == '__main__':
    main()
