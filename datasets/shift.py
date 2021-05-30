import yfinance as yf
import pandas as pd
from datasets.linear_reg import *
from datasets.logistic_reg import *
from sklearn.linear_model import LinearRegression

def find_correlation(stock_a, stock_b):
    joint = pd.merge(stock_a, stock_b, left_index=True, right_index=True, how='outer').dropna()
    returns = (joint - joint.shift(1)) / joint.shift(1)
    # stock_a from yesterday
    X = returns.shift(1)['Open_y'].dropna()
    # stock_b for today
    Y = returns['Open_y'].dropna()



def main():
    btc = yf.download(tickers='BTC-USD', period='10y')
    spy = yf.download(tickers='spy', period='10y')

    joint = pd.merge(btc, spy, left_index=True, right_index=True, how='outer').dropna()
    returns = (joint - joint.shift(1)) / joint.shift(1)
    # btc from yesterday
    X = returns.shift(1)['Open_y'].dropna()
    # spy for today
    Y = returns['Open_y'].dropna()

    print(X)
    print(Y)




if __name__ == '__main__':
    main()
