from pandas_datareader import data as pdr
import yfinance as yf
import pandas as pd
import util


"""
TOP 10 COMPANIES IN THE S&P 500 INDEX (2021)
--------------------------------------------
1.  Apple
2.  Microsoft
3.  Amazon
4.  Alphabet
5.  Facebook
6.  Tesla
7.  Berkshire Hathaway
8.  JPMorgan
9.  Visa
10. Johnson & Johnson
"""

TICKERS = {
    "Apple":              "AAPL",
    "Microsoft":          "MSFT",
    "Amazon":             "AMZN",
    "Alphabet":           "GOOGL",
    "Facebook":           "FB",
    "Tesla":              "TSLA",
    "Berkshire Hathaway": "BRK.A",
    "JP Morgan":          "JPM",
    "Visa":               "V",
    "Johnson & Johnson":  "JNJ"
}


def get_dates():
    start_date = util.generate_calendar("Select Start Date")
    end_date = util.generate_calendar("Select End Date")
    return start_date, end_date


if __name__ == '__main__':

    start, end = get_dates()
    stocks = util.select_stocks()
    features = ["Open", "High", "Low", "Volume"]
    labels = ["Close"]

    yf.pdr_override()
    for stock in stocks:
        data = pdr.get_data_yahoo(TICKERS[stock], start=start, end=end)
        # filename = TICKERS[stock] + "_" + start + "_to_" + end + ".txt"
        pd.DataFrame.to_csv(data, TICKERS[stock] + "_features_train.txt", columns=features, index=False)
        pd.DataFrame.to_csv(data, TICKERS[stock] + "_labels_train.txt", columns=labels, index=False)



