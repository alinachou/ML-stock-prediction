import pandas_datareader as pdr
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression


def logistic_regress():
    df = pdr.get_data_yahoo("MSFT", "2010-11-01", "2020-11-01")
    df["Diff"] = df.Close.diff()
    df["SMA_2"] = df.Close.rolling(2).mean()
    df["Force_Index"] = df["Close"] * df["Volume"]
    df["y"] = df["Diff"].apply(lambda x: 1 if x > 0 else 0).shift(-1)
    df = df.drop(["Open", "High", "Low", "Close", "Volume", "Diff", "Adj Close"], axis=1).dropna()

    X = df.drop(["y"], axis=1).values
    y = df["y"].values
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    clf = LogisticRegression()
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)

    print(accuracy_score(y_test, y_pred))


def main():
    logistic_regress()


if __name__ == '__main__':
    main()
