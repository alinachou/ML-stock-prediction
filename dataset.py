from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import util


class Dataset:
    def __init__(self, stock, period=21):
        self.period = period
        self.predictions = None
        self.stock = stock

        self.features, self.labels = util.load_dataset("datasets/" + self.stock + "_data.txt")
        self.num_examples = self.features.shape[0]

        self.sma = self.generate_simple_moving_average(self.period)
        self.bollinger_bands, self.bollinger_band_width = self.generate_bollinger_bands_and_width(self.period)

        temp_features = np.hstack((self.features[self.period:], self.bollinger_bands[0], self.bollinger_bands[1], self.bollinger_bands[2]))
        temp_labels = self.labels[self.period:]

        features = train_test_split(temp_features, train_size=0.8, test_size=0.2, shuffle=False)
        labels = train_test_split(temp_labels, train_size=0.8, test_size=0.2, shuffle=False)

        self.train_features = features[0]
        self.train_labels = labels[0]
        self.num_train_examples = self.train_features.shape[0]

        self.test_features = features[1]
        self.test_labels = labels[1]
        self.num_test_examples = self.test_features.shape[0]

        self.return_labels = False

    def returns(self):
        train_l, test_l = self.train_labels, self.test_labels

        self.train_features = np.delete(self.train_features, self.num_train_examples - 1, 0)
        self.train_labels = np.asarray([((train_l[i] - train_l[i - 1]) / train_l[i - 1]) * 100 for i in range(1, len(train_l))])
        self.num_train_examples = self.train_features.shape[0]

        self.test_features = np.delete(self.test_features, self.num_test_examples - 1, 0)
        self.test_labels = np.asarray([((test_l[i] - test_l[i - 1]) / test_l[i - 1]) * 100 for i in range(1, len(test_l))])
        self.num_test_examples = self.test_features.shape[0]

        self.return_labels = True

    def generate_simple_moving_average(self, period):
        averages = []

        for i in range(period, self.num_examples):
            averages.append(np.mean(self.labels[i - self.period: i]))

        self.sma = np.asarray(averages)

        return self.sma

    def generate_bollinger_bands_and_width(self, period, multiplier=None):
        """
        Bollinger Bands are a type of price envelope developed by John Bollinger.
        (Price envelops define upper and lower price range levels.) Bollinger Bands are envelopes plotted at
        a standard deviation level above and below a simple moving average of the price. Because the distance
        of the bands is based on standard deviation, they adjust to volatility swings in the underlying price.

        Args:
            period:

            multiplier:
                Short term: 10 day moving average, bands at 1.5 standard deviations. (1.5x Std Dev +/- SMA)
                Medium Term: 20 day moving average, bands at 2 standard deviations.
                Long Term: 50 day moving average, bands at 2.5 standard deviations.

        Returns: a tuple consisting of the Lower Band, the SMA, and the Upper Band

        """

        std_devs = np.asarray([np.std(self.labels[i - self.period: i]) for i in range(self.period, self.num_examples)])

        if multiplier is None:
            if self.period <= 10:
                multiplier = 1.5
            elif self.period <= 20:
                multiplier = 2
            else:
                multiplier = 2.5

        lower_band = (self.sma - multiplier * std_devs).reshape((-1, 1))
        upper_band = (self.sma + multiplier * std_devs).reshape((-1, 1))

        self.bollinger_bands = (lower_band, self.sma.reshape((-1, 1)), upper_band)
        self.bollinger_band_width = (upper_band - lower_band) / self.sma

        return self.bollinger_bands, self.bollinger_band_width

    def linear_regression(self):
        """
        Straightforward method that utilizes sklearn to calculate predictions on
        testing dataset using linear regression model
        """

        reg = LinearRegression().fit(self.train_features, self.train_labels)
        self.predictions = reg.predict(self.test_features)

    def logistic_regression(self):

        train_labels = np.asarray([1 if label >= 0 else 0 for label in self.train_labels])
        self.test_labels = np.asarray([1 if label >= 0 else 0 for label in self.test_labels])
        reg = LogisticRegression().fit(self.train_features, train_labels)
        self.predictions = reg.predict(self.test_features)

    def mlp(self, classification=False):

        self.train_labels = np.sign(self.train_labels[1:]) if classification else self.train_labels[1:]
        self.test_labels = np.sign(self.test_labels[1:].reshape((-1, 1))) if classification else\
            self.test_labels[1:].reshape((-1, 1))

        """
        stock_train_features, stock_test_features = [], []

        all_stocks = util.all_stocks()
        for stock in all_stocks:

            ds = Dataset(stock)
            ds.returns()
            stock_train_features.append(ds.train_labels.reshape(-1, 1))
            stock_test_features.append(ds.test_labels.reshape(-1, 1))
            print(stock)

        stock_train_features, stock_test_features = tuple(stock_train_features), tuple(stock_test_features)
        train_matrix, test_matrix = np.sign(np.hstack(stock_train_features)), np.sign(np.hstack(stock_test_features))
        train_matrix, test_matrix = np.delete(train_matrix, train_matrix.shape[0] - 1, 0), np.delete(test_matrix, test_matrix.shape[0] - 1, 0)
        
        
        np.save("Train_Matrix.npy", train_matrix)
        np.save("Test_Matrix.npy", test_matrix)
        """

        train_matrix = np.load("Train_Matrix.npy")
        test_matrix = np.load("Test_Matrix.npy")

        if classification:
            reg = MLPClassifier(hidden_layer_sizes=100, activation="logistic", solver="adam", shuffle=False)
        else:
            reg = MLPRegressor(hidden_layer_sizes=700, activation='relu', solver='adam')

        reg.fit(train_matrix, self.train_labels)
        self.predictions = reg.predict(test_matrix)

    def plot(self, predictions=False, bollinger_bands=False, filename=None):
        """
        The plot method will plot the dataset upon which the class was initialized.
        It also has the ability plot the bollinger bands of the corresponding dataset
        as well as the predictions if they have been calculated. Finally, passing a filename
        will allow for the resulting plot to be saved to that file.

        Args:

            predictions: Boolean indicating whether predictions should be plotted if present

            bollinger_bands: Boolean indicating whether bollinger_bands should be plotted

            filename: Filename where resulting plot should be saved

        Returns: Nothing
        """

        array = [i for i in range(self.test_labels.shape[0])]
        plt.plot(array, self.test_labels)
        lines = ["Actual"]

        if predictions and self.predictions is not None:
            plt.plot(array, self.predictions)
            lines.append("Predicted")

        if bollinger_bands:
            for i in range(4, 7):
                plt.plot(array, self.test_features[:, i])
            lines.extend(["Lower Band", str(self.period) + " Period SMA", "Upper Band"])

        if self.return_labels:
            plt.title(self.stock + " Returns")
            plt.ylabel("Returns")
        else:
            plt.title(self.stock + " Closing Price")
            plt.ylabel("Price")

        plt.legend(lines)
        plt.xlabel("Time")
        plt.savefig(self.stock + ".png") if filename is None else plt.savefig(filename)


if __name__ == '__main__':
    stocks = ["AAPL", "MSFT", "AMZN", "GOOGL", "FB", "TSLA", "BRK-A", "JPM", "V", "JNJ"]

    """
    output = pd.read_pickle("expected_returns.pkl")


    returns = np.load("returns.npy")
    df = pd.DataFrame(returns, columns=stocks)
    df.to_pickle("expected_returns.pkl")

    
    returns = []
    for stock in stocks:
        ds = Dataset(stock)
        ds.returns()
        ds.mlp()
        returns.append(ds.predictions.reshape(-1, 1))
        ds.plot(predictions=True)
    returns = tuple(returns)
    returns = np.hstack(returns)
    np.savetxt("returns.txt", returns)
    np.save("returns.npy", returns)

    prediction = np.sign(ds.predictions)
    actual = np.sign(ds.test_labels)

    true_positive, true_negative = 0, 0
    false_positive, false_negative = 0, 0
    positives, negatives = 0, 0

    for i in range(actual.shape[0]):
        if actual[i] == 1:
            if prediction[i] == actual[i]:
                true_positive += 1
            else:
                false_negative += 1
        if actual[i] == -1:
            if prediction[i] == actual[i]:
                true_negative += 1
            else:
                false_positive += 1

    print("True Positive = " + str(true_positive/(true_positive + false_negative)))
    print("True Negative = " + str(true_negative / (true_negative + false_positive)))
    print("False Positive = " + str(false_positive / (false_positive + true_negative)))
    print("False Negative = " + str(false_negative / (false_negative + true_positive)))
    print()
    print("Accuracy = " + str((true_positive + true_negative) / (true_positive + true_negative + false_positive + false_negative)))
    """












