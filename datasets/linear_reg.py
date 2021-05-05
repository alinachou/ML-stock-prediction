import numpy as np
import yfinance as yf
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import data

def main():

    data.download_data()
    data_X = np.loadtxt("AAPL_features_train.txt", delimiter=',', skiprows=1)
    data_Y = np.loadtxt("AAPL_labels_train.txt", delimiter=',', skiprows=1)
    train_X, test_X, train_y,test_y = train_test_split(data_X,data_Y,test_size=0.25)
    run('linear_reg', linear_reg, 'Apple', train_X, train_y, test_X, test_y)


def linear_reg():
    regressor = LinearRegression()
    return regressor



def analyze_preds(model_name, regressor, test_X, test_y):
    predict_y = regressor.predict(test_X)
    pred_score = 'Prediction Score: ' + (str)(regressor.score(test_X, test_y))
    print(pred_score)
    error = 'Mean Squared Error: ' + (str)(mean_squared_error(test_y,predict_y))
    print(error)
    metrics = [pred_score, error]
    np.savetxt('{}_preds'.format(model_name), predict_y)
    np.savetxt('{}_metrics'.format(model_name), metrics,fmt='%s')
    return predict_y


def plot_pred(model_name, stock, test_X, test_y, predict_y):
    fig = plt.figure()
    ax = plt.axes()
    ax.grid()
    ax.set(xlabel='Close ($)',ylabel='Open ($)', title='{} Stock Prediction using Linear Regression'.format(stock))
    ax.plot(test_X[:,0],test_y)
    ax.plot(test_X[:,0],predict_y)
    fig.savefig('{}_plot.png'.format(model_name))

def run(model_name, model, stock, train_X, train_y, test_X, test_y):
    regressor = model()
    regressor.fit(train_X,train_y)
    predict_y = analyze_preds(model_name, regressor, test_X, test_y)
    plot_pred(model_name, stock, test_X, test_y, predict_y)


if __name__ == "__main__":
    main()