import numpy as np
from scipy import stats

ACTUAL_RETURNS = np.load("actual_returns.npy")
PREDICTED_RETURNS = np.load("predicted_returns.npy")
INITIAL_PORTFOLIO_VALUE = 1000000
NUM_DAYS = ACTUAL_RETURNS.shape[0]
NUM_TRIALS = 1000


def hypothesis_test(portfolio, num_trials):
    value = portfolio
    for day in range(NUM_DAYS):
        stock = select_stock(day, criteria="predicted")
        value = update_portfolio(day, stock, value)

    profits_predicted = value
    print(profits_predicted)

    values = []
    for i in range(num_trials):
        value = portfolio
        for day in range(NUM_DAYS):
            stock = select_stock(day, criteria="random")
            value = update_portfolio(day, stock, value)
        values.append(value)
    values = np.asarray(values)

    average_random = np.average(values)
    std_dev_random = np.std(values)

    test_statistic = (profits_predicted - average_random) / (std_dev_random / np.sqrt(num_trials))
    p_value = stats.t.sf(test_statistic, num_trials - 1)
    return p_value


def select_stock(day, criteria):
    if criteria == "actual":
        return np.argmax(ACTUAL_RETURNS[day])
    elif criteria == "predicted":
        return np.argmax(PREDICTED_RETURNS[day])
    else:
        return np.random.randint(0, 9)


def update_portfolio(day, stock, portfolio):
    returns = ACTUAL_RETURNS[day][stock]
    multiplier = (1 + returns / 100)
    return portfolio * multiplier


if __name__ == '__main__':

    hypothesis_test(INITIAL_PORTFOLIO_VALUE, NUM_TRIALS)








