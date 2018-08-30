import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt

from canary.generator.utils import X_metric_to_df, y_metric_to_df


def percentile(array, point):
    """
    Computes the value of empirical CDF for a given point, i.e. counts the values
    smaller than given point and divides by length of an array
    :param array: array of values from some distribution
    :param point: value to compute empirical CDF for
    """
    return np.mean([i <= point for i in array])


def plot_dists(X_dist, X_hist, y_hist=None):
    plt.figure()
    # important magic numbers
    width_1 = 1.6
    width_2 = 2
    height = 1.5

    if y_hist is not None:
        width_2 = 2.5
    ax0 = plt.axes([0, height, width_1, height], label='0')
    plt.setp(ax0.get_xticklabels(), visible=False)
    plt.plot(X_dist)
    plt.xlim(-0.5, len(X_dist) - 0.5)
    plt.xlabel('Date')
    plt.ylabel('Distance value')
    plt.legend(X_dist.columns)

    ax1 = plt.axes([0, 0, width_2, height], label='1')
    df = X_metric_to_df(X_hist)
    sns.heatmap(df, cmap="YlGnBu")
    if y_hist is not None:
        y_df = y_metric_to_df(y_hist)
        sns.heatmap(df, mask=1 - y_df[0], cmap='YlOrRd')
    plt.xlabel('Date')
    plt.ylabel('Bucket')
    plt.subplots_adjust(hspace=.0)
    plt.show()


def plot_preds(X_test, y_true, y_pred):
    df = X_metric_to_df(X_test)
    y_df = y_metric_to_df(y_true)
    y_pred = y_metric_to_df(y_pred)
    a4_dims = (15, 6)
    plt.subplots(figsize=a4_dims)
    sns.heatmap(df.iloc, cmap="YlGnBu")
    sns.heatmap(df.iloc, mask=1 - y_df[0], cmap='YlOrRd')
    plt.xlabel('Date')
    plt.ylabel('Bucket')
    plt.title('TRUE')
    plt.show()

    a4_dims = (15, 6)
    plt.subplots(figsize=a4_dims)
    sns.heatmap(df.iloc, cmap="YlGnBu")
    sns.heatmap(df.iloc, mask=1 - y_pred.set_index(y_df.index)[0], cmap='YlOrRd')
    plt.xlabel('Date')
    plt.ylabel('Bucket')
    plt.title('PRED')
    plt.show()
